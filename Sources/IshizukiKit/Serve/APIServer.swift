// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public final class APIServer: @unchecked Sendable {
  /// The pack directory or the `.gguf` this server is serving. Either can be handed to
  /// `BonsaiModel` and `ChatTemplate`, which is all this needs it for.
  public private(set) var modelPath: URL
  public private(set) var template: ChatTemplate
  public private(set) var modelName: String
  public var defaultThinking: Bool
  public var samplingOptions: SamplingOptions
  public var kvConfig: KVCacheConfig
  public let politeness: Politeness.Level
  public let ropeScaling: RopeScaling
  public let residency: ResidencyManager
  public let sessions: SessionCache
  public let prefixStore: PrefixStore?
  public private(set) var budget: MemoryBudget

  /// Every pack this server could serve. A request naming one of these swaps to it.
  public private(set) var catalog: ModelCatalog
  /// Where `catalog` is rescanned from before anything reads it, so a pack quantized or pulled
  /// after this server started is still found. Empty means the catalog passed to `init` is all
  /// there is, which is what a caller that never scanned a filesystem (a test, say) wants.
  private let catalogRoots: [URL]
  public let stats = ServeStats()

  private var loaded: BonsaiModel?
  /// Whether a pack arrives with its vision tower already read, rather than on the first image.
  private let hot: Bool
  let generationQueue = DispatchQueue(label: "bonsai.generate")
  private var server: HTTPServer?
  public var log: (@Sendable (String) -> Void)?

  /// The high-water mark a readout reports, kept here so every reader sees the same peak.
  var peakHeld = 0

  public init(
    directory: URL, modelName: String = "ternary-bonsai-2-27b",
    thinking: Bool = true, samplingOptions: SamplingOptions = SamplingOptions(),
    kvConfig: KVCacheConfig = KVCacheConfig(),
    residency: ResidencyManager.Options = ResidencyManager.Options(),
    politeness: Politeness.Level = .normal,
    ropeScaling: RopeScaling = .none,
    budget: MemoryBudget? = nil,
    prefixStore: PrefixStore? = nil,
    catalog: ModelCatalog = ModelCatalog(entries: []),
    catalogRoots: [URL] = [],
    preload: Bool = true,
    hot: Bool = false
  ) throws {
    self.hot = hot
    self.politeness = politeness
    self.ropeScaling = ropeScaling
    self.catalog = catalog
    self.catalogRoots = catalogRoots
    self.modelPath = directory
    self.template = try ChatTemplate(path: directory)
    self.modelName = modelName
    self.defaultThinking = thinking
    self.samplingOptions = samplingOptions
    self.kvConfig = kvConfig
    let budget =
      budget
      ?? MemoryBudget(
        kvBits: kvConfig.bits,
        maxContextTokens: ropeScaling.effectiveContext,
        weights: StreamedPlan.residentBytes(in: directory) ?? MemoryBudget.defaultWeights)
    self.budget = budget
    self.sessions = SessionCache(capacity: budget.tier.slots)
    self.prefixStore = prefixStore
    // The pack's own directory names the weights it was built from, which is what an archive
    // has to agree with before it can be read back.
    self.sessions.setStore(prefixStore, modelID: directory.lastPathComponent)
    self.residency = ResidencyManager(options: residency)
    budget.apply()

    if preload { _ = try model() }

    self.residency.onIdle = { [weak self] in
      guard let self else { return }
      self.generationQueue.async {
        // Prefix caches survive idle: dropping them costs a full re-prefill on the next
        // turn. The reclaimable buffer pool has already been freed by the residency timer.
        self.log?("idle: released buffer pool (\(ResidencyManager.describeMemory()))")
      }
    }
    self.residency.onEvict = { [weak self] in
      guard let self else { return }
      self.generationQueue.async {
        // Archived before the pool is dropped, so the next turn of a conversation that is
        // merely idle does not pay for a full re-prefill.
        self.sessions.persistAll()
        self.sessions.evict()
        self.loaded = nil
        Memory.clearCache()
        self.applyBudget(budget.reset())
        self.log?("idle: unloaded model (\(ResidencyManager.describeMemory()))")
      }
    }
  }

  private func applyBudget(_ step: MemoryBudget.Step?) {
    guard let step else { return }
    budget.apply()
    sessions.setCapacity(step.tier.slots)
    sessions.setByteLimit(step.tier.kvBytes(bytesPerToken: budget.bytesPerToken))
    log?("budget: \(step.summary)")
  }

  /// The pack currently loaded, or the one that would be if a request arrived.
  public var activeModelID: String { modelName }

  public var isLoaded: Bool { loaded != nil }

  /// The loaded pack's weights, for a readout that wants to ask them something. Never loads
  /// one: a server that has not been asked for a token reports nothing rather than paying for
  /// a model to say so.
  public var loadedStore: WeightStore? { loaded?.store }

  /// Switches the server to another pack from the catalog. The one in memory is dropped first,
  /// because two 17 GB packs do not sit side by side on this hardware.
  ///
  /// Everything derived from the pack goes with it: the chat template, the memory budget sized
  /// from its weights, and the prefix archives, which are keyed per model and so are simply no
  /// longer matched rather than discarded.
  public func activate(_ id: String) throws {
    guard id != modelName else { return }
    refreshCatalog()
    guard let entry = catalog[id] else {
      throw BonsaiError.missingComponent(
        "no pack named '\(id)'; this server offers "
          + catalog.entries.map(\.id).joined(separator: ", "))
    }

    let template = try ChatTemplate(path: entry.url)

    sessions.persistAll()
    sessions.evict()
    loaded = nil
    Memory.clearCache()

    modelPath = entry.url
    modelName = entry.id
    self.template = template
    budget = MemoryBudget(
      kvBits: kvConfig.bits,
      maxContextTokens: ropeScaling.effectiveContext,
      weights: StreamedPlan.residentBytes(in: entry.url)
        ?? (entry.byteCount > 0 ? entry.byteCount : MemoryBudget.defaultWeights))
    budget.apply()
    sessions.setCapacity(budget.tier.slots)
    sessions.setByteLimit(budget.tier.kvBytes(bytesPerToken: budget.bytesPerToken))
    sessions.setStore(prefixStore, modelID: entry.id)
    log?("model: switched to \(entry.id)")
  }

  /// Honours a request that names a pack other than the loaded one. Names that match nothing in
  /// the catalog are ignored, so a client sending its own alias keeps working.
  func activateIfRequested(_ requested: String?) {
    guard let requested, !requested.isEmpty, requested != modelName else { return }
    refreshCatalog()
    guard catalog[requested] != nil else { return }
    do { try activate(requested) } catch { log?("model: \(error)") }
  }

  /// Rescans `catalogRoots`, so a pack that appeared on disk after this server started is
  /// listed and can be switched to without a restart. A no-op when nothing was given to scan.
  private func refreshCatalog() {
    guard !catalogRoots.isEmpty else { return }
    catalog = ModelCatalog.discover(in: catalogRoots)
  }

  @discardableResult
  public func model() throws -> BonsaiModel {
    if let loaded { return loaded }
    let start = Date()
    let model = try BonsaiModel(path: modelPath, ropeScaling: ropeScaling, hot: hot)
    loaded = model
    log?(String(format: "loaded model in %.1fs", -start.timeIntervalSinceNow))
    if let experts = model.store.expertTraffic {
      log?(
        "experts: streamed, \(experts.slots) slots a layer across \(experts.layers) layers, "
          + "\(experts.heldBytes >> 30) GB held, prefill chunk "
          + "\(Politeness.prefillChunk(for: politeness, default: BonsaiRuntime.streamedPrefillChunk))")
    }
    return model
  }

  public func listen(port: UInt16) throws {
    let server = try HTTPServer(port: port) { [weak self] request, writer in
      guard let self else { return }
      let id = self.stats.enqueue(for: request.path)
      self.generationQueue.async { self.route(request, writer, id) }
    }
    self.server = server
    server.start()
    residency.startMonitoring(on: generationQueue)
  }

  public func stop() {
    server?.stop()
    server = nil
    residency.stopMonitoring()
    residency.unwire()
    sessions.persistAll()
  }

  private func route(_ request: HTTPRequest, _ writer: ResponseWriter, _ id: Int?) {
    defer { stats.end(id) }
    if writer.isCancelled {
      stats.cancel(id)
      if let id { log?("#\(id) cancelled while queued") }
      writer.finish()
      return
    }
    let path = request.path.split(separator: "?").first.map(String.init) ?? request.path

    switch (request.method, path) {
    case ("OPTIONS", _):
      writer.send(json: [:])
    case ("GET", "/health"):
      writer.send(json: ["status": "ok", "model": modelName])
    case ("GET", "/v1/models"):
      // Every pack on the machine, so a client can offer the choice rather than be told one.
      refreshCatalog()
      var data: [[String: Any]] = catalog.entries.map { entry in
        [
          "id": entry.id, "object": "model", "owned_by": "prism-ml",
          "loaded": entry.id == modelName && loaded != nil,
          "context_window": entry.contextTokens,
          "quantization": entry.quantization,
          "size_bytes": entry.byteCount,
          "vision": entry.hasVision,
          "mtp": entry.hasMTP,
        ]
      }
      if !data.contains(where: { $0["id"] as? String == modelName }) {
        data.insert(
          [
            "id": modelName, "object": "model", "owned_by": "prism-ml",
            "loaded": loaded != nil,
          ], at: 0)
      }
      writer.send(json: ["object": "list", "data": data])
    case ("POST", "/v1/chat/completions"):
      handleOpenAI(request, writer, id)
    case ("POST", "/v1/messages"):
      handleAnthropic(request, writer, id)
    case ("POST", "/v1/messages/count_tokens"):
      handleCountTokens(request, writer)
    default:
      writer.sendError(
        status: 404, type: "not_found_error", message: "no route for \(path)")
    }
  }

  struct Request {
    var messages: [ChatMessage]
    var tools: [[String: Any]]?
    var maxTokens: Int
    var temperature: Float?
    var stream: Bool
    var thinking: Bool
    var images: [ProcessedImage]
    var responseSchema: [String: Any]?
    /// What the client asked to talk to. Honoured when it names a pack in the catalog.
    var model: String?
    /// How long the model is asked to think, when the template spells that out.
    var effort: ReasoningEffort?
    /// The conversation this belongs to, carried onto the cache slot and any archive of it.
    var tag: String? = nil
  }

  func complete(
    _ request: Request,
    id: Int? = nil,
    isCancelled: (@Sendable () -> Bool)? = nil,
    onText: ((String) -> Void)? = nil,
    onReasoning: ((String) -> Void)? = nil,
    onToolStanza: (() -> Void)? = nil,
    onToolText: ((String) -> Void)? = nil
  ) throws -> (
    parsed: ParsedCompletion, promptTokens: Int, completionTokens: Int, cancelled: Bool,
    tokens: [Int], reused: Int
  ) {
    residency.beginRequest()
    defer { residency.endRequest() }

    activateIfRequested(request.model)
    let model = try self.model()
    let rendered = try template.render(
      messages: request.messages,
      addGenerationPrompt: true,
      enableThinking: request.thinking,
      reasoningEffort: request.effort,
      tools: request.tools)

    stats.enter(id, phase: .prefill)
    var promptTokens = model.tokenizer.encode(rendered)
    var embeddings: MLXArray?
    var positions: MLXArray?
    if !request.images.isEmpty {
      if !model.isVisionLoaded {
        let start = Date()
        try model.vision()
        log?(String(format: "loaded vision tower in %.1fs", -start.timeIntervalSinceNow))
      }
      let multimodal = try model.prepareMultimodal(
        tokens: promptTokens, images: request.images)
      promptTokens = multimodal.tokens
      embeddings = multimodal.embeddings
      positions = multimodal.positions
    }

    var options = samplingOptions
    if let temperature = request.temperature { options.temperature = temperature }

    applyBudget(budget.observe(contextTokens: promptTokens.count + request.maxTokens))

    var cache: ModelCache?
    var reused = 0
    var lease: SessionCache.Lease?
    if request.images.isEmpty {
      let prepared = sessions.prepare(
        for: promptTokens, model: model, kvConfig: kvConfig, tag: request.tag)
      lease = prepared
      cache = prepared.cache
      reused = prepared.reused
      if prepared.recycled {
        applyBudget(budget.notePrefixEviction())
      }
      log?(
        "cache: reused \(reused) of \(promptTokens.count) prompt tokens "
          + sessions.lastTrace)
    }
    defer { if let lease { sessions.release(lease) } }
    stats.update(id) { record in
      record.promptTokens = promptTokens.count
      record.cachedTokens = reused
      record.maxTokens = request.maxTokens
    }

    var constraint: OutputConstraint?
    if let schema = request.responseSchema {
      let documents = try JSONSchema.documents(schema)
      constraint = OutputConstraint(documents: documents)
      log?("schema: constrained to \(documents.count) documents")
    }

    let generator = Generator(
      model: model, kvConfig: kvConfig, politeness: politeness)

    let opened =
      request.thinking
      && rendered.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("<think>")
    var filter = StreamFilter(thinking: opened)

    let promptLease = lease
    // A rewind point a few tokens short of the end, which is where the next prompt parts ways.
    // Not a chunk boundary: the chunk is whatever politeness and the neural engine make it,
    // and on a short prompt there may be no boundary before the end at all.
    let rewindReserve = 8
    let checkpointAt =
      promptTokens.count - rewindReserve > reused
      ? promptTokens.count - rewindReserve : nil
    let result = try withError { box in
      generator.generate(
        promptTokens: promptTokens, options: options, maxTokens: request.maxTokens,
        cache: cache, promptEmbeddings: embeddings, positions: positions,
        cachedPrefixLength: reused,
        constraint: constraint,
        checkpointAt: checkpointAt,
        isCancelled: { box.firstError != nil || isCancelled?() == true },
        onCheckpoint: { [sessions = self.sessions] in
          if let promptLease, let checkpointAt {
            sessions.checkpointPrefill(promptLease, at: checkpointAt)
          }
        },
        onPrefilled: { [sessions = self.sessions] in
          if let promptLease, checkpointAt == nil {
            sessions.checkpointPrompt(promptLease)
          }
        },
        onProgress: { [stats = self.stats] progress in
          switch progress {
          case .prefill(let done, let total):
            stats.enter(id, phase: .prefill)
            stats.update(id) { record in
              record.prefilled = done
              record.prefillTotal = total
            }
          case .decode(let count):
            stats.enter(id, phase: .decode)
            stats.update(id) { record in record.generated = count }
          }
        }
      ) { fragment in
        let piece = filter.push(fragment)
        if let thought = piece.reasoning { onReasoning?(thought) }
        if let visible = piece.content { onText?(visible) }
        if piece.startedToolCall { onToolStanza?() }
        if let command = piece.toolText { onToolText?(command) }
        return true
      }
    }
    if !result.cancelled {
      let tail = filter.flush()
      if let thought = tail.reasoning { onReasoning?(thought) }
      if let visible = tail.content { onText?(visible) }
      if let command = tail.toolText { onToolText?(command) }
    }
    if let lease { sessions.commit(lease, generated: result.tokens) }
    applyBudget(budget.notePoolPressure(cacheMemory: Memory.cacheMemory))
    if result.cancelled {
      stats.cancel(id)
      log?(
        "cancelled by client after \(result.stats.promptTokens) prefilled, "
          + "\(result.tokens.count) generated")
    } else {
      stats.enter(id, phase: .finishing)
      stats.record(id, generation: result.stats, cached: reused, reused: reused > 0)
    }

    let raw = opened ? "<think>" + result.text : result.text
    return (
      ToolCallParser.parse(raw, types: ToolCallParser.parameterTypes(request.tools)),
      promptTokens.count, result.tokens.count, result.cancelled, promptTokens, reused
    )
  }

  /// The part of a conversation that the next turn will begin with, rendered and tokenized.
  ///
  /// Rendered twice with two different stand-in user messages after it: whatever the two agree
  /// on is what any real message will be preceded by, however the template dresses a turn.
  /// Cut back to a line end so the last token cannot fuse with whatever is typed next.
  func stablePrefix(
    messages: [ChatMessage], tools: [[String: Any]]?, thinking: Bool, effort: ReasoningEffort?
  ) throws -> [Int] {
    func render(_ probe: String) throws -> String {
      try template.render(
        messages: messages + [.user(probe)], addGenerationPrompt: false,
        enableThinking: thinking, reasoningEffort: effort, tools: tools)
    }
    let first = Array(try render("\u{1}A").utf8)
    let second = Array(try render("\u{1}B").utf8)
    var shared = 0
    while shared < min(first.count, second.count), first[shared] == second[shared] { shared += 1 }
    while shared > 0, first[shared - 1] != UInt8(ascii: "\n") { shared -= 1 }
    guard shared > 0 else { return [] }
    return try model().tokenizer.encode(String(decoding: first[..<shared], as: UTF8.self))
  }

  /// Lays a conversation's stable prefix into the cache without generating anything, so the
  /// turn that follows only pays for what it adds. Stopping part way keeps what was read.
  func prefill(
    messages: [ChatMessage], tools: [[String: Any]]?, thinking: Bool, effort: ReasoningEffort?,
    tag: String?, pin: Bool, id: Int?, isCancelled: @escaping @Sendable () -> Bool
  ) throws -> (tokens: Int, reused: Int, cancelled: Bool) {
    residency.beginRequest()
    defer { residency.endRequest() }

    let model = try self.model()
    let tokens = try stablePrefix(
      messages: messages, tools: tools, thinking: thinking, effort: effort)
    guard tokens.count > 1, !isCancelled() else { return (tokens.count, 0, true) }

    applyBudget(budget.observe(contextTokens: tokens.count))
    let lease = sessions.prepare(for: tokens, model: model, kvConfig: kvConfig, tag: tag)
    defer { sessions.release(lease) }
    if lease.recycled { applyBudget(budget.notePrefixEviction()) }
    stats.update(id) { record in
      record.promptTokens = tokens.count
      record.cachedTokens = lease.reused
    }
    log?("readahead: \(lease.reused) of \(tokens.count) already held")

    guard lease.reused < tokens.count - 1 else {
      sessions.commit(lease, generated: [])
      if pin { sessions.pin(lease, tokens: tokens) }
      return (tokens.count, lease.reused, false)
    }

    let rewindReserve = 8
    let checkpointAt =
      tokens.count - rewindReserve > lease.reused ? tokens.count - rewindReserve : nil
    let generator = Generator(model: model, kvConfig: kvConfig, politeness: politeness)
    let result = generator.generate(
      promptTokens: tokens, maxTokens: 0, cache: lease.cache,
      cachedPrefixLength: lease.reused, checkpointAt: checkpointAt,
      isCancelled: isCancelled,
      onCheckpoint: { [sessions = self.sessions] in
        if let checkpointAt { sessions.checkpointPrefill(lease, at: checkpointAt) }
      },
      onPrefilled: { [sessions = self.sessions] in
        if checkpointAt == nil { sessions.checkpointPrompt(lease) }
      },
      onProgress: { [stats = self.stats] progress in
        guard case .prefill(let done, let total) = progress else { return }
        stats.enter(id, phase: .prefill)
        stats.update(id) { record in
          record.prefilled = done
          record.prefillTotal = total
        }
      })
    sessions.commit(lease, generated: [])
    applyBudget(budget.notePoolPressure(cacheMemory: Memory.cacheMemory))
    if !result.cancelled, pin { sessions.pin(lease, tokens: tokens) }
    log?(
      "readahead: read \(result.stats.promptTokens) tokens in "
        + String(format: "%.1fs", result.stats.promptSeconds)
        + (result.cancelled ? ", stopped early" : ""))
    return (tokens.count, lease.reused, result.cancelled)
  }

  private func handleOpenAI(
    _ request: HTTPRequest, _ writer: ResponseWriter, _ id: Int? = nil
  ) {
    guard let body = request.json() else {
      writer.sendError(status: 400, type: "invalid_request_error", message: "invalid JSON")
      return
    }

    do {
      let parsed = try parseOpenAI(body)
      stats.update(id) { $0.stream = parsed.stream }
      let identifier = "chatcmpl-" + UUID().uuidString.prefix(12)
      let created = Int(Date().timeIntervalSince1970)

      if parsed.stream {
        writer.beginEventStream()
        func chunk(_ delta: [String: Any], finish: String? = nil) {
          writer.sendEvent(data: [
            "id": identifier, "object": "chat.completion.chunk", "created": created,
            "model": modelName,
            "choices": [
              [
                "index": 0, "delta": delta,
                "finish_reason": finish as Any? ?? NSNull(),
              ]
            ],
          ])
        }
        chunk(["role": "assistant", "content": ""])

        let completion = try complete(
          parsed, id: id, isCancelled: { writer.isCancelled },
          onText: { text in
            chunk(["content": text])
          },
          onReasoning: { thought in
            chunk(["reasoning_content": thought])
          }
        )
        if completion.cancelled {
          writer.finish()
          return
        }
        if !completion.parsed.toolCalls.isEmpty {
          for (index, call) in completion.parsed.toolCalls.enumerated() {
            chunk([
              "tool_calls": [
                [
                  "index": index, "id": call.id, "type": "function",
                  "function": [
                    "name": call.name, "arguments": call.argumentsJSON,
                  ],
                ]
              ]
            ])
          }
        }
        chunk([:], finish: completion.parsed.toolCalls.isEmpty ? "stop" : "tool_calls")
        writer.sendRaw("data: [DONE]\n\n")
        writer.finish()
        return
      }

      let completion = try complete(parsed, id: id, isCancelled: { writer.isCancelled })
      if completion.cancelled {
        writer.finish()
        return
      }
      var message: [String: Any] = [
        "role": "assistant",
        "content": completion.parsed.content.isEmpty
          ? NSNull() : completion.parsed.content,
      ]
      if let reasoning = completion.parsed.reasoning {
        message["reasoning_content"] = reasoning
      }
      if !completion.parsed.toolCalls.isEmpty {
        message["tool_calls"] = completion.parsed.toolCalls.map { call in
          [
            "id": call.id, "type": "function",
            "function": ["name": call.name, "arguments": call.argumentsJSON],
          ]
        }
      }
      writer.send(json: [
        "id": identifier, "object": "chat.completion", "created": created,
        "model": modelName,
        "choices": [
          [
            "index": 0, "message": message,
            "finish_reason": completion.parsed.toolCalls.isEmpty
              ? "stop" : "tool_calls",
          ]
        ],
        "usage": [
          "prompt_tokens": completion.promptTokens,
          "completion_tokens": completion.completionTokens,
          "total_tokens": completion.promptTokens + completion.completionTokens,
        ],
      ])
    } catch {
      writer.sendError(
        status: 400, type: "invalid_request_error", message: "\(error)")
    }
  }

  private func parseOpenAI(_ body: [String: Any]) throws -> Request {
    var messages: [ChatMessage] = []
    var images: [ProcessedImage] = []

    let incoming = body["messages"] as? [[String: Any]] ?? []

    let systemParts =
      incoming
      .filter { $0["role"] as? String == "system" }
      .map { stringContent($0["content"]) }
      .filter { !$0.isEmpty }
    if !systemParts.isEmpty {
      messages.append(.system(systemParts.joined(separator: "\n\n")))
    }

    for raw in incoming {
      let role = raw["role"] as? String ?? "user"
      if role == "system" { continue }

      if role == "tool" {
        messages.append(.toolResult(stringContent(raw["content"])))
        continue
      }

      var imageCount = 0
      var text = ""
      if let parts = raw["content"] as? [[String: Any]] {
        for part in parts {
          switch part["type"] as? String {
          case "text": text += part["text"] as? String ?? ""
          case "image_url":
            if let url = (part["image_url"] as? [String: Any])?["url"] as? String,
              let image = try? decodeImage(url)
            {
              images.append(image)
              imageCount += 1
            }
          default: break
          }
        }
      } else {
        text = stringContent(raw["content"])
      }

      let calls = (raw["tool_calls"] as? [[String: Any]] ?? []).compactMap {
        call -> ToolCall? in
        guard let function = call["function"] as? [String: Any],
          let name = function["name"] as? String
        else { return nil }
        return ToolCall(
          id: call["id"] as? String ?? "call_" + UUID().uuidString.prefix(8),
          name: name,
          argumentsJSON: function["arguments"] as? String ?? "{}")
      }

      if imageCount > 0 {
        messages.append(.user(text: text, imageCount: imageCount))
      } else {
        messages.append(ChatMessage(role: role, content: .text(text), toolCalls: calls))
      }
    }

    let tools = (body["tools"] as? [[String: Any]])?.compactMap { tool -> [String: Any]? in
      guard let function = tool["function"] as? [String: Any] else { return tool }
      return function
    }

    if body["grammar"] != nil {
      throw BonsaiError.unsupportedModel(
        "GBNF grammars are not supported; send response_format with a json_schema instead")
    }
    let schema = jsonSchema(from: body["response_format"])

    return Request(
      messages: messages, tools: tools,
      maxTokens: body["max_tokens"] as? Int ?? 1024,
      temperature: (body["temperature"] as? NSNumber)?.floatValue,
      stream: body["stream"] as? Bool ?? false,
      // A constrained document has no room for a reasoning block.
      thinking: schema != nil ? false : (thinkingPreference(body) ?? defaultThinking),
      images: images,
      responseSchema: schema,
      model: body["model"] as? String)
  }

  private func jsonSchema(from value: Any?) -> [String: Any]? {
    guard let format = value as? [String: Any],
      format["type"] as? String == "json_schema",
      let wrapper = format["json_schema"] as? [String: Any]
    else { return nil }
    return wrapper["schema"] as? [String: Any]
  }

  private func handleAnthropic(
    _ request: HTTPRequest, _ writer: ResponseWriter, _ id: Int? = nil
  ) {
    guard let body = request.json() else {
      writer.sendError(status: 400, type: "invalid_request_error", message: "invalid JSON")
      return
    }

    do {
      let parsed = try parseAnthropic(body)
      stats.update(id) { $0.stream = parsed.stream }
      let identifier = "msg_" + UUID().uuidString.prefix(16)

      if parsed.stream {
        writer.beginEventStream()
        writer.sendEvent(
          name: "message_start",
          data: [
            "type": "message_start",
            "message": [
              "id": identifier, "type": "message", "role": "assistant",
              "model": modelName, "content": [], "stop_reason": NSNull(),
              "stop_sequence": NSNull(),
              "usage": ["input_tokens": 0, "output_tokens": 0],
            ],
          ])
        var nextIndex = 0
        var open: String?
        func close() {
          guard let kind = open else { return }
          if kind == "thinking" {
            writer.sendEvent(
              name: "content_block_delta",
              data: [
                "type": "content_block_delta", "index": nextIndex - 1,
                "delta": ["type": "signature_delta", "signature": ""],
              ])
          }
          writer.sendEvent(
            name: "content_block_stop",
            data: ["type": "content_block_stop", "index": nextIndex - 1])
          open = nil
        }
        func begin(_ kind: String) {
          guard open != kind else { return }
          close()
          let block: [String: Any] =
            kind == "thinking"
            ? ["type": "thinking", "thinking": ""] : ["type": "text", "text": ""]
          writer.sendEvent(
            name: "content_block_start",
            data: ["type": "content_block_start", "index": nextIndex, "content_block": block])
          nextIndex += 1
          open = kind
        }

        let completion = try complete(
          parsed, id: id, isCancelled: { writer.isCancelled },
          onText: { text in
            begin("text")
            writer.sendEvent(
              name: "content_block_delta",
              data: [
                "type": "content_block_delta", "index": nextIndex - 1,
                "delta": ["type": "text_delta", "text": text],
              ])
          },
          onReasoning: { thought in
            begin("thinking")
            writer.sendEvent(
              name: "content_block_delta",
              data: [
                "type": "content_block_delta", "index": nextIndex - 1,
                "delta": ["type": "thinking_delta", "thinking": thought],
              ])
          }
        )
        if completion.cancelled {
          writer.finish()
          return
        }
        if open != "text" && (open == nil || completion.parsed.toolCalls.isEmpty) {
          begin("text")
        }
        close()

        for call in completion.parsed.toolCalls {
          let index = nextIndex
          nextIndex += 1
          writer.sendEvent(
            name: "content_block_start",
            data: [
              "type": "content_block_start", "index": index,
              "content_block": [
                "type": "tool_use", "id": call.id, "name": call.name,
                "input": [:],
              ],
            ])
          writer.sendEvent(
            name: "content_block_delta",
            data: [
              "type": "content_block_delta", "index": index,
              "delta": [
                "type": "input_json_delta", "partial_json": call.argumentsJSON,
              ],
            ])
          writer.sendEvent(
            name: "content_block_stop",
            data: ["type": "content_block_stop", "index": index])
        }

        writer.sendEvent(
          name: "message_delta",
          data: [
            "type": "message_delta",
            "delta": [
              "stop_reason": completion.parsed.toolCalls.isEmpty
                ? "end_turn" : "tool_use",
              "stop_sequence": NSNull(),
            ],
            "usage": ["output_tokens": completion.completionTokens],
          ])
        writer.sendEvent(name: "message_stop", data: ["type": "message_stop"])
        writer.finish()
        return
      }

      let completion = try complete(parsed, id: id, isCancelled: { writer.isCancelled })
      if completion.cancelled {
        writer.finish()
        return
      }
      var blocks: [[String: Any]] = []
      if let reasoning = completion.parsed.reasoning {
        blocks.append(["type": "thinking", "thinking": reasoning, "signature": ""])
      }
      if !completion.parsed.content.isEmpty {
        blocks.append(["type": "text", "text": completion.parsed.content])
      }
      for call in completion.parsed.toolCalls {
        let input =
          (call.argumentsJSON.data(using: .utf8)
            .flatMap { try? JSONSerialization.jsonObject(with: $0) }
            as? [String: Any]) ?? [:]
        blocks.append([
          "type": "tool_use", "id": call.id, "name": call.name, "input": input,
        ])
      }
      if !blocks.contains(where: { $0["type"] as? String != "thinking" }) {
        blocks.append(["type": "text", "text": ""])
      }

      writer.send(json: [
        "id": identifier, "type": "message", "role": "assistant", "model": modelName,
        "content": blocks,
        "stop_reason": completion.parsed.toolCalls.isEmpty ? "end_turn" : "tool_use",
        "stop_sequence": NSNull(),
        "usage": [
          "input_tokens": completion.promptTokens,
          "output_tokens": completion.completionTokens,
        ],
      ])
    } catch {
      fail(writer, error)
    }
  }

  private func parseAnthropic(_ body: [String: Any]) throws -> Request {
    var messages: [ChatMessage] = []
    var images: [ProcessedImage] = []

    let incoming = body["messages"] as? [[String: Any]] ?? []

    var systemParts: [String] = []
    if let system = body["system"] {
      let text = stringContent(system)
      if !text.isEmpty { systemParts.append(text) }
    }
    for raw in incoming where raw["role"] as? String == "system" {
      let text = stringContent(raw["content"])
      if !text.isEmpty { systemParts.append(text) }
    }
    if !systemParts.isEmpty {
      messages.append(.system(systemParts.joined(separator: "\n\n")))
    }

    for raw in incoming {
      let role = raw["role"] as? String ?? "user"
      if role == "system" { continue }

      guard let blocks = raw["content"] as? [[String: Any]] else {
        messages.append(ChatMessage(role: role, content: .text(stringContent(raw["content"]))))
        continue
      }

      var text = ""
      var calls: [ToolCall] = []
      var imageCount = 0
      var results: [String] = []

      for block in blocks {
        switch block["type"] as? String {
        case "text":
          text += block["text"] as? String ?? ""
        case "image":
          if let source = block["source"] as? [String: Any],
            let data = source["data"] as? String,
            let media = source["media_type"] as? String,
            let image = try? decodeImage("data:\(media);base64,\(data)")
          {
            images.append(image)
            imageCount += 1
          }
        case "tool_use":
          let input = block["input"] as? [String: Any] ?? [:]
          let json =
            (try? JSONSerialization.data(withJSONObject: input))
            .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
          calls.append(
            ToolCall(
              id: block["id"] as? String ?? "call_" + UUID().uuidString.prefix(8),
              name: block["name"] as? String ?? "",
              argumentsJSON: json))
        case "tool_result":
          results.append(stringContent(block["content"]))
        default:
          break
        }
      }

      for result in results { messages.append(.toolResult(result)) }

      if !text.isEmpty || !calls.isEmpty || imageCount > 0 {
        if imageCount > 0 {
          messages.append(.user(text: text, imageCount: imageCount))
        } else {
          messages.append(
            ChatMessage(role: role, content: .text(text), toolCalls: calls))
        }
      }
    }

    let tools = (body["tools"] as? [[String: Any]])?.map { tool -> [String: Any] in
      [
        "name": tool["name"] as? String ?? "",
        "description": tool["description"] as? String ?? "",
        "parameters": tool["input_schema"] as? [String: Any] ?? [:],
      ]
    }

    var thinking = defaultThinking
    if let requested = body["thinking"] as? [String: Any] {
      thinking = (requested["type"] as? String) == "enabled"
    }

    return Request(
      messages: messages, tools: tools,
      maxTokens: body["max_tokens"] as? Int ?? 1024,
      temperature: (body["temperature"] as? NSNumber)?.floatValue,
      stream: body["stream"] as? Bool ?? false,
      thinking: thinking,
      images: images,
      responseSchema: nil,
      model: body["model"] as? String)
  }

  private func handleCountTokens(_ request: HTTPRequest, _ writer: ResponseWriter) {
    guard let body = request.json() else {
      writer.sendError(status: 400, type: "invalid_request_error", message: "invalid JSON")
      return
    }
    do {
      let parsed = try parseAnthropic(body)
      let rendered = try template.render(
        messages: parsed.messages, addGenerationPrompt: true,
        enableThinking: parsed.thinking, tools: parsed.tools)
      writer.send(json: ["input_tokens": try model().tokenizer.encode(rendered).count])
    } catch {
      fail(writer, error)
    }
  }

  /// `thinking`, or the `chat_template_kwargs.enable_thinking` form that llama.cpp and oMLX
  /// clients send, since the top-level OpenAI fields for this are not carried by either.
  private func thinkingPreference(_ body: [String: Any]) -> Bool? {
    if let direct = body["thinking"] as? Bool { return direct }
    if let kwargs = body["chat_template_kwargs"] as? [String: Any],
      let enabled = kwargs["enable_thinking"] as? Bool
    {
      return enabled
    }
    if let effort = body["reasoning_effort"] as? String {
      return effort != "none"
    }
    return nil
  }

  /// A request that failed mid-generation must not try to send a second status line: the
  /// stream is already committed, so it is closed instead. An MLX failure is ours, not the
  /// caller's, so it reports as a 500.
  private func fail(_ writer: ResponseWriter, _ error: Error) {
    log?("request failed: \(error)")
    guard !writer.hasBegun else {
      writer.finish()
      return
    }
    let isInternal = error is MLXError
    writer.sendError(
      status: isInternal ? 500 : 400,
      type: isInternal ? "api_error" : "invalid_request_error",
      message: "\(error)")
  }

  private func stringContent(_ value: Any?) -> String {
    if let text = value as? String { return text }
    if let blocks = value as? [[String: Any]] {
      return blocks.compactMap { $0["text"] as? String }.joined()
    }
    return ""
  }

  /// One picture off disk, processed by whatever the resident pack's tower expects. Reached by
  /// the window as well as the port, so a conversation in the app sees what a client would.
  /// Call it on the generation queue: it reads the model.
  public func processImage(at url: URL) throws -> ProcessedImage {
    let model = try self.model()
    guard let visionConfig = model.config.visionConfig, model.hasVision else {
      throw BonsaiError.missingComponent("this pack has no vision tower")
    }
    return try ImageProcessor(config: visionConfig).process(contentsOf: url)
  }

  private func decodeImage(_ source: String) throws -> ProcessedImage {
    let model = try self.model()
    guard let visionConfig = model.config.visionConfig, model.hasVision else {
      throw BonsaiError.missingComponent("this pack has no vision tower")
    }
    let processor = ImageProcessor(config: visionConfig)

    if source.hasPrefix("data:") {
      guard let comma = source.firstIndex(of: ","),
        let data = Data(
          base64Encoded: String(source[source.index(after: comma)...]))
      else {
        throw BonsaiError.imageProcessing("could not decode the inline image data")
      }
      let url = FileManager.default.temporaryDirectory
        .appending(path: UUID().uuidString)
      try data.write(to: url)
      defer { try? FileManager.default.removeItem(at: url) }
      return try processor.process(contentsOf: url)
    }
    return try processor.process(contentsOf: URL(filePath: source))
  }
}

/// Splits a raw stream into the thinking and the answer, handing back both as they arrive. A
/// guard window is held back so a tag split across two fragments is never mistaken for text.
struct StreamFilter {
  struct Output {
    var reasoning: String?
    var content: String?
    /// The tool call as it is being written. Kept rather than dropped so the wait between a
    /// thought ending and a call landing has something in it.
    var toolText: String?
    /// True on the one push where the model opens a tool call.
    var startedToolCall = false

    var isEmpty: Bool { reasoning == nil && content == nil && toolText == nil }
  }

  private enum Phase {
    case thinking
    case answer
    case tool
  }

  private var buffer = ""
  private var phase: Phase
  private var started: Bool
  private let guardLength = 12

  init(thinking: Bool) {
    self.phase = thinking ? .thinking : .answer
    self.started = thinking
  }

  mutating func push(_ fragment: String) -> Output {
    buffer += fragment
    var out = Output()

    if !started {
      let lead = buffer.drop(while: \.isWhitespace)
      if lead.hasPrefix("<think>") {
        buffer = String(lead.dropFirst("<think>".count))
        phase = .thinking
      } else if "<think>".hasPrefix(lead) {
        return out
      }
      started = true
    }

    if phase == .thinking {
      guard let end = buffer.range(of: "</think>") else {
        out.reasoning = takeGuarded()
        return out
      }
      let thought = String(buffer[buffer.startIndex..<end.lowerBound])
      if !thought.isEmpty { out.reasoning = thought }
      buffer = String(buffer[end.upperBound...])
      phase = .answer
    }

    if phase == .answer {
      if let call = buffer.range(of: "<tool_call>") {
        let visible = String(buffer[buffer.startIndex..<call.lowerBound])
        if !visible.isEmpty { out.content = visible }
        buffer = String(buffer[call.upperBound...])
        phase = .tool
        out.startedToolCall = true
      } else {
        out.content = takeGuarded()
        return out
      }
    }

    out.toolText = takeGuarded()
    return out
  }

  mutating func flush() -> Output {
    guard !buffer.isEmpty else { return Output() }
    let rest = buffer
    buffer = ""
    switch phase {
    case .thinking: return Output(reasoning: rest)
    case .answer: return Output(content: rest)
    case .tool: return Output(toolText: rest)
    }
  }

  private mutating func takeGuarded() -> String? {
    let characters = Array(buffer)
    guard characters.count > guardLength else { return nil }
    let cut = characters.count - guardLength
    buffer = String(characters[cut...])
    return String(characters[0..<cut])
  }
}
