// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public final class APIServer: @unchecked Sendable {
  public let directory: URL
  public let template: ChatTemplate
  public let modelName: String
  public var defaultThinking: Bool
  public var samplingOptions: SamplingOptions
  public var kvConfig: KVCacheConfig
  public let politeness: Politeness.Level
  public let ropeScaling: RopeScaling
  public let residency: ResidencyManager
  public let sessions: SessionCache
  public let prefixStore: PrefixStore?
  public let budget: MemoryBudget
  public let stats = ServeStats()

  private var loaded: BonsaiModel?
  private let generationQueue = DispatchQueue(label: "bonsai.generate")
  private var server: HTTPServer?
  public var log: (@Sendable (String) -> Void)?

  public init(
    directory: URL, modelName: String = "ternary-bonsai-2-27b",
    thinking: Bool = true, samplingOptions: SamplingOptions = SamplingOptions(),
    kvConfig: KVCacheConfig = KVCacheConfig(),
    residency: ResidencyManager.Options = ResidencyManager.Options(),
    politeness: Politeness.Level = .adaptive,
    ropeScaling: RopeScaling = .none,
    budget: MemoryBudget? = nil,
    prefixStore: PrefixStore? = nil,
    preload: Bool = true
  ) throws {
    self.politeness = politeness
    self.ropeScaling = ropeScaling
    self.directory = directory
    self.template = try ChatTemplate(directory: directory)
    self.modelName = modelName
    self.defaultThinking = thinking
    self.samplingOptions = samplingOptions
    self.kvConfig = kvConfig
    let budget =
      budget
      ?? MemoryBudget(
        kvBits: kvConfig.bits,
        maxContextTokens: ropeScaling.effectiveContext,
        weights: MemoryBudget.weightBytes(in: directory) ?? MemoryBudget.defaultWeights)
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

  @discardableResult
  public func model() throws -> BonsaiModel {
    if let loaded { return loaded }
    let start = Date()
    let model = try BonsaiModel(directory: directory, ropeScaling: ropeScaling)
    loaded = model
    log?(String(format: "loaded model in %.1fs", -start.timeIntervalSinceNow))
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
      writer.send(
        json: [
          "object": "list",
          "data": [["id": modelName, "object": "model", "owned_by": "prism-ml"]],
        ])
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

  private struct Request {
    var messages: [ChatMessage]
    var tools: [[String: Any]]?
    var maxTokens: Int
    var temperature: Float?
    var stream: Bool
    var thinking: Bool
    var images: [ProcessedImage]
    var responseSchema: [String: Any]?
  }

  private func complete(
    _ request: Request,
    id: Int? = nil,
    isCancelled: (@Sendable () -> Bool)? = nil,
    onText: ((String) -> Void)? = nil
  ) throws -> (
    parsed: ParsedCompletion, promptTokens: Int, completionTokens: Int, cancelled: Bool
  ) {
    residency.beginRequest()
    defer { residency.endRequest() }

    let model = try self.model()
    let rendered = try template.render(
      messages: request.messages,
      addGenerationPrompt: true,
      enableThinking: request.thinking,
      tools: request.tools)

    stats.enter(id, phase: .prefill)
    var promptTokens = model.tokenizer.encode(rendered)
    var embeddings: MLXArray?
    var positions: MLXArray?
    if !request.images.isEmpty {
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
        for: promptTokens, model: model, kvConfig: kvConfig)
      lease = prepared
      cache = prepared.cache
      reused = prepared.reused
      if prepared.recycled {
        applyBudget(budget.notePrefixEviction())
      }
      if reused > 0 {
        log?("cache: reused \(reused) of \(promptTokens.count) prompt tokens")
      }
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

    var filter = StreamFilter(thinking: request.thinking)

    let result = try withError { box in
      generator.generate(
      promptTokens: promptTokens, options: options, maxTokens: request.maxTokens,
      cache: cache, promptEmbeddings: embeddings, positions: positions,
      cachedPrefixLength: reused,
      constraint: constraint,
      isCancelled: { box.firstError != nil || isCancelled?() == true },
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
      if let onText, let visible = filter.push(fragment), !visible.isEmpty {
        onText(visible)
      }
      return true
    }
    }
    if let onText, !result.cancelled, let tail = filter.flush(), !tail.isEmpty {
      onText(tail)
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

    let raw = request.thinking ? "<think>" + result.text : result.text
    return (
      ToolCallParser.parse(raw), promptTokens.count, result.tokens.count, result.cancelled
    )
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
          parsed, id: id, isCancelled: { writer.isCancelled }
        ) { text in
          chunk(["content": text])
        }
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

    let systemParts = incoming
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
      responseSchema: schema)
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
        writer.sendEvent(
          name: "content_block_start",
          data: [
            "type": "content_block_start", "index": 0,
            "content_block": ["type": "text", "text": ""],
          ])

        let completion = try complete(
          parsed, id: id, isCancelled: { writer.isCancelled }
        ) { text in
          writer.sendEvent(
            name: "content_block_delta",
            data: [
              "type": "content_block_delta", "index": 0,
              "delta": ["type": "text_delta", "text": text],
            ])
        }
        if completion.cancelled {
          writer.finish()
          return
        }
        writer.sendEvent(
          name: "content_block_stop",
          data: ["type": "content_block_stop", "index": 0])

        for (offset, call) in completion.parsed.toolCalls.enumerated() {
          let index = offset + 1
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
      if blocks.isEmpty { blocks = [["type": "text", "text": ""]] }

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
      responseSchema: nil)
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

struct StreamFilter {
  private var buffer = ""
  private var emitted = 0
  private var inThinking: Bool
  private var stopped = false
  private let guardLength = 12

  init(thinking: Bool) {
    self.inThinking = thinking
  }

  mutating func push(_ fragment: String) -> String? {
    guard !stopped else { return nil }
    buffer += fragment

    if inThinking {
      guard let end = buffer.range(of: "</think>") else { return nil }
      buffer = String(buffer[end.upperBound...])
      inThinking = false
    }
    if let call = buffer.range(of: "<tool_call>") {
      buffer = String(buffer[buffer.startIndex..<call.lowerBound])
      stopped = true
      return take(all: true)
    }
    return take(all: false)
  }

  mutating func flush() -> String? {
    guard !stopped, !inThinking else { return nil }
    return take(all: true)
  }

  private mutating func take(all: Bool) -> String? {
    let characters = Array(buffer)
    let available = all ? characters.count : max(0, characters.count - guardLength)
    guard available > emitted else { return nil }
    let slice = String(characters[emitted..<available])
    emitted = available
    return slice
  }
}
