// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One turn of an agentic conversation, run on the server's own generation queue so a chat in
// the window and a client on the port never hold the weights at once.

import Foundation
import Jinja
import OrderedCollections

/// A tool as the chat template wants it: a name, a sentence, and a JSON Schema for the
/// arguments. Carried as text so a schema can cross to the generation queue.
public struct ToolSchema: Sendable, Equatable {
  public var name: String
  public var description: String
  public var parametersJSON: String

  public init(name: String, description: String, parametersJSON: String) {
    self.name = name
    self.description = description
    self.parametersJSON = parametersJSON
  }

  var templateValue: [String: Any] {
    let parameters =
      (parametersJSON.data(using: .utf8)
        .flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any]) ?? [:]
    return ["name": name, "description": description, "parameters": parameters]
  }

  /// The same, keys in the order the schema's text has them, for a template that writes the
  /// schema out as JSON and would otherwise have to pick an order of its own.
  var orderedValue: Value {
    var function = OrderedDictionary<ObjectKey, Value>()
    function["name"] = .string(name)
    function["description"] = .string(description)
    function["parameters"] = OrderedJSON.object(parametersJSON)
    return .object(function)
  }
}

public struct AgentTurn: Sendable {
  public var reasoning: String?
  public var content: String
  public var toolCalls: [ToolCall]
  public var promptTokens: Int
  public var cachedTokens: Int
  public var completionTokens: Int
  public var seconds: Double
  public var cancelled: Bool
}

/// Drives turns against the resident pack. The server is the owner of the weights, the prefix
/// cache and the memory budget; this only ever borrows them, so the readout stays one picture.
public final class AgentEngine: @unchecked Sendable {
  private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var raised = false
    var isRaised: Bool {
      lock.lock()
      defer { lock.unlock() }
      return raised
    }
    func raise() {
      lock.lock()
      raised = true
      lock.unlock()
    }
    func lower() {
      lock.lock()
      raised = false
      lock.unlock()
    }
  }

  private final class Text: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = ""
    var value: String {
      lock.lock()
      defer { lock.unlock() }
      return stored
    }
    func append(_ more: String) {
      lock.lock()
      stored += more
      lock.unlock()
    }
    func clear() {
      lock.lock()
      stored = ""
      lock.unlock()
    }
  }

  private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = 0
    var value: Int {
      lock.lock()
      defer { lock.unlock() }
      return stored
    }
    func set(_ count: Int) {
      lock.lock()
      stored = count
      lock.unlock()
    }
  }

  /// Pictures already read for this pack, so a conversation carrying an image does not pay
  /// the tower's price again on every turn that follows it.
  private final class ImageCache: @unchecked Sendable {
    private struct Key: Hashable {
      var path: String
      var modified: Date?
      var size: Int?
    }

    private let lock = NSLock()
    private var entries: [Key: ProcessedImage] = [:]
    private var order: [Key] = []
    private let limit = 24

    private func key(for url: URL) -> Key {
      let values = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
      return Key(
        path: url.path, modified: values?.contentModificationDate, size: values?.fileSize)
    }

    func image(for url: URL, make: (URL) throws -> ProcessedImage) -> ProcessedImage? {
      let key = key(for: url)
      lock.lock()
      let hit = entries[key]
      lock.unlock()
      if let hit { return hit }

      guard let made = try? make(url) else { return nil }
      lock.lock()
      entries[key] = made
      order.append(key)
      if order.count > limit { entries[order.removeFirst()] = nil }
      lock.unlock()
      return made
    }

    func empty() {
      lock.lock()
      entries.removeAll()
      order.removeAll()
      lock.unlock()
    }
  }

  private let imageCache = ImageCache()

  public let server: APIServer
  /// Whether the generation running now has stopped answering and started writing a tool
  /// call. Read by the window, which has no other way to tell the two apart mid-turn.
  private let toolStanza = Flag()

  public var isWritingToolCall: Bool { toolStanza.isRaised }

  /// The tool call as it is being written, so the window can show it arriving rather than
  /// leaving a gap between the thought that preceded it and the call itself.
  public var writingCommand: String { commandText.value }

  /// What this turn has generated so far, held here because it is the one place that has it
  /// the moment it exists. The session's transcript catches up on its own schedule.
  public var liveReasoning: String { reasoningText.value }
  public var liveAnswer: String { answerText.value }

  private final class Tokens: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [Int] = []
    var value: [Int] { lock.withLock { stored } }
    func set(_ tokens: [Int]) { lock.withLock { stored = tokens } }
  }

  private let promptTokens = Tokens()

  /// The last prompt this engine rendered, which is what pairs a conversation with the
  /// archives on disk that hold its prefix.
  public var lastPromptTokens: [Int] { promptTokens.value }
  private let systemCount = Counter()
  private let instructionCount = Counter()
  private let schemaCount = Counter()
  private let commandText = Text()
  private let reasoningText = Text()
  private let answerText = Text()

  /// How much of a prompt is the instructions and the tool schemas — the part that is the same
  /// every turn, and the part someone waiting on a first answer is mostly waiting for.
  public var systemTokens: Int { systemCount.value }
  /// The two halves of that, kept apart so the breakdown behind the context bar can say which
  /// of them is the one worth doing something about.
  public var instructionTokens: Int { instructionCount.value }
  public var toolSchemaTokens: Int { schemaCount.value }
  /// Where generation stops when nothing else stops it first. Held high because an agent's
  /// turn is a tool call away from being long, and never shown to the model as a bound.
  public var maxTokens: Int
  public var effort: ReasoningEffort
  public var thinking: Bool

  public init(
    server: APIServer, maxTokens: Int = 8192, effort: ReasoningEffort = .xhigh,
    thinking: Bool = true
  ) {
    self.server = server
    self.maxTokens = maxTokens
    self.effort = effort
    self.thinking = thinking
  }

  /// The arguments each tool call was generated with, exactly as written, so a transcript that
  /// hands back a re-serialized copy still renders the call the way the model spelled it.
  private final class Spellings: @unchecked Sendable {
    private let lock = NSLock()
    private var byID: [String: String] = [:]
    private var order: [String] = []
    private let limit = 4096

    func note(_ calls: [ToolCall]) {
      lock.lock()
      defer { lock.unlock() }
      for call in calls where byID[call.id] == nil {
        byID[call.id] = call.argumentsJSON
        order.append(call.id)
      }
      if order.count > limit {
        for id in order.prefix(order.count - limit) { byID[id] = nil }
        order.removeFirst(order.count - limit)
      }
    }

    func spelling(of id: String) -> String? {
      lock.lock()
      defer { lock.unlock() }
      return byID[id]
    }
  }

  private let spellings = Spellings()
  private let measured = Text()

  public func spelling(ofCall id: String) -> String? { spellings.spelling(of: id) }

  public func run(
    messages: [ChatMessage],
    tools: [ToolSchema] = [],
    maxTokens: Int? = nil,
    effort: ReasoningEffort? = nil,
    tag: String? = nil,
    model: String? = nil,
    onText: (@Sendable (String) -> Void)? = nil,
    onReasoning: (@Sendable (String) -> Void)? = nil
  ) async throws -> AgentTurn {
    stopReadahead()
    let cancel = Flag()
    let maxTokens = maxTokens ?? self.maxTokens
    let effort = effort ?? self.effort
    let thinking = self.thinking
    toolStanza.lower()
    commandText.clear()
    reasoningText.clear()
    answerText.clear()
    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        server.generationQueue.async { [self] in
          let started = Date()
          let id = server.stats.enqueue(api: "chat")
          defer { server.stats.end(id) }
          if cancel.isRaised {
            continuation.resume(throwing: CancellationError())
            return
          }
          do {
            if let model, model != server.activeModelID {
              try server.activate(model)
              imageCache.empty()
            }
            let opened = resolveImages(in: messages)
            let messages = opened.messages
            let schemas = tools.isEmpty ? nil : tools.map(\.templateValue)
            let ordered = tools.isEmpty ? nil : tools.map(\.orderedValue)

            let request = APIServer.Request(
              messages: messages,
              tools: schemas,
              orderedTools: ordered,
              maxTokens: maxTokens,
              temperature: nil,
              stream: onText != nil,
              thinking: thinking,
              images: opened.images,
              responseSchema: nil,
              model: nil,
              effort: effort,
              tag: tag)
            measure(
              messages: messages, tools: schemas, ordered: ordered, thinking: thinking,
              effort: effort)

            let outcome = try server.complete(
              request,
              id: id,
              isCancelled: { cancel.isRaised },
              onText: { [answerText] fragment in
                answerText.append(fragment)
                onText?(fragment)
              },
              onReasoning: { [reasoningText] fragment in
                reasoningText.append(fragment)
                onReasoning?(fragment)
              },
              onToolStanza: { [toolStanza] in toolStanza.raise() },
              onToolText: { [commandText] fragment in commandText.append(fragment) })
            promptTokens.set(outcome.tokens)
            spellings.note(outcome.parsed.toolCalls)
            continuation.resume(
              returning: AgentTurn(
                reasoning: outcome.parsed.reasoning,
                content: outcome.parsed.content,
                toolCalls: outcome.parsed.toolCalls,
                promptTokens: outcome.promptTokens,
                cachedTokens: outcome.reused,
                completionTokens: outcome.completionTokens,
                seconds: -started.timeIntervalSinceNow,
                cancelled: outcome.cancelled))
          } catch {
            continuation.resume(throwing: error)
          }
        }
      }
    } onCancel: {
      cancel.raise()
    }
  }

  /// How much of a prompt is instructions and how much is tool schemas, measured by rendering
  /// it without each. Done once for each model, effort and tool set rather than every turn.
  private func measure(
    messages: [ChatMessage], tools: [[String: Any]]?, ordered: [Value]?, thinking: Bool,
    effort: ReasoningEffort
  ) {
    let names = (tools ?? []).compactMap { $0["name"] as? String }.joined(separator: ",")
    let signature = "\(server.activeModelID)|\(effort.rawValue)|\(thinking)|\(names)"
    guard measured.value != signature, messages.first?.role == "system" else { return }

    func size(_ of: [ChatMessage], tools: [[String: Any]]?) -> Int? {
      guard !of.isEmpty,
        let text = try? server.template.render(
          messages: of, addGenerationPrompt: true, enableThinking: thinking,
          reasoningEffort: effort, tools: tools, orderedTools: tools == nil ? nil : ordered),
        let encoded = try? server.model().tokenizer.encode(text)
      else { return nil }
      return encoded.count
    }

    let probe = [messages[0], .user("")]
    guard let whole = size(probe, tools: tools),
      let withoutTools = size(probe, tools: nil),
      let bare = size([.user("")], tools: nil)
    else { return }
    let schemas = max(0, whole - withoutTools)
    let instructions = max(0, withoutTools - bare)
    schemaCount.set(schemas)
    instructionCount.set(instructions)
    systemCount.set(schemas + instructions)
    measured.clear()
    measured.append(signature)
    server.log?("prompt: \(instructions) instructions + \(schemas) tool schemas")
  }

  /// Reads a conversation into the cache ahead of its next turn. Only one runs at a time: a new
  /// one, or a turn, stops whichever is going, and what it had read so far stays read.
  public func readahead(
    messages: [ChatMessage], tools: [ToolSchema], effort: ReasoningEffort? = nil,
    tag: String?, pin: Bool = false, model: String? = nil
  ) async -> (tokens: Int, reused: Int, finished: Bool) {
    guard !messages.contains(where: { !$0.imagePaths.isEmpty }) else { return (0, 0, false) }
    let effort = effort ?? self.effort
    let thinking = self.thinking
    let cancel = Flag()
    readaheadFlag.swap(cancel)?.raise()
    return await withTaskCancellationHandler {
      await withCheckedContinuation { continuation in
        server.generationQueue.async { [self] in
          guard !cancel.isRaised, model == nil || model == server.activeModelID else {
            continuation.resume(returning: (0, 0, false))
            return
          }
          let id = server.stats.enqueue(api: "readahead")
          defer {
            server.stats.dismiss(id)
            server.stats.end(id)
          }
          let schemas = tools.isEmpty ? nil : tools.map(\.templateValue)
          let stats = server.stats
          let result = try? server.prefill(
            messages: messages, tools: schemas, thinking: thinking, effort: effort, tag: tag,
            pin: pin, id: id, isCancelled: { cancel.isRaised || stats.hasQueued(besides: id) })
          continuation.resume(
            returning: (result?.tokens ?? 0, result?.reused ?? 0, result?.cancelled == false))
        }
      }
    } onCancel: {
      cancel.raise()
    }
  }

  /// Stops a readahead that is running or waiting, keeping what it has read.
  public func stopReadahead() {
    readaheadFlag.swap(nil)?.raise()
  }

  private final class FlagSlot: @unchecked Sendable {
    private let lock = NSLock()
    private var held: Flag?
    func swap(_ next: Flag?) -> Flag? {
      lock.lock()
      defer { lock.unlock() }
      let previous = held
      held = next
      return previous
    }
  }

  private let readaheadFlag = FlagSlot()

  /// The packs the switcher offers, and the swap it performs. Activation is the server's, so a
  /// swap made here is the one the port sees too.
  public var catalog: ModelCatalog { server.catalog }

  public func activate(_ id: String) async throws {
    try await withCheckedThrowingContinuation { continuation in
      server.generationQueue.async { [self] in
        do {
          try server.activate(id)
          imageCache.empty()
          continuation.resume()
        } catch {
          continuation.resume(throwing: error)
        }
      }
    }
  }

  /// Messages with every picture they name opened, and the pictures themselves in the order
  /// the template will want them.
  private func resolveImages(
    in messages: [ChatMessage]
  ) -> (messages: [ChatMessage], images: [ProcessedImage]) {
    guard messages.contains(where: { !$0.imagePaths.isEmpty }) else { return (messages, []) }

    var out: [ChatMessage] = []
    var images: [ProcessedImage] = []
    out.reserveCapacity(messages.count)

    for message in messages {
      guard !message.imagePaths.isEmpty else {
        out.append(message)
        continue
      }
      var opened: [ProcessedImage] = []
      for path in message.imagePaths {
        guard
          let image = imageCache.image(for: URL(filePath: path), make: server.processImage)
        else { continue }
        opened.append(image)
      }
      images.append(contentsOf: opened)
      out.append(.user(text: message.plainText, imageCount: opened.count))
    }
    return (out, images)
  }

  /// Dropped when the pack changes: a picture is processed for the tower that will read it.
  public func forgetImages() {
    imageCache.empty()
  }

  public var contextCeiling: Int { server.budget.maxContextTokens }

  public func readout() -> ServeReadout { server.readout() }
}
