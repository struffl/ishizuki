// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One turn of an agentic conversation, run on the server's own generation queue so a chat in
// the window and a client on the port never hold the weights at once.

import Foundation

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

  /// The last turn's prompt, kept only to say how much of it the next one still agrees with.
  /// A prefix cache that never hits is usually a prompt that is not stable, not a cache that
  /// is not working, and the two look identical from the readout.
  private var promptTokens: [Int] = []

  /// The last prompt this engine rendered, which is what pairs a conversation with the
  /// archives on disk that hold its prefix.
  public var lastPromptTokens: [Int] { promptTokens }
  private let systemCount = Counter()
  private let commandText = Text()
  private let reasoningText = Text()
  private let answerText = Text()

  /// How much of a prompt is the instructions and the tool schemas — the part that is the same
  /// every turn, and the part someone waiting on a first answer is mostly waiting for.
  public var systemTokens: Int { systemCount.value }
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

  public func run(
    messages: [ChatMessage],
    tools: [ToolSchema] = [],
    onText: (@Sendable (String) -> Void)? = nil,
    onReasoning: (@Sendable (String) -> Void)? = nil
  ) async throws -> AgentTurn {
    let cancel = Flag()
    toolStanza.lower()
    commandText.clear()
    reasoningText.clear()
    answerText.clear()
    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        server.generationQueue.async { [self] in
          let started = Date()
          // Registered with the same accounting the port uses, so the window's dial and the
          // readout are reading one set of numbers rather than two.
          let id = server.stats.enqueue(api: "chat")
          defer { server.stats.end(id) }
          do {
            let request = APIServer.Request(
              messages: messages,
              tools: tools.isEmpty ? nil : tools.map(\.templateValue),
              maxTokens: maxTokens,
              temperature: nil,
              stream: onText != nil,
              thinking: thinking,
              images: [],
              responseSchema: nil,
              model: nil,
              effort: effort)
            let rendered = try? server.template.render(
              messages: messages,
              addGenerationPrompt: true,
              enableThinking: thinking,
              reasoningEffort: effort,
              tools: tools.isEmpty ? nil : tools.map(\.templateValue))
            let tokens = rendered.flatMap { try? server.model().tokenizer.encode($0) } ?? []

            // Measured by difference, and both halves of it: the instructions and the tool
            // schemas are each the same every turn, and a bar that counted only the first
            // called the other one the person's own tokens.
            if systemCount.value == 0, !tokens.isEmpty {
              let withoutSystem = messages.filter { $0.role != "system" }
              let schema = tools.isEmpty ? nil : tools.map(\.templateValue)

              func size(_ of: [ChatMessage], tools: [[String: Any]]?) -> Int? {
                guard !of.isEmpty,
                  let text = try? server.template.render(
                    messages: of, addGenerationPrompt: true, enableThinking: thinking,
                    reasoningEffort: effort, tools: tools),
                  let encoded = try? server.model().tokenizer.encode(text)
                else { return nil }
                return encoded.count
              }

              let withoutTools = size(messages, tools: nil)
              let bare = size(withoutSystem, tools: nil)
              if let withoutTools, let bare {
                let schemas = max(0, tokens.count - withoutTools)
                let instructions = max(0, withoutTools - bare)
                systemCount.set(schemas + instructions)
                server.log?(
                  "prompt: \(instructions) instructions + \(schemas) tool schemas "
                    + "+ \(bare) conversation = \(tokens.count)")
              } else if let bare {
                systemCount.set(max(0, tokens.count - bare))
              }
              _ = schema
            }

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
            continuation.resume(
              returning: AgentTurn(
                reasoning: outcome.parsed.reasoning,
                content: outcome.parsed.content,
                toolCalls: outcome.parsed.toolCalls,
                promptTokens: outcome.promptTokens,
                cachedTokens: server.sessions.lastReusedTokens,
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

  /// The packs the switcher offers, and the swap it performs. Activation is the server's, so a
  /// swap made here is the one the port sees too.
  public var catalog: ModelCatalog { server.catalog }

  public func activate(_ id: String) async throws {
    try await withCheckedThrowingContinuation { continuation in
      server.generationQueue.async { [self] in
        do {
          try server.activate(id)
          continuation.resume()
        } catch {
          continuation.resume(throwing: error)
        }
      }
    }
  }

  public var contextCeiling: Int { server.budget.maxContextTokens }

  public func readout() -> ServeReadout { server.readout() }
}
