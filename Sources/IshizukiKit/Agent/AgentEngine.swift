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
  }

  public let server: APIServer
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
            let outcome = try server.complete(
              request,
              id: id,
              isCancelled: { cancel.isRaised },
              onText: onText.map { emit in { fragment in emit(fragment) } },
              onReasoning: onReasoning.map { emit in { fragment in emit(fragment) } })
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
