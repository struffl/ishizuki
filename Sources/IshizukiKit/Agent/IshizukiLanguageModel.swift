// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The resident pack, presented as a Foundation Models model. Conforming here is what lets a
// session, its tools, its transcript and its guided generation all run on Bonsai instead.

import Foundation
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct IshizukiModel: LanguageModel {
  public typealias Executor = IshizukiExecutor

  public let executorConfiguration: IshizukiExecutor.Configuration

  public init(engine: AgentEngine) {
    self.executorConfiguration = IshizukiExecutor.Configuration(engine: engine)
  }

  public var capabilities: LanguageModelCapabilities {
    LanguageModelCapabilities([.toolCalling, .reasoning, .guidedGeneration, .vision])
  }

  public var engine: AgentEngine { executorConfiguration.engine }
}

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
public struct IshizukiExecutor: LanguageModelExecutor {
  public typealias Model = IshizukiModel

  public struct Configuration: Hashable, Sendable {
    public let engine: AgentEngine

    public static func == (a: Configuration, b: Configuration) -> Bool {
      a.engine === b.engine
    }

    public func hash(into hasher: inout Hasher) {
      hasher.combine(ObjectIdentifier(engine))
    }
  }

  private let configuration: Configuration

  public init(configuration: Configuration) throws {
    self.configuration = configuration
  }

  /// A prewarm is a prefill: the prefix cache keeps what this lays down, so the first turn of a
  /// conversation pays only for what the prompt adds.
  public func prewarm(model: Model, transcript: Transcript) {
    let messages = TranscriptBridge.messages(from: transcript)
    guard !messages.isEmpty else { return }
    let engine = configuration.engine
    Task.detached(priority: .utility) {
      _ = try? await engine.run(messages: messages, tools: [], onText: nil)
    }
  }

  public func respond(
    to request: LanguageModelExecutorGenerationRequest,
    model: Model,
    streamingInto channel: LanguageModelExecutorGenerationChannel
  ) async throws {
    let engine = configuration.engine
    let messages = TranscriptBridge.messages(from: request.transcript)
    let tools = TranscriptBridge.tools(from: request.enabledToolDefinitions)

    if let maximum = request.generationOptions.maximumResponseTokens {
      engine.maxTokens = maximum
    }

    let (pieces, emit) = AsyncStream<TurnPiece>.makeStream()
    let turn = Task {
      defer { emit.finish() }
      return try await engine.run(
        messages: messages, tools: tools,
        onText: { emit.yield(.content($0)) },
        onReasoning: { emit.yield(.reasoning($0)) })
    }

    var streamedReasoning = false
    var content = ""
    var reasoning = ""
    var lastFlush = ContinuousClock.now

    // Every send crosses into the session and has it rebuild its transcript, which at thirty
    // fragments a second it cannot keep up with — the window ends up showing four characters
    // of an answer that is a hundred tokens along. Fragments are gathered and handed over in
    // batches instead, which is the same text at a tenth of the traffic.
    func flush() async {
      if !reasoning.isEmpty {
        streamedReasoning = true
        await channel.send(.reasoning(action: .appendText(reasoning, tokenCount: 0)))
        reasoning = ""
      }
      if !content.isEmpty {
        await channel.send(.response(action: .appendText(content, tokenCount: 0)))
        content = ""
      }
    }

    for await piece in pieces {
      switch piece {
      case .content(let text): content += text
      case .reasoning(let text): reasoning += text
      }
      let now = ContinuousClock.now
      if lastFlush.duration(to: now) > .milliseconds(60) {
        await flush()
        lastFlush = now
      }
    }
    await flush()

    let outcome = try await turn.value

    // Only when nothing arrived live, so a stream and the final parse cannot both land.
    if !streamedReasoning, let reasoning = outcome.reasoning, !reasoning.isEmpty {
      await channel.send(.reasoning(action: .appendText(reasoning, tokenCount: 0)))
    }

    for call in outcome.toolCalls {
      await channel.send(
        .toolCalls(
          action: .toolCall(
            id: call.id, name: call.name,
            action: .appendArguments(call.argumentsJSON, tokenCount: 0))))
    }

    await channel.send(
      .response(
        action: .updateUsage(
          input: .init(
            totalTokenCount: outcome.promptTokens,
            cachedTokenCount: outcome.cachedTokens),
          output: .init(
            totalTokenCount: outcome.completionTokens,
            reasoningTokenCount: 0))))
  }
}

enum TurnPiece: Sendable {
  case content(String)
  case reasoning(String)
}

/// Turns a session's transcript into the messages the chat template renders, and its tool
/// definitions into the schemas that template spells out.
@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum TranscriptBridge {
  static func messages(from transcript: Transcript) -> [ChatMessage] {
    var messages: [ChatMessage] = []
    for entry in transcript {
      switch entry {
      case .instructions(let instructions):
        let text = plainText(instructions.segments)
        if !text.isEmpty { messages.append(.system(text)) }
      case .prompt(let prompt):
        messages.append(.user(plainText(prompt.segments)))
      case .response(let response):
        messages.append(.assistant(plainText(response.segments)))
      case .toolCalls(let calls):
        messages.append(
          .assistant(
            "",
            toolCalls: calls.map {
              ToolCall(id: $0.id, name: $0.toolName, argumentsJSON: $0.arguments.jsonString)
            }))
      case .toolOutput(let output):
        messages.append(.toolResult(plainText(output.segments)))
      // Replaying a previous turn's thinking is not what the template wants; it reopens it.
      case .reasoning:
        continue
      @unknown default:
        continue
      }
    }
    return messages
  }

  static func tools(from definitions: [Transcript.ToolDefinition]) -> [ToolSchema] {
    let encoder = JSONEncoder()
    return definitions.map { definition in
      let json =
        (try? encoder.encode(definition.parameters))
        .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
      return ToolSchema(
        name: definition.name, description: definition.description, parametersJSON: json)
    }
  }

  private static func plainText(_ segments: [Transcript.Segment]) -> String {
    segments.compactMap { segment in
      switch segment {
      case .text(let text): return text.content
      case .structure(let structure): return structure.content.jsonString
      case .attachment(let attachment): return attachment.label
      @unknown default: return nil
      }
    }
    .joined(separator: "\n")
  }
}
