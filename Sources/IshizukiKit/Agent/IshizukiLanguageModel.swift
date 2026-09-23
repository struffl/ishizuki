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

  /// `tag` names the conversation, so its cache slot and archives are known to be its own;
  /// `effort` and `model` are fixed for the conversation because its prefix is built on both.
  public init(
    engine: AgentEngine, tag: String? = nil, effort: ReasoningEffort? = nil, model: String? = nil
  ) {
    self.executorConfiguration = IshizukiExecutor.Configuration(
      engine: engine, tag: tag, effort: effort, model: model)
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
    public let tag: String?
    public let effort: ReasoningEffort?
    public let model: String?

    public static func == (a: Configuration, b: Configuration) -> Bool {
      a.engine === b.engine && a.tag == b.tag && a.effort == b.effort && a.model == b.model
    }

    public func hash(into hasher: inout Hasher) {
      hasher.combine(ObjectIdentifier(engine))
      hasher.combine(tag)
      hasher.combine(effort)
      hasher.combine(model)
    }
  }

  private let configuration: Configuration

  public init(configuration: Configuration) throws {
    self.configuration = configuration
  }

  public func prewarm(model: Model, transcript: Transcript) {
    let configuration = configuration
    Task.detached(priority: .utility) {
      _ = await configuration.engine.readahead(
        transcript: transcript, tools: TranscriptBridge.tools(in: transcript),
        tag: configuration.tag, effort: configuration.effort, model: configuration.model)
    }
  }

  public func respond(
    to request: LanguageModelExecutorGenerationRequest,
    model: Model,
    streamingInto channel: LanguageModelExecutorGenerationChannel
  ) async throws {
    let engine = configuration.engine
    let messages = TranscriptBridge.messages(from: request.transcript, engine: engine)
    let tools = TranscriptBridge.tools(from: request.enabledToolDefinitions)
    let maxTokens = request.generationOptions.maximumResponseTokens
    let effort = configuration.effort
    let tag = configuration.tag
    let model = configuration.model

    let (pieces, emit) = AsyncStream<TurnPiece>.makeStream()
    let turn = Task {
      defer { emit.finish() }
      return try await engine.run(
        messages: messages, tools: tools, maxTokens: maxTokens, effort: effort, tag: tag,
        model: model,
        onText: { emit.yield(.content($0)) },
        onReasoning: { emit.yield(.reasoning($0)) })
    }

    try await withTaskCancellationHandler {
      try await stream(pieces, turn: turn, engine: engine, into: channel)
    } onCancel: {
      turn.cancel()
    }
  }

  private func stream(
    _ pieces: AsyncStream<TurnPiece>, turn: Task<AgentTurn, Error>, engine: AgentEngine,
    into channel: LanguageModelExecutorGenerationChannel
  ) async throws {
    // One id per kind for the whole turn, so every append and the replacement that ends it
    // land in the same entry.
    let responseID = UUID().uuidString
    let reasoningID = UUID().uuidString

    var streamedReasoning = false
    var content = ""
    var reasoning = ""
    var streamed = 0
    var lastFlush = ContinuousClock.now

    func flush() async {
      if !reasoning.isEmpty {
        streamedReasoning = true
        await channel.send(
          .reasoning(entryID: reasoningID, action: .appendText(reasoning, tokenCount: 0)))
        reasoning = ""
      }
      if !content.isEmpty {
        streamed += content.count
        await channel.send(
          .response(entryID: responseID, action: .appendText(content, tokenCount: 0)))
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

    // A stopped generation keeps what it had streamed and nothing else: a tool call cut off
    // mid-write must never reach the session, which would run it.
    if outcome.cancelled || Task.isCancelled { throw CancellationError() }

    if let reasoning = outcome.reasoning, !reasoning.isEmpty {
      let action: LanguageModelExecutorGenerationChannel.Reasoning.Action =
        streamedReasoning
        ? .replaceTextSegment(reasoning, tokenCount: 0)
        : .appendText(reasoning, tokenCount: 0)
      await channel.send(.reasoning(entryID: reasoningID, action: action))
    }
    if !outcome.content.isEmpty {
      await channel.send(
        .response(
          entryID: responseID,
          action: .replaceTextSegment(outcome.content, tokenCount: 0)))
    }

    if streamed != outcome.content.count {
      engine.server.log?(
        "stream: delivered \(streamed) of \(outcome.content.count) characters, "
          + "replaced with the parsed reply")
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

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
extension AgentEngine {
  /// Reads a session's conversation into the cache as its next turn will render it. A session
  /// with nothing said yet is only instructions and tool schemas, the prefix every conversation
  /// shares, so that one is also kept on disk.
  public func readahead(
    transcript: Transcript, tools: [Transcript.ToolDefinition], tag: String?,
    effort: ReasoningEffort?, model: String? = nil
  ) async -> (tokens: Int, reused: Int, finished: Bool) {
    let messages = TranscriptBridge.messages(from: transcript, engine: self)
    guard !messages.isEmpty else { return (0, 0, false) }
    let fresh = messages.allSatisfy { $0.role == "system" }
    return await readahead(
      messages: messages, tools: TranscriptBridge.tools(from: tools), effort: effort,
      tag: fresh ? "shared" : tag, pin: fresh, model: model)
  }
}

/// Turns a session's transcript into the messages the chat template renders, and its tool
/// definitions into the schemas that template spells out.
///
/// An assistant step is one message however many entries the session split it into: its
/// thinking, what it said and the calls it made, rendered back together exactly as generated.
@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum TranscriptBridge {
  static func messages(from transcript: Transcript, engine: AgentEngine? = nil) -> [ChatMessage] {
    var messages: [ChatMessage] = []
    var step: ChatMessage?

    func close() {
      if let step { messages.append(step) }
      step = nil
    }

    func open() -> ChatMessage {
      step ?? .assistant("", reasoning: "")
    }

    for entry in transcript {
      switch entry {
      case .instructions(let instructions):
        close()
        let text = plainText(instructions.segments)
        if !text.isEmpty { messages.append(.system(text)) }
      case .prompt(let prompt):
        close()
        let split = PromptAttachments.split(plainText(prompt.segments))
        messages.append(.user(text: split.body, imagePaths: split.images))
      case .reasoning(let reasoning):
        var message = open()
        if !message.toolCalls.isEmpty || !message.plainText.isEmpty {
          close()
          message = open()
        }
        let text = plainText(reasoning.segments)
        let held = message.reasoning ?? ""
        message.reasoning = held.isEmpty ? text : held + "\n" + text
        step = message
      case .response(let response):
        var message = open()
        if !message.toolCalls.isEmpty {
          close()
          message = open()
        }
        message.content = .text(message.plainText + plainText(response.segments))
        step = message
      case .toolCalls(let calls):
        var message = open()
        message.toolCalls += calls.map {
          ToolCall(
            id: $0.id, name: $0.toolName,
            argumentsJSON: engine?.spelling(ofCall: $0.id) ?? $0.arguments.jsonString)
        }
        step = message
      case .toolOutput(let output):
        close()
        messages.append(.toolResult(plainText(output.segments)))
      @unknown default:
        continue
      }
    }
    close()
    return messages
  }

  static func tools(from definitions: [Transcript.ToolDefinition]) -> [ToolSchema] {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return definitions.map { definition in
      let json =
        (try? encoder.encode(definition.parameters))
        .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
      return ToolSchema(
        name: definition.name, description: definition.description, parametersJSON: json)
    }
  }

  static func tools(in transcript: Transcript) -> [Transcript.ToolDefinition] {
    for entry in transcript {
      if case .instructions(let instructions) = entry { return instructions.toolDefinitions }
    }
    return []
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
