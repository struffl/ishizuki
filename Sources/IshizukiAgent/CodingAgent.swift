// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The coding loop: a Foundation Models session whose model is the resident pack. The session
// resolves a turn's tool calls itself; what is here is the prompt, the person's line into a
// running turn, and reading a conversation into the cache ahead of time.

import Foundation
import FoundationModels
import IshizukiKit

@available(macOS 27.0, *)
public final class CodingAgent: Sendable {
  /// What answers the turn: the resident pack, or one of Apple's own models reached straight
  /// through the Foundation Models framework.
  public enum ModelChoice: Sendable {
    case resident(AgentEngine, effort: ReasoningEffort, model: String?)
    case apple(AppleFoundationModel, reasoningLevel: AppleReasoningLevel, guardrails: AppleGuardrails)
  }

  public let workspace: Workspace
  /// Held so the window can observe the transcript: thinking, tool calls and their output all
  /// land here as the executor reports them.
  public let modelSession: LanguageModelSession
  /// Set only for Private Cloud Compute, which takes this per turn rather than at construction.
  private let reasoningLevel: AppleReasoningLevel?
  private let resident: (engine: AgentEngine, effort: ReasoningEffort, model: String?)?
  private let tag: String?
  private let definitions: [Transcript.ToolDefinition]
  private let environment: String?

  public init(
    model: ModelChoice,
    workspace: Workspace,
    instructions: String = CodingAgent.defaultInstructions,
    /// A conversation being resumed. Its transcript already carries the instructions it was
    /// started with, so they are not given again.
    transcript: Transcript? = nil,
    /// Names the conversation to the cache, so its prefix is kept and counted as its own.
    tag: String? = nil,
    /// Where and when, written at the head of the first prompt rather than into the
    /// instructions, which must stay the same for every conversation to share one cached copy.
    environment: String? = nil
  ) {
    self.workspace = workspace
    self.tag = tag
    self.environment = environment

    let tools = codingTools(for: workspace)
    self.definitions = tools.map { Transcript.ToolDefinition(tool: $0) }
    let session: LanguageModelSession
    switch model {
    case .resident(let engine, let effort, let pack):
      self.reasoningLevel = nil
      self.resident = (engine, effort, pack)
      session = Self.makeSession(
        model: IshizukiModel(engine: engine, tag: tag, effort: effort, model: pack),
        tools: tools, instructions: instructions, transcript: transcript)
    case .apple(.onDevice, _, let guardrails):
      self.reasoningLevel = nil
      self.resident = nil
      let systemModel = SystemLanguageModel(
        guardrails: guardrails == .permissive ? .permissiveContentTransformations : .default)
      session = Self.makeSession(
        model: systemModel, tools: tools, instructions: instructions, transcript: transcript)
    case .apple(.privateCloudCompute, let level, _):
      self.reasoningLevel = level
      self.resident = nil
      session = Self.makeSession(
        model: PrivateCloudComputeLanguageModel(), tools: tools, instructions: instructions,
        transcript: transcript)
    }
    // A turn that throws must not take the conversation with it: the default winds the
    // transcript back past the prompt that started the turn.
    session.transcriptErrorHandlingPolicy = .preserveTranscript
    self.modelSession = session
  }

  private static func makeSession(
    model: some LanguageModel, tools: [any Tool], instructions: String, transcript: Transcript?
  ) -> LanguageModelSession {
    if let transcript, !transcript.isEmpty {
      LanguageModelSession(model: model, tools: tools, transcript: transcript)
    } else {
      LanguageModelSession(model: model, tools: tools, instructions: Instructions(instructions))
    }
  }

  @discardableResult
  public func send(_ text: String) async throws -> String {
    let contextOptions = ContextOptions(
      reasoningLevel: reasoningLevel.map { $0 == .light ? .light : .deep })
    var latest = ""
    for try await snapshot in modelSession.streamResponse(
      to: prompt(for: text), contextOptions: contextOptions)
    {
      latest = snapshot.content
    }
    return latest
  }

  /// The environment goes in front of the first thing said, after any pictures' marker so the
  /// bridge still finds that at the very start.
  private func prompt(for text: String) -> String {
    let started = transcript.contains { if case .prompt = $0 { true } else { false } }
    guard let environment, !started else { return text }
    let split = PromptAttachments.split(text)
    return PromptAttachments.marker(for: split.images.map { URL(filePath: $0) }) + environment
      + split.body
  }

  /// Said while a turn runs. It reaches the model on the back of the next tool result; if the
  /// turn ends first it is still waiting here for whoever sends the next one.
  public func steer(_ steer: TurnInbox.Steer) {
    workspace.inbox.steer(steer)
  }

  public var inbox: TurnInbox { workspace.inbox }

  /// Reads the conversation into the resident pack's cache before anyone asks it anything, so
  /// the next turn starts from a warm prefix. Nothing to do for Apple's models.
  @discardableResult
  public func readahead() async -> (tokens: Int, reused: Int, finished: Bool)? {
    guard let resident else { return nil }
    return await resident.engine.readahead(
      transcript: modelSession.transcript, tools: definitions, tag: tag,
      effort: resident.effort, model: resident.model)
  }

  public var isResponding: Bool { modelSession.isResponding }

  public var transcript: Transcript { modelSession.transcript }

  /// Written for a small model on a slow machine: every line is either a rule about which tool
  /// to reach for or a rule about not reading more than it needs. Nothing in it changes between
  /// conversations, so every one of them can start from the same cached prefix.
  public static let defaultInstructions = """
    You are a coding agent working in one directory. You change code by using tools, not by \
    describing changes. The first message begins with an <environment> block saying which \
    folder you are in and what day it is.

    Finding things:
    - Use grep to find where something is, then read only those lines. Do not read a whole \
    file to get to twenty lines of it.
    - Use glob when you need to know which files exist.
    - A read hands back a slice and tells you how many lines it withheld. Ask again with an \
    offset when you need more.

    Changing things:
    - Use edit for a change to part of a file. You must have read the lines you are changing.
    - Use write only for a new file, or when you have read the whole of an existing one.
    - If a tool refuses, the refusal says what to do. Do that rather than trying the same \
    call again.

    Checking your work:
    - Use shell for builds, tests and git. Read its output before deciding what it means.
    - When a build or test fails, fix the cause rather than reporting the failure back.

    Commands that take a while:
    - A command still running after fifteen seconds is not killed. It keeps going in the \
    background and shell answers with a job id.
    - Start a server or a long build with background, so the turn is not spent waiting on it.
    - Use output with a job id to read what it has written since last time, and pass wait to \
    give it a few more seconds to finish. Use jobs to see what is still going.
    - Kill a job you are done with rather than leaving it running.

    Working with the person:
    - When a choice is theirs to make, or you need something no tool can find, use ask and \
    wait for the answer rather than guessing. Give options when there are a few clear ones.
    - Do not ask about what you can find out yourself, and do not ask to confirm routine work.
    - A tool result may end with a <steer> block. That is the person talking to you while \
    you work: it comes straight from them, and it overrides your plan where the two disagree.

    Answer briefly. The person can see the tool calls, so do not narrate them.
    """
}
