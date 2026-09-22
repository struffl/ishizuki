// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The coding loop: a Foundation Models session whose model is the resident pack. The session
// resolves a turn's tool calls itself; what is here is the streaming, the steers waiting for
// the next prompt, and the events the window reads.

import Foundation
import FoundationModels
import IshizukiKit

@available(macOS 27.0, *)
public final class CodingAgent: Sendable {
  /// What answers the turn: the resident pack, or one of Apple's own models reached straight
  /// through the Foundation Models framework.
  public enum ModelChoice: Sendable {
    case resident(AgentEngine)
    case apple(AppleFoundationModel, reasoningLevel: AppleReasoningLevel, guardrails: AppleGuardrails)
  }

  public enum Event: Sendable {
    /// The answer so far, whole each time rather than as deltas, which is what the stream
    /// hands over and what a text view wants anyway.
    case content(String)
    case finished(content: String, seconds: Double)
    case failed(String)
  }

  public let workspace: Workspace
  /// Held so the window can observe the transcript: thinking, tool calls and their output all
  /// land here as the executor reports them.
  public let modelSession: LanguageModelSession
  public let events: AsyncStream<Event>

  private let emit: AsyncStream<Event>.Continuation
  private let pending = Steers()
  /// Set only for Private Cloud Compute, which takes this per turn rather than at construction.
  /// The on-device model and the resident pack have no equivalent and ignore it.
  private let reasoningLevel: AppleReasoningLevel?

  public init(
    model: ModelChoice,
    workspace: Workspace,
    instructions: String = CodingAgent.defaultInstructions,
    /// A conversation being resumed. Its transcript already carries the instructions it was
    /// started with, so they are not given again.
    transcript: Transcript? = nil
  ) {
    self.workspace = workspace

    let tools = codingTools(for: workspace)
    let session: LanguageModelSession
    switch model {
    case .resident(let engine):
      self.reasoningLevel = nil
      session = Self.makeSession(
        model: IshizukiModel(engine: engine), tools: tools, instructions: instructions,
        transcript: transcript)
    case .apple(.onDevice, _, let guardrails):
      self.reasoningLevel = nil
      let systemModel = SystemLanguageModel(
        guardrails: guardrails == .permissive ? .permissiveContentTransformations : .default)
      session = Self.makeSession(
        model: systemModel, tools: tools, instructions: instructions, transcript: transcript)
    case .apple(.privateCloudCompute, let level, _):
      self.reasoningLevel = level
      session = Self.makeSession(
        model: PrivateCloudComputeLanguageModel(), tools: tools, instructions: instructions,
        transcript: transcript)
    }
    // A turn that throws — a tool that could not run, a stop mid-answer, an inference
    // failure — must not take the conversation with it. The default winds the transcript back
    // past the prompt that started the turn, which reads in the window as the chat erasing
    // itself. Kept, a dead turn costs only its own tail.
    session.transcriptErrorHandlingPolicy = .preserveTranscript
    self.modelSession = session

    let (stream, continuation) = AsyncStream<Event>.makeStream()
    self.events = stream
    self.emit = continuation
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
    let started = Date()
    let prompt = pending.fold(into: text)
    let contextOptions = ContextOptions(
      reasoningLevel: reasoningLevel.map { $0 == .light ? .light : .deep })
    do {
      var latest = ""
      for try await snapshot in modelSession.streamResponse(
        to: prompt, contextOptions: contextOptions)
      {
        latest = snapshot.content
        emit.yield(.content(latest))
      }
      emit.yield(.finished(content: latest, seconds: -started.timeIntervalSinceNow))
      return latest
    } catch {
      emit.yield(.failed(error.localizedDescription))
      throw error
    }
  }

  /// Guidance for the turn after this one. It is not an interrupt: it goes in front of the
  /// next prompt rather than into the turn already running.
  public func steer(_ text: String) {
    pending.add(text)
  }

  public var isResponding: Bool { modelSession.isResponding }

  public var transcript: Transcript { modelSession.transcript }

  /// What was said while a turn was running, waiting for the prompt that follows it.
  private final class Steers: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []

    func add(_ text: String) {
      lock.lock()
      lines.append(text)
      lock.unlock()
    }

    func fold(into text: String) -> String {
      lock.lock()
      let waiting = lines
      lines.removeAll()
      lock.unlock()
      guard !waiting.isEmpty else { return text }
      return (waiting + [text]).joined(separator: "\n\n")
    }
  }

  /// Written for a small model on a slow machine: every line is either a rule about which tool
  /// to reach for or a rule about not reading more than it needs.
  public static let defaultInstructions = """
    You are a coding agent working in one directory. You change code by using tools, not by \
    describing changes.

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

    Answer briefly. The person can see the tool calls, so do not narrate them.
    """
}
