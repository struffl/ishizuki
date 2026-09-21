// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The coding loop: a Foundation Models session whose model is the resident pack, driven by
// SwiftAgent's conversation so a turn's tool calls resolve without the window arranging them.

import Foundation
import FoundationModels
import IshizukiKit
import SwiftAgent

@available(macOS 27.0, *)
public final class CodingAgent: Sendable {
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

  private let conversation: Conversation
  private let emit: AsyncStream<Event>.Continuation

  public init(
    engine: AgentEngine,
    workspace: Workspace,
    instructions: String = CodingAgent.defaultInstructions
  ) {
    self.workspace = workspace

    let session = LanguageModelSession(
      model: IshizukiModel(engine: engine),
      tools: codingTools(for: workspace),
      instructions: Instructions(instructions))
    self.modelSession = session

    let (stream, continuation) = AsyncStream<Event>.makeStream()
    self.events = stream
    self.emit = continuation

    self.conversation = Conversation(languageModelSession: session) {
      GenerateText<Prompt>(
        session: session,
        prompt: { $0 },
        onStream: { snapshot in
          continuation.yield(.content(snapshot.content))
        })
    }
  }

  @discardableResult
  public func send(_ text: String) async throws -> String {
    do {
      let response = try await conversation.send(text)
      emit.yield(
        .finished(
          content: response.content,
          seconds: Double(response.duration.components.seconds)))
      return response.content
    } catch {
      emit.yield(.failed(error.localizedDescription))
      throw error
    }
  }

  /// Guidance for the turn after this one. It is not an interrupt: the conversation folds it
  /// into the next prompt rather than into the turn already running.
  public func steer(_ text: String) {
    conversation.steer(text)
  }

  public var isResponding: Bool { conversation.isResponding }

  public var transcript: Transcript { modelSession.transcript }

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

    Answer briefly. The person can see the tool calls, so do not narrate them.
    """
}
