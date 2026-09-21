// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The chat's state: a workspace, a loop over the resident pack, and what a turn cost.

import Foundation
import FoundationModels
import IshizukiAgent
import IshizukiKit
import Observation
import SwiftUI

@available(macOS 27.0, *)
@MainActor
@Observable
final class ChatController {
  /// One row in the conversation. The transcript is the source of these, read as it fills.
  struct Row: Identifiable {
    enum Kind {
      case prompt
      case reasoning
      case answer
      case toolCall(name: String)
      case toolOutput(name: String)
    }

    var id: String
    var kind: Kind
    var text: String
  }

  struct Meter {
    var turns = 0
    var seconds = 0.0
    var tokens = 0

    var averageSeconds: Double { turns > 0 ? seconds / Double(turns) : 0 }
    var averageTokens: Int { turns > 0 ? tokens / turns : 0 }
  }

  private(set) var rows: [Row] = []
  private(set) var isResponding = false
  private(set) var failure: String?
  private(set) var meter = Meter()
  /// What the last turn put in front of the model, so the dial can show the prefill against it.
  private(set) var lastPrompt = 0
  private(set) var lastCached = 0

  var draft = ""
  var workspace: URL? {
    didSet { agent = nil }
  }

  var effort: ReasoningEffort {
    didSet { engine?.effort = effort }
  }

  private let defaults = UserDefaults.standard
  private weak var server: ServerController?
  private var agent: CodingAgent?
  /// The server counts tokens for its whole life, so a turn's share is the difference.
  private var tokensAtTurnStart = 0
  private var turn: Task<Void, Never>?
  private var poller: Task<Void, Never>?

  init(server: ServerController) {
    self.server = server
    let stored = defaults.string(forKey: "chat.effort")
    self.effort = stored.flatMap(ReasoningEffort.init(rawValue:)) ?? .xhigh
    if let path = defaults.string(forKey: "chat.workspace") {
      self.workspace = URL(filePath: path)
    }
  }

  private var engine: AgentEngine? { server?.engine }

  var readout: ServeReadout? { server?.readout }

  /// The turn the engine is on, which is what the status line and the dial are both reading.
  var inFlight: ServeStats.Request? {
    readout?.inFlight.first { $0.phase == .prefill || $0.phase == .decode }
  }

  /// Nil once the prompt is in, which is how the status line knows to stop saying Reading.
  var prefillFraction: Double? {
    guard let request = inFlight, request.phase == .prefill, request.prefillTotal > 0 else {
      return nil
    }
    return Double(request.prefilled) / Double(request.prefillTotal)
  }

  var isGenerating: Bool { inFlight?.phase == .decode }

  var canSend: Bool {
    !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      && workspace != nil && engine != nil && !isResponding
  }

  /// Why the composer is closed, said plainly rather than left to be guessed at.
  var blocker: String? {
    if workspace == nil { return "Choose a folder to work in." }
    if engine == nil { return "Start the server to load a pack." }
    return nil
  }

  func chooseWorkspace() {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    panel.prompt = "Work Here"
    guard panel.runModal() == .OK, let url = panel.url else { return }
    workspace = url
    defaults.set(url.path, forKey: "chat.workspace")
  }

  func send() {
    let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard canSend, let agent = resolveAgent() else { return }
    draft = ""
    failure = nil
    isResponding = true
    tokensAtTurnStart = server?.readout?.totals.generatedTokens ?? 0
    startPolling(agent)

    let started = Date()
    turn = Task { [weak self] in
      do {
        _ = try await agent.send(text)
      } catch is CancellationError {
        // Stopping a turn is an ordinary thing to do, not a failure to report.
      } catch {
        self?.failure = error.localizedDescription
      }
      guard let self else { return }
      self.finish(agent, seconds: -started.timeIntervalSinceNow)
    }
  }

  func stop() {
    turn?.cancel()
  }

  /// Guidance for the turn after this one, which is what the conversation folds it into.
  func steer() {
    let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty, let agent else { return }
    agent.steer(text)
    draft = ""
    rows.append(Row(id: "steer-\(UUID().uuidString)", kind: .prompt, text: text))
  }

  private func resolveAgent() -> CodingAgent? {
    if let agent { return agent }
    guard let engine, let workspace else { return nil }
    engine.effort = effort
    let made = CodingAgent(
      engine: engine,
      workspace: Workspace(host: LocalShellHost(workspace: workspace)))
    agent = made
    return made
  }

  /// The transcript is read rather than mirrored: the executor fills it as the tokens land, so
  /// polling it is enough to show thinking, tool calls and the answer as they arrive.
  private func startPolling(_ agent: CodingAgent) {
    poller?.cancel()
    poller = Task { [weak self] in
      while !Task.isCancelled {
        self?.rows = Self.rows(from: agent.transcript)
        try? await Task.sleep(for: .milliseconds(120))
      }
    }
  }

  private func finish(_ agent: CodingAgent, seconds: Double) {
    poller?.cancel()
    poller = nil
    rows = Self.rows(from: agent.transcript)
    isResponding = false

    meter.turns += 1
    meter.seconds += seconds
    if let readout = server?.readout {
      meter.tokens += max(0, readout.totals.generatedTokens - tokensAtTurnStart)
      lastPrompt = readout.context.peakTokens
      lastCached = readout.prefix?.hits ?? 0
    }
  }

  func saveEffort() {
    defaults.set(effort.rawValue, forKey: "chat.effort")
  }

  private static func rows(from transcript: Transcript) -> [Row] {
    var rows: [Row] = []
    for entry in transcript {
      switch entry {
      case .instructions:
        continue
      case .prompt(let prompt):
        rows.append(Row(id: prompt.id, kind: .prompt, text: text(prompt.segments)))
      case .response(let response):
        let body = text(response.segments)
        if !body.isEmpty {
          rows.append(Row(id: response.id, kind: .answer, text: body))
        }
      case .reasoning(let reasoning):
        let body = text(reasoning.segments)
        if !body.isEmpty {
          rows.append(Row(id: reasoning.id, kind: .reasoning, text: body))
        }
      case .toolCalls(let calls):
        for call in calls {
          rows.append(
            Row(
              id: call.id, kind: .toolCall(name: call.toolName),
              text: call.arguments.jsonString))
        }
      case .toolOutput(let output):
        rows.append(
          Row(
            id: output.id, kind: .toolOutput(name: output.toolName),
            text: text(output.segments)))
      @unknown default:
        continue
      }
    }
    return rows
  }

  /// Thinking arrives with runs of blank lines in it, which read as a hole in the row rather
  /// than as breathing room. One blank line is a paragraph; more is an accident.
  private static func text(_ segments: [Transcript.Segment]) -> String {
    let joined =
      segments.compactMap { segment in
        switch segment {
        case .text(let text): return text.content
        case .structure(let structure): return structure.content.jsonString
        case .attachment(let attachment): return attachment.label
        @unknown default: return nil
        }
      }
      .joined(separator: "\n")

    return
      joined
      .replacing(/\n{3,}/, with: "\n\n")
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }
}
