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
      case system
      case prompt
      case steer
      case reasoning
      case answer
      case toolCall(name: String)
      case toolOutput(name: String)

      /// Everything the model had to read before it could answer, which is what the token
      /// count on a prompt row is explaining.
      var isInput: Bool {
        switch self {
        case .system, .prompt, .steer, .toolOutput: true
        case .reasoning, .answer, .toolCall: false
        }
      }
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

  /// The transcript's own rows, replaced wholesale as it fills.
  private(set) var transcriptRows: [Row] = []
  /// Steering waits for the next turn, so it is not in the transcript yet and has to be held
  /// here or the next poll would wipe it.
  private(set) var pendingSteers: [Row] = []

  var rows: [Row] { transcriptRows }

  /// What a row cost, kept beside the transcript because the transcript carries no clock and
  /// no token counts of its own.
  struct RowMeta: Equatable {
    var at: Date
    var seconds: Double?
    var tokens: Int?
    /// Reading for what went in, writing for what came out, which decides the wording.
    var wasRead: Bool
  }

  private(set) var meta: [String: RowMeta] = [:]
  private var rowsThisTurn: Set<String> = []
  private var turnStart: ServeStats.Totals?

  func meta(for row: Row) -> RowMeta? { meta[row.id] }
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

  /// The instructions' share of the prompt, so the reading bar can show what is the same every
  /// turn and what is this turn's own.
  var systemTokens: Int { engine?.systemTokens ?? 0 }

  /// What the turn is doing, taken from the phase rather than guessed at. Writing used to be
  /// what the status line said whenever it knew nothing, which is how it came to say Writing
  /// through an entire prefill.
  enum Activity: Equatable {
    case queued
    case reading(Double?)
    case writing
    /// Still generating, but into a tool call rather than into an answer.
    case writingCommand
    case unknown
  }

  var activity: Activity {
    guard let request = inFlight else { return .unknown }
    switch request.phase {
    case .queued: return .queued
    case .prefill: return .reading(prefillFraction)
    case .decode, .finishing:
      return engine?.isWritingToolCall == true ? .writingCommand : .writing
    }
  }

  /// What pressing return does. Nothing here is a dead end: if the pack is not up, return
  /// brings it up and sends what was typed once it is.
  enum Submission: Equatable {
    case chooseFolder
    case load
    case loading
    case send
    case steer
    case nothingToSay
  }

  var submission: Submission {
    if workspace == nil { return .chooseFolder }
    if isResponding { return .steer }
    if server?.phase.isBusy == true { return .loading }
    if engine == nil { return .load }
    if !typed.isEmpty || !pendingSteers.isEmpty { return .send }
    return .nothingToSay
  }

  var submissionLabel: String {
    switch submission {
    case .chooseFolder: "Choose"
    case .load: "Load"
    case .loading: "Loading"
    case .steer: "Steer"
    case .send, .nothingToSay: "Send"
    }
  }

  var canSubmit: Bool {
    switch submission {
    case .loading, .nothingToSay: false
    case .steer: !typed.isEmpty
    case .chooseFolder, .load, .send: true
    }
  }

  func submit() {
    switch submission {
    case .chooseFolder: chooseWorkspace()
    case .load: load()
    case .send: send()
    case .steer: steer()
    case .loading, .nothingToSay: break
    }
  }

  private var typed: String {
    draft.trimmingCharacters(in: .whitespacesAndNewlines)
  }

  /// Why the transcript is empty, said plainly rather than left to be guessed at.
  var blocker: String? {
    switch submission {
    case .chooseFolder: "Choose a folder to work in."
    case .load: "Press Load to bring the pack up."
    case .loading: "Bringing the pack up…"
    default: nil
    }
  }

  private func load() {
    guard let server else { return }
    failure = nil
    server.start()
    Task { await self.sendOnceLoaded() }
  }

  /// Pressing Load with something already typed should send it, rather than asking for the
  /// same keystroke twice.
  private func sendOnceLoaded() async {
    while !Task.isCancelled {
      guard let server else { return }
      switch server.phase {
      case .failed(let message):
        failure = message
        return
      case .running:
        if engine != nil {
          if !typed.isEmpty { send() }
          return
        }
      case .stopped:
        return
      case .starting:
        break
      }
      try? await Task.sleep(for: .milliseconds(150))
    }
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
    // Whatever was queued while the last turn ran goes out ahead of what was just typed.
    let text = (pendingSteers.map(\.text) + [typed])
      .filter { !$0.isEmpty }
      .joined(separator: "\n\n")
    guard !text.isEmpty, !isResponding, let agent = resolveAgent() else { return }
    draft = ""
    failure = nil
    pendingSteers.removeAll()
    isResponding = true
    rowsThisTurn.removeAll()
    turnStart = server?.readout?.totals
    tokensAtTurnStart = turnStart?.generatedTokens ?? 0
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

  /// Guidance for the turn after this one. Queued here rather than inside the conversation,
  /// which has no way to hand a queued message back — and sending it early means taking it
  /// out of the queue first.
  func steer() {
    let text = typed
    guard !text.isEmpty else { return }
    draft = ""
    pendingSteers.append(Row(id: "steer-\(UUID().uuidString)", kind: .steer, text: text))
  }

  func drop(_ row: Row) {
    pendingSteers.removeAll { $0.id == row.id }
  }

  /// Cut the turn in flight short and send what is queued now. Cancelling is not instant, so
  /// this waits for the turn to unwind rather than sending into a busy engine.
  func sendQueuedNow() {
    guard !pendingSteers.isEmpty else { return }
    guard isResponding else {
      send()
      return
    }
    turn?.cancel()
    Task { [weak self] in
      while self?.isResponding == true {
        try? await Task.sleep(for: .milliseconds(60))
      }
      self?.send()
    }
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
        self?.absorb(Self.rows(from: agent.transcript))
        try? await Task.sleep(for: .milliseconds(50))
      }
    }
  }

  private func finish(_ agent: CodingAgent, seconds: Double) {
    poller?.cancel()
    poller = nil
    absorb(Self.rows(from: agent.transcript))
    isResponding = false
    recordTurnCost()

    meter.turns += 1
    meter.seconds += seconds
    if let readout = server?.readout {
      meter.tokens += max(0, readout.totals.generatedTokens - tokensAtTurnStart)
      lastPrompt = readout.context.peakTokens
      lastCached = readout.prefix?.hits ?? 0
    }
  }

  /// A row's clock starts the first time it is seen, which is as close to when it happened as
  /// a transcript without timestamps allows.
  private func absorb(_ rows: [Row]) {
    transcriptRows = rows
    for row in rows where meta[row.id] == nil {
      meta[row.id] = RowMeta(at: Date(), wasRead: row.kind.isInput)
      if isResponding { rowsThisTurn.insert(row.id) }
    }
  }

  /// The turn's cost, split the way the engine splits it and handed to the rows it produced.
  private func recordTurnCost() {
    guard let totals = server?.readout?.totals, let start = turnStart else { return }
    let read = max(0, totals.prefillSeconds - start.prefillSeconds)
    let wrote = max(0, totals.decodeSeconds - start.decodeSeconds)
    let promptTokens = max(0, totals.promptTokens - start.promptTokens)
    let generated = max(0, totals.generatedTokens - start.generatedTokens)

    for id in rowsThisTurn {
      guard var record = meta[id] else { continue }
      record.seconds = record.wasRead ? read : wrote
      record.tokens = record.wasRead ? promptTokens : generated
      meta[id] = record
    }
    rowsThisTurn.removeAll()
  }

  func saveEffort() {
    defaults.set(effort.rawValue, forKey: "chat.effort")
  }

  private static func rows(from transcript: Transcript) -> [Row] {
    var rows: [Row] = []
    for entry in transcript {
      switch entry {
      case .instructions(let instructions):
        let body = text(instructions.segments)
        if !body.isEmpty {
          rows.append(Row(id: instructions.id, kind: .system, text: body))
        }
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
