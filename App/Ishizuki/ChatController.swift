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
  nonisolated struct Row: Identifiable, Equatable {
    enum Kind: Equatable {
      case system
      case prompt
      case steer
      case reasoning
      case answer
      case toolCall(name: String)
      case toolOutput(name: String)

      /// Everything the model had to read before it could answer, which is what the token
      /// count on a prompt row is explaining.
      /// Whether a following entry of this kind is a continuation of the same row rather than
      /// a new one. Only the streamed prose continues; a second tool call is its own row.
      func joins(_ next: Kind) -> Bool {
        switch (self, next) {
        case (.reasoning, .reasoning), (.answer, .answer): true
        case (.toolOutput(let a), .toolOutput(let b)): a == b
        default: false
        }
      }

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

  /// Every conversation that has been had, newest first, and which one is open.
  private(set) var chats: [SavedChat] = []
  private(set) var current: SavedChat

  let captioner = Captioner()
  private let store = ChatStore()
  private let defaults = UserDefaults.standard
  private weak var server: ServerController?
  private var agent: CodingAgent?
  /// The server counts tokens for its whole life, so a turn's share is the difference.
  private var tokensAtTurnStart = 0
  private var turn: Task<Void, Never>?
  private var poller: Task<Void, Never>?
  /// The rows folded so far, so a poll costs what has just arrived rather than the whole
  /// conversation.
  private var builder = RowBuilder()
  /// When a turn in flight last hit disk, so a crash or a forced quit loses at most a few
  /// seconds of it rather than the whole thing.
  private var lastCheckpoint = Date.distantPast

  init(server: ServerController) {
    self.server = server
    let stored = defaults.string(forKey: "chat.effort")
    let effort = stored.flatMap(ReasoningEffort.init(rawValue:)) ?? .xhigh
    self.effort = effort

    let loaded = ChatStore().load()
    self.chats = loaded
    self.current = loaded.first ?? SavedChat(effort: effort)
    if let path = current.workspace ?? defaults.string(forKey: "chat.workspace") {
      self.workspace = URL(filePath: path)
    }
    if chats.isEmpty { chats = [current] }
    self.transcriptRows = builder.rows(from: current.transcript)

    NotificationCenter.default.addObserver(
      forName: NSApplication.willTerminateNotification, object: nil, queue: nil
    ) { [weak self] _ in self?.persist() }
  }

  // MARK: - Chats

  func startNewChat() {
    guard !isResponding else { return }
    persist()
    let chat = SavedChat(
      workspace: workspace?.path, model: server?.settings.activeModelID, effort: effort)
    chats.insert(chat, at: 0)
    open(chat)
  }

  func select(_ chat: SavedChat) {
    guard !isResponding, chat.id != current.id else { return }
    persist()
    open(chat)
  }

  func rename(_ chat: SavedChat, to title: String) {
    let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return }
    update(chat.id) {
      $0.title = trimmed
      $0.titleIsCustom = true
    }
    store.save(chats.first { $0.id == chat.id } ?? current)
  }

  func delete(_ chat: SavedChat) {
    guard !(isResponding && chat.id == current.id) else { return }
    store.delete(chat.id)
    chats.removeAll { $0.id == chat.id }
    if let prefixes = server?.prefixStore {
      store.pruneCache(for: chat, keeping: chats, in: prefixes)
    }
    guard chat.id == current.id else { return }
    open(chats.first ?? SavedChat(workspace: workspace?.path, effort: effort))
  }

  /// Opening a conversation rebuilds its session from the transcript it was saved with, so the
  /// model picks up the thread rather than being told about it.
  private func open(_ chat: SavedChat) {
    current = chat
    agent = nil
    failure = nil
    pendingSteers.removeAll()
    meta.removeAll()
    effort = chat.effort
    if let path = chat.workspace { workspace = URL(filePath: path) }
    builder = RowBuilder()
    transcriptRows = builder.rows(from: chat.transcript)
    if !chats.contains(where: { $0.id == chat.id }) { chats.insert(chat, at: 0) }
  }

  private func update(_ id: UUID, _ change: (inout SavedChat) -> Void) {
    if current.id == id { change(&current) }
    guard let index = chats.firstIndex(where: { $0.id == id }) else { return }
    change(&chats[index])
  }

  /// Written after every turn, so closing the window is never a way to lose a conversation.
  private func persist() {
    guard let agent else { return }
    let transcript = agent.transcript
    guard !transcript.isEmpty else { return }
    current.transcript = transcript
    current.updated = Date()
    current.workspace = workspace?.path
    current.model = server?.settings.activeModelID
    current.effort = effort
    if let tokens = engine?.lastPromptTokens, !tokens.isEmpty {
      current.promptTokens = tokens
    }
    if !current.titleIsCustom, let derived = SavedChat.title(from: transcript) {
      current.title = derived
    }
    captionTurn()
    let saved = current
    update(saved.id) { $0 = saved }
    chats.sort { $0.updated > $1.updated }
    store.save(saved)
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

  /// The command being written, for the gap between a thought ending and a call landing.
  var writingCommand: String {
    (engine?.writingCommand ?? "")
      .replacing(/<\/?(function|parameter)[^>]*>/, with: " ")
      .replacing(/\s+/, with: " ")
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }

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
    builder = RowBuilder()
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
      workspace: Workspace(host: LocalShellHost(workspace: workspace)),
      transcript: current.transcript.isEmpty ? nil : current.transcript)
    agent = made
    return made
  }

  /// The transcript is read rather than mirrored: the executor fills it as the tokens land, so
  /// polling it is enough to show thinking, tool calls and the answer as they arrive.
  private func startPolling(_ agent: CodingAgent) {
    poller?.cancel()
    lastCheckpoint = Date()
    poller = Task { [weak self] in
      while !Task.isCancelled {
        self?.absorbTranscript(of: agent)
        self?.checkpoint(agent)
        try? await Task.sleep(for: .milliseconds(50))
      }
    }
  }

  /// A turn can run for minutes; writing only once it finishes means quitting or crashing
  /// mid-turn loses all of it. This writes the transcript as it stands every few seconds, so
  /// the worst a forced exit costs is the last stretch of one response.
  private func checkpoint(_ agent: CodingAgent) {
    guard Date().timeIntervalSince(lastCheckpoint) >= 3 else { return }
    lastCheckpoint = Date()
    let transcript = agent.transcript
    guard !transcript.isEmpty else { return }
    var saved = current
    saved.transcript = transcript
    saved.updated = Date()
    saved.workspace = workspace?.path
    saved.model = server?.settings.activeModelID
    saved.effort = effort
    // Encoding a long transcript is not something a turn should stop for: the window is trying
    // to draw tokens while this runs.
    let directory = store.folder
    Task.detached(priority: .utility) { ChatStore.write(saved, in: directory) }
  }

  private func finish(_ agent: CodingAgent, seconds: Double) {
    poller?.cancel()
    poller = nil
    absorbTranscript(of: agent)
    isResponding = false
    recordTurnCost()
    persist()

    meter.turns += 1
    meter.seconds += seconds
    if let readout = server?.readout {
      meter.tokens += max(0, readout.totals.generatedTokens - tokensAtTurnStart)
      lastPrompt = readout.context.peakTokens
      lastCached = readout.prefix?.hits ?? 0
    }
  }

  private func absorbTranscript(of agent: CodingAgent) {
    absorb(builder.rows(from: agent.transcript))
  }

  /// A row's clock starts the first time it is seen, which is as close to when it happened as
  /// a transcript without timestamps allows.
  private func absorb(_ rows: [Row]) {
    // Assigning an identical array would still be a change to everything watching it, and at
    // twenty polls a second that is a re-render of the whole transcript for nothing.
    guard rows != transcriptRows else { return }
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

  /// Captions are asked for once the turn has settled, so the system model is not being asked
  /// to describe a sentence that is still being written.
  private func captionTurn() {
    for row in transcriptRows {
      switch row.kind {
      case .reasoning:
        captioner.request(row.id, text: row.text, as: .thought)
      case .toolCall(let name):
        captioner.request(row.id, text: row.text, as: .command(tool: name))
      default:
        continue
      }
    }
    guard !current.titleIsCustom else { return }
    let said =
      transcriptRows
      .filter { if case .prompt = $0.kind { true } else { false } }
      .map(\.text)
      .joined(separator: " ")
    let chatID = current.id
    let key = "title-\(chatID.uuidString)"
    captioner.request(key, text: said, as: .conversation) { [weak self] written in
      guard let self, self.current.id == chatID, !self.current.titleIsCustom else { return }
      self.update(chatID) { $0.title = written }
      self.store.save(self.current)
    }
  }

  func saveEffort() {
    defaults.set(effort.rawValue, forKey: "chat.effort")
  }

  /// Read straight through, appending rather than replacing. The session is free to split a
  /// streamed reply across as many entries as it likes, so consecutive entries of the same
  /// kind are joined into one row: whatever was generated is shown, however it arrived.
  nonisolated private static func fold(
    _ entries: some Sequence<Transcript.Entry>, into existing: [Row]
  ) -> [Row] {
    var rows = existing

    func add(_ id: String, _ kind: Row.Kind, _ text: String) {
      guard !text.isEmpty else { return }
      if let last = rows.last, last.kind.joins(kind) {
        rows[rows.count - 1].text += text
        return
      }
      rows.append(Row(id: id, kind: kind, text: text))
    }

    for entry in entries {
      switch entry {
      case .instructions(let instructions):
        add(instructions.id, .system, text(instructions.segments))
      case .prompt(let prompt):
        add(prompt.id, .prompt, text(prompt.segments))
      case .response(let response):
        add(response.id, .answer, text(response.segments))
      case .reasoning(let reasoning):
        add(reasoning.id, .reasoning, text(reasoning.segments))
      case .toolCalls(let calls):
        for call in calls {
          rows.append(
            Row(
              id: call.id, kind: .toolCall(name: call.toolName),
              text: call.arguments.jsonString))
        }
      case .toolOutput(let output):
        add(output.id, .toolOutput(name: output.toolName), text(output.segments))
      @unknown default:
        continue
      }
    }
    return rows
  }

  /// A transcript only grows, and only its last entry is still being written into, so every
  /// entry before that one can be folded once and kept. Reading the whole thing twenty times a
  /// second was the transcript's own length being copied and re-joined on the main thread for
  /// every fifty milliseconds of a turn, which is what made a long conversation stutter.
  struct RowBuilder {
    private var settled: [Row] = []
    private var folded = 0

    mutating func rows(from transcript: Transcript) -> [Row] {
      let entries = Array(transcript)
      if entries.count < folded {
        settled = []
        folded = 0
      }
      let stable = max(0, entries.count - 1)
      if stable > folded {
        settled = ChatController.fold(entries[folded..<stable], into: settled)
        folded = stable
      }
      guard stable < entries.count else { return settled }
      return ChatController.fold(entries[stable...], into: settled)
    }
  }

  nonisolated private static func text(_ segments: [Transcript.Segment]) -> String {
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
