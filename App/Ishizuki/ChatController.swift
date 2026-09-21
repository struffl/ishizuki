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

      /// Which voice the row is in, which is what decides how tightly it sits under the row
      /// above it: a run of tool traffic reads as one block of machinery, not as six separate
      /// remarks, and only a change of voice earns a real gap.
      enum Voice {
        case mine, said, machinery
      }

      var voice: Voice {
        switch self {
        case .prompt, .steer: .mine
        case .answer: .said
        case .system, .reasoning, .toolCall, .toolOutput: .machinery
        }
      }
    }

    var id: String
    var kind: Kind
    var text: String

    /// A tool output carries the id of the call it answers, so the two rows need telling apart
    /// before anything keys on one: sharing an id, only the call survived the list's identity.
    static func outputID(_ callID: String) -> String { callID + "\u{2192}" }
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

  /// What the transcript draws. A queued steer has not been said yet, and a row that renders
  /// nothing still costs a line of height and a gap above it.
  var visibleRows: [Row] { transcriptRows.filter { $0.kind != .steer } }

  /// How a row is shown: open or shut, and whether a capped body has been let out in full.
  ///
  /// This lives here rather than in the row's own view because a `LazyVStack` discards a row's
  /// `@State` once it is far enough out of sight. A row that came back collapsed changed height
  /// underneath a scroll position that had been measured against it open, and the transcript
  /// jumped to make up the difference — which is the whole of what made scrolling bounce.
  nonisolated struct RowDisplay: Equatable {
    /// nil follows the default (open while live, shut once it settles); set the moment someone
    /// clicks, so a click mid-stream is not overruled on the next frame.
    var expanded: Bool?
    var showFull = false
  }

  private(set) var display: [String: RowDisplay] = [:]

  func display(for row: Row) -> RowDisplay { display[row.id] ?? RowDisplay() }

  func setExpanded(_ open: Bool, for id: String) {
    display[id, default: RowDisplay()].expanded = open
  }

  func setShowFull(_ full: Bool, for id: String) {
    display[id, default: RowDisplay()].showFull = full
  }

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

  func meta(for row: Row) -> RowMeta? { meta[row.id] }

  /// A turn in flight, held beside the conversation it belongs to rather than inside the one
  /// on screen. Switching away from a turn leaves it running; coming back picks it up where it
  /// has got to.
  private final class Run {
    let chatID: UUID
    let agent: CodingAgent
    let started = Date()
    var task: Task<Void, Never>?
    var poller: Task<Void, Never>?
    /// The rows folded so far, so a poll costs what has just arrived rather than the whole
    /// conversation.
    var builder = RowBuilder()
    var rowsThisTurn: Set<String> = []
    /// The server counts tokens for its whole life, so a turn's share is the difference.
    var totalsAtStart: ServeStats.Totals?
    var tokensAtStart = 0
    /// When the turn last hit disk, so a crash or a forced quit loses at most a few seconds of
    /// it rather than the whole thing.
    var lastCheckpoint = Date()

    init(chatID: UUID, agent: CodingAgent) {
      self.chatID = chatID
      self.agent = agent
    }
  }

  /// What a conversation looked like when it was left: enough to put it back as it was, kept
  /// only for one that is still being answered. Not observed, because nothing is drawing it —
  /// a turn running in a conversation you have switched away from costs the window nothing.
  private struct Parked {
    var rows: [Row]
    var meta: [String: RowMeta]
    var display: [String: RowDisplay]
    var steers: [Row]
    var failure: String?
  }

  @ObservationIgnored private var run: Run?
  @ObservationIgnored private var parked: [UUID: Parked] = [:]
  /// Bumped whenever a turn starts or ends, so the window notices a run it cannot otherwise
  /// see: the run itself is deliberately unobserved.
  private var runToken = 0

  /// Whether the conversation on screen is the one being answered. A turn in another
  /// conversation leaves this false: that one is running, this one can be read and scrolled.
  var isResponding: Bool {
    _ = runToken
    return run?.chatID == current.id
  }

  /// Whether any conversation is being answered. The pack answers one turn at a time, which is
  /// what stops a second conversation from sending while the first is still going.
  var isRunningTurn: Bool {
    _ = runToken
    return run != nil
  }

  func isRunning(_ chat: SavedChat) -> Bool {
    _ = runToken
    return run?.chatID == chat.id
  }
  private(set) var failure: String?
  private(set) var meter = Meter()
  /// What the last turn put in front of the model, so the dial can show the prefill against it.
  private(set) var lastPrompt = 0
  private(set) var lastCached = 0

  var draft = ""
  var workspace: URL? {
    didSet {
      // A session is bound to the folder it was made for, so changing the folder retires them
      // — except the one answering, which keeps the folder it started in.
      guard workspace != oldValue else { return }
      agents = agents.filter { $0.key == run?.chatID }
    }
  }

  var effort: ReasoningEffort {
    // Not while a turn is in flight: switching conversations sets this, and the turn already
    // running chose its own. What is set here reaches the engine when the next turn starts.
    didSet {
      guard !isRunningTurn else { return }
      engine?.effort = effort
    }
  }

  /// Every conversation that has been had, newest first, and which one is open.
  private(set) var chats: [SavedChat] = []
  private(set) var current: SavedChat

  let captioner = Captioner()
  private let store = ChatStore()
  private let defaults = UserDefaults.standard
  private weak var server: ServerController?
  /// One session per conversation, kept so switching away and back resumes the thread rather
  /// than rebuilding it from the transcript every time.
  @ObservationIgnored private var agents: [UUID: CodingAgent] = [:]

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
    var builder = RowBuilder()
    self.transcriptRows = builder.rows(from: current.transcript)

    NotificationCenter.default.addObserver(
      forName: NSApplication.willTerminateNotification, object: nil, queue: nil
    ) { [weak self] _ in self?.persistAll() }
  }

  // MARK: - Chats

  func startNewChat() {
    persist()
    let chat = SavedChat(
      workspace: workspace?.path, model: server?.settings.activeModelID, effort: effort)
    chats.insert(chat, at: 0)
    open(chat)
  }

  func select(_ chat: SavedChat) {
    guard chat.id != current.id else { return }
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
    // The one being answered stays: there is a turn writing into it.
    guard !isRunning(chat) else { return }
    agents[chat.id] = nil
    parked[chat.id] = nil
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
    park()
    current = chat
    effort = chat.effort
    if let path = chat.workspace { workspace = URL(filePath: path) }

    if let state = parked.removeValue(forKey: chat.id) {
      // A conversation that was left mid-answer comes back as it was, rows and all: its turn
      // has been folding into that copy the whole time it was away.
      transcriptRows = state.rows
      meta = state.meta
      display = state.display
      pendingSteers = state.steers
      failure = state.failure
    } else {
      var builder = RowBuilder()
      transcriptRows = builder.rows(from: agents[chat.id]?.transcript ?? chat.transcript)
      meta.removeAll()
      display.removeAll()
      pendingSteers.removeAll()
      failure = nil
    }
    if !chats.contains(where: { $0.id == chat.id }) { chats.insert(chat, at: 0) }
  }

  /// Set aside what the conversation being left looks like, but only while it is being
  /// answered: anything else is cheap enough to fold again from its transcript.
  private func park() {
    guard run?.chatID == current.id else { return }
    parked[current.id] = Parked(
      rows: transcriptRows, meta: meta, display: display, steers: pendingSteers,
      failure: failure)
  }

  private func update(_ id: UUID, _ change: (inout SavedChat) -> Void) {
    if current.id == id { change(&current) }
    guard let index = chats.firstIndex(where: { $0.id == id }) else { return }
    change(&chats[index])
  }

  /// Written after every turn, so closing the window is never a way to lose a conversation.
  private func persist() { persist(current.id) }

  /// Both the conversation on screen and the one being answered, for the one moment that has
  /// to catch both: the window going away.
  private func persistAll() {
    persist(current.id)
    if let id = run?.chatID, id != current.id { persist(id, prompt: true) }
  }

  /// A conversation is saved from its own session, which may not be the one on screen: a turn
  /// that finishes while another conversation is being read still has to land on disk.
  private func persist(_ id: UUID, prompt: Bool = false) {
    guard let agent = agents[id] else { return }
    let transcript = agent.transcript
    guard !transcript.isEmpty else { return }
    guard var saved = saved(id) else { return }
    saved.transcript = transcript
    saved.updated = Date()
    if id == current.id {
      saved.workspace = workspace?.path
      saved.model = server?.settings.activeModelID
      saved.effort = effort
    }
    // Only for the turn that just ran: the engine holds one last prompt, and it belongs to
    // whichever conversation was being answered.
    if prompt, let tokens = engine?.lastPromptTokens, !tokens.isEmpty {
      saved.promptTokens = tokens
    }
    if !saved.titleIsCustom, let derived = SavedChat.title(from: transcript) {
      saved.title = derived
    }
    captionTurn(for: id)
    update(id) { $0 = saved }
    chats.sort { $0.updated > $1.updated }
    store.save(saved)
  }

  private func saved(_ id: UUID) -> SavedChat? {
    id == current.id ? current : chats.first { $0.id == id }
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
    /// Another conversation is being answered. The pack takes one turn at a time, so this one
    /// waits rather than queueing into a busy engine.
    case busy
    case nothingToSay
  }

  var submission: Submission {
    if workspace == nil { return .chooseFolder }
    if isResponding { return .steer }
    if isRunningTurn { return .busy }
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
    case .busy: "Busy"
    case .send, .nothingToSay: "Send"
    }
  }

  var canSubmit: Bool {
    switch submission {
    case .loading, .busy, .nothingToSay: false
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
    case .loading, .busy, .nothingToSay: break
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
    case .busy: "Another conversation is being answered."
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
    guard !text.isEmpty, !isRunningTurn, let agent = resolveAgent() else { return }
    draft = ""
    failure = nil
    pendingSteers.removeAll()
    engine?.effort = effort

    let run = Run(chatID: current.id, agent: agent)
    run.totalsAtStart = server?.readout?.totals
    run.tokensAtStart = run.totalsAtStart?.generatedTokens ?? 0
    self.run = run
    runToken += 1
    startPolling(run)

    run.task = Task { [weak self] in
      do {
        _ = try await agent.send(text)
      } catch is CancellationError {
        // Stopping a turn is an ordinary thing to do, not a failure to report.
      } catch {
        self?.report(error.localizedDescription, for: run.chatID)
      }
      guard let self else { return }
      self.finish(run)
    }
  }

  func stop() {
    run?.task?.cancel()
  }

  /// A failure belongs to the conversation that earned it, not to whichever one is being read
  /// when it lands.
  private func report(_ message: String, for chatID: UUID) {
    if chatID == current.id {
      failure = message
    } else {
      parked[chatID]?.failure = message
    }
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
    run?.task?.cancel()
    let chatID = current.id
    Task { [weak self] in
      while self?.isRunningTurn == true {
        try? await Task.sleep(for: .milliseconds(60))
      }
      // Only into the conversation the queue belongs to: switching away while the turn unwinds
      // must not put it in front of a different one.
      guard let self, self.current.id == chatID else { return }
      self.send()
    }
  }

  private func resolveAgent() -> CodingAgent? {
    if let agent = agents[current.id] { return agent }
    guard let engine, let workspace else { return nil }
    engine.effort = effort
    let made = CodingAgent(
      engine: engine,
      workspace: Workspace(host: LocalShellHost(workspace: workspace)),
      transcript: current.transcript.isEmpty ? nil : current.transcript)
    agents[current.id] = made
    return made
  }

  /// The transcript is read rather than mirrored: the executor fills it as the tokens land, so
  /// polling it is enough to show thinking, tool calls and the answer as they arrive.
  private func startPolling(_ run: Run) {
    run.poller?.cancel()
    run.lastCheckpoint = Date()
    run.poller = Task { [weak self] in
      while !Task.isCancelled {
        self?.absorbTranscript(of: run)
        self?.checkpoint(run)
        try? await Task.sleep(for: .milliseconds(50))
      }
    }
  }

  /// A turn can run for minutes; writing only once it finishes means quitting or crashing
  /// mid-turn loses all of it. This writes the transcript as it stands every few seconds, so
  /// the worst a forced exit costs is the last stretch of one response.
  private func checkpoint(_ run: Run) {
    guard Date().timeIntervalSince(run.lastCheckpoint) >= 3 else { return }
    run.lastCheckpoint = Date()
    let transcript = run.agent.transcript
    guard !transcript.isEmpty, var saved = saved(run.chatID) else { return }
    saved.transcript = transcript
    saved.updated = Date()
    if run.chatID == current.id {
      saved.workspace = workspace?.path
      saved.model = server?.settings.activeModelID
      saved.effort = effort
    }
    // Encoding a long transcript is not something a turn should stop for: the window is trying
    // to draw tokens while this runs.
    let directory = store.folder
    Task.detached(priority: .utility) { ChatStore.write(saved, in: directory) }
  }

  private func finish(_ run: Run) {
    run.poller?.cancel()
    run.poller = nil
    absorbTranscript(of: run)
    if self.run === run {
      self.run = nil
      runToken += 1
    }
    recordTurnCost(run)
    persist(run.chatID, prompt: true)

    meter.turns += 1
    meter.seconds += -run.started.timeIntervalSinceNow
    if let readout = server?.readout {
      meter.tokens += max(0, readout.totals.generatedTokens - run.tokensAtStart)
      lastPrompt = readout.context.peakTokens
      lastCached = readout.prefix?.hits ?? 0
    }
  }

  private func absorbTranscript(of run: Run) {
    absorb(run.builder.rows(from: run.agent.transcript), for: run)
  }

  /// A row's clock starts the first time it is seen, which is as close to when it happened as
  /// a transcript without timestamps allows.
  ///
  /// Where the rows land depends on whether the turn is the one being watched: the conversation
  /// on screen takes them through the observed properties, and one left running takes them into
  /// its parked copy, which draws nothing until it is opened again.
  private func absorb(_ rows: [Row], for run: Run) {
    if run.chatID == current.id {
      // Assigning an identical array would still be a change to everything watching it, and at
      // twenty polls a second that is a re-render of the whole transcript for nothing.
      guard rows != transcriptRows else { return }
      transcriptRows = rows
      for row in rows where meta[row.id] == nil {
        meta[row.id] = RowMeta(at: Date(), wasRead: row.kind.isInput)
        run.rowsThisTurn.insert(row.id)
      }
    } else {
      guard var state = parked[run.chatID], rows != state.rows else { return }
      state.rows = rows
      for row in rows where state.meta[row.id] == nil {
        state.meta[row.id] = RowMeta(at: Date(), wasRead: row.kind.isInput)
        run.rowsThisTurn.insert(row.id)
      }
      parked[run.chatID] = state
    }
  }

  /// The turn's cost, split the way the engine splits it and handed to the rows it produced.
  private func recordTurnCost(_ run: Run) {
    guard let totals = server?.readout?.totals, let start = run.totalsAtStart else { return }
    let read = max(0, totals.prefillSeconds - start.prefillSeconds)
    let wrote = max(0, totals.decodeSeconds - start.decodeSeconds)
    let promptTokens = max(0, totals.promptTokens - start.promptTokens)
    let generated = max(0, totals.generatedTokens - start.generatedTokens)

    withMeta(of: run.chatID) { meta in
      for id in run.rowsThisTurn {
        guard var record = meta[id] else { continue }
        record.seconds = record.wasRead ? read : wrote
        record.tokens = record.wasRead ? promptTokens : generated
        meta[id] = record
      }
    }
    run.rowsThisTurn.removeAll()
  }

  private func withMeta(of chatID: UUID, _ change: (inout [String: RowMeta]) -> Void) {
    if chatID == current.id {
      change(&meta)
    } else if parked[chatID] != nil {
      change(&parked[chatID]!.meta)
    }
  }

  /// Captions are asked for once the turn has settled, so the system model is not being asked
  /// to describe a sentence that is still being written.
  private func captionTurn(for chatID: UUID) {
    let rows = chatID == current.id ? transcriptRows : (parked[chatID]?.rows ?? [])
    for row in rows {
      switch row.kind {
      case .reasoning:
        captioner.request(row.id, text: row.text, as: .thought)
      case .toolCall(let name):
        captioner.request(row.id, text: row.text, as: .command(tool: name))
      default:
        continue
      }
    }
    guard saved(chatID)?.titleIsCustom == false else { return }
    let said =
      rows
      .filter { if case .prompt = $0.kind { true } else { false } }
      .map(\.text)
      .joined(separator: " ")
    let key = "title-\(chatID.uuidString)"
    captioner.request(key, text: said, as: .conversation) { [weak self] written in
      guard let self, self.saved(chatID)?.titleIsCustom == false else { return }
      self.update(chatID) { $0.title = written }
      if let chat = self.saved(chatID) { self.store.save(chat) }
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
        let said = text(response.segments)
        // A reply that is really an unclosed thought is shown as one. The parser keeps the two
        // apart now, but conversations saved before it did still hold answers that are nothing
        // but a chain of reasoning, and drawing that as prose put thousands of points of it in
        // the transcript. Folded as reasoning it joins the thought beside it and stays shut.
        if said.hasPrefix("<think>") {
          add(
            response.id, .reasoning,
            String(said.dropFirst("<think>".count))
              .trimmingCharacters(in: .whitespacesAndNewlines))
        } else {
          add(response.id, .answer, said)
        }
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
        add(Row.outputID(output.id), .toolOutput(name: output.toolName), text(output.segments))
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
