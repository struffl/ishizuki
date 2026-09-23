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
      /// What became of a turn when it was not an answer: stopped, or failed.
      case notice(tone: ChatNotice.Tone)

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
        case .reasoning, .answer, .toolCall, .notice: false
        }
      }

      /// A step rather than something said. A run of these is what the transcript folds away
      /// into one line; a thought is machinery too, but it is worth reading and stays out.
      var isMachinery: Bool {
        switch self {
        case .toolCall(let name), .toolOutput(let name): name != ChatController.askTool
        default: false
        }
      }

      /// Which voice the row is in, which is what decides how tightly it sits under the row
      /// above it: a run of tool traffic reads as one block of machinery, not as six separate
      /// remarks, and only a change of voice earns a real gap.
      enum Voice {
        case mine, said, machinery, aside
      }

      var voice: Voice {
        switch self {
        case .prompt, .steer: .mine
        case .answer: .said
        case .toolCall(let name) where name == ChatController.askTool: .said
        case .toolOutput(let name) where name == ChatController.askTool: .mine
        case .system, .reasoning, .toolCall, .toolOutput: .machinery
        // Its own voice, so it never closes up against the row it is explaining.
        case .notice: .aside
        }
      }
    }

    var id: String
    var kind: Kind
    var text: String
    /// The pictures handed over with a prompt, as paths. Kept on the row so the transcript can
    /// show what was attached without going back to the marker the prompt travelled with.
    var images: [String] = []

    /// A tool output carries the id of the call it answers, so the two rows need telling apart
    /// before anything keys on one: sharing an id, only the call survived the list's identity.
    static func outputID(_ callID: String) -> String { callID + "\u{2192}" }
  }

  nonisolated static let askTool = "ask"

  /// The transcript's own rows, replaced wholesale as it fills.
  private(set) var transcriptRows: [Row] = []
  /// What was said while a turn ran and has not reached the model yet. It rides in on the next
  /// tool result, or goes out with the next turn if this one ends first.
  private(set) var pendingSteers: [Row] = []

  var rows: [Row] { transcriptRows }

  var visibleRows: [Row] { transcriptRows }

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
    /// What this one step took, measured from when the row appeared to when the next one did.
    /// The turn's own total is spread across every row it produced; this is the part that
    /// belongs to this row alone, and it is what the badge beside a tool call shows.
    var elapsed: Double?
    /// Set on the last row of a turn, which is where the turn's total is worth saying.
    var turnSeconds: Double?
  }

  private(set) var meta: [String: RowMeta] = [:]

  func meta(for row: Row) -> RowMeta? { meta[row.id] }

  /// One file a turn changed, and by how much.
  struct FileChange: Identifiable, Equatable {
    var path: String
    var added: Int
    var removed: Int

    var id: String { path }
    var name: String { (path as NSString).lastPathComponent }
  }

  /// What a turn came to, which is the thing worth reading once a long stretch of tool traffic
  /// has scrolled past: which files it touched and what it did to them.
  struct TurnSummary: Equatable {
    var files: [FileChange]
    var added: Int
    var removed: Int
  }

  /// Keyed by the id of a turn's last row, which is where the card is drawn. Built once when a
  /// turn ends and once when a conversation is opened, never while one is streaming: diffing
  /// every edit in a transcript is not work for a poll twenty times a second.
  private(set) var turnSummaries: [String: TurnSummary] = [:]

  func summary(after row: Row) -> TurnSummary? { turnSummaries[row.id] }

  /// A turn runs from one prompt to the next, and its summary hangs off its final row.
  private func rebuildSummaries() {
    let rows = transcriptRows
    var built: [String: TurnSummary] = [:]
    var changes: [String: FileChange] = [:]
    var order: [String] = []
    var started = false

    func close(at id: String?) {
      defer {
        changes = [:]
        order = []
      }
      guard let id, !order.isEmpty else { return }
      let files = order.compactMap { changes[$0] }
      built[id] = TurnSummary(
        files: files,
        added: files.reduce(0) { $0 + $1.added },
        removed: files.reduce(0) { $0 + $1.removed })
    }

    var lastID: String?
    for row in rows {
      if case .prompt = row.kind {
        close(at: lastID)
        started = true
      }
      if started, case .toolCall(let name) = row.kind,
        let change = Self.change(from: name, arguments: row.text)
      {
        if var existing = changes[change.path] {
          existing.added += change.added
          existing.removed += change.removed
          changes[change.path] = existing
        } else {
          changes[change.path] = change
          order.append(change.path)
        }
      }
      lastID = row.id
    }
    close(at: lastID)

    if built != turnSummaries { turnSummaries = built }
  }

  /// What one call did to one file. An edit is a diff; a write is every line of what it wrote,
  /// since the file it replaced is not in the call to compare against.
  private static func change(from tool: String, arguments: String) -> FileChange? {
    guard tool == "edit" || tool == "write" else { return nil }
    guard
      let object = (try? JSONSerialization.jsonObject(with: Data(arguments.utf8)))
        as? [String: Any],
      let path = (object["path"] ?? object["file"]) as? String, !path.isEmpty
    else { return nil }

    if tool == "write" {
      let body = (object["contents"] ?? object["content"] ?? object["text"]) as? String ?? ""
      return FileChange(
        path: path, added: body.isEmpty ? 0 : body.components(separatedBy: "\n").count,
        removed: 0)
    }
    guard let old = object["old"] as? String, let new = object["new"] as? String else {
      return FileChange(path: path, added: 0, removed: 0)
    }
    let oldLines = old.isEmpty ? [] : old.components(separatedBy: "\n")
    let newLines = new.isEmpty ? [] : new.components(separatedBy: "\n")
    let diff = newLines.difference(from: oldLines)
    return FileChange(path: path, added: diff.insertions.count, removed: diff.removals.count)
  }

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
  /// Steering sent from the companion for a conversation with no turn in flight and no parked
  /// copy: held here until one of its turns goes out.
  @ObservationIgnored private var queuedSteers: [UUID: [Row]] = [:]
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

  /// What the conversation on screen has cost, which moves with it: every chat keeps its own
  /// tally, so switching to another one shows that one's turns rather than the window's.
  var meter: TurnMeter { current.meter }
  /// What the last turn put in front of the model, so the dial can show the prefill against it.
  private(set) var lastPrompt = 0
  private(set) var lastCached = 0

  var draft = ""
  /// What is waiting to go out with the next turn: pictures for the tower, files for the agent.
  private(set) var attachments: [Attachment] = []
  /// Named rather than shown, for a drop the composer could make nothing of.
  private(set) var refusedDrop: String?

  /// The chosen folder as git sees it, watched so the strip above the composer is never stale.
  let git = GitProbe()
  /// The VMs and pods this window has going, one per conversation that asked for one.
  let sandboxes = Sandboxes()

  var workspace: URL? {
    didSet {
      // A session is bound to the folder it was made for, so changing the folder retires them
      // — except the one answering, which keeps the folder it started in.
      guard workspace != oldValue else { return }
      agents = agents.filter { $0.key == run?.chatID }
      git.watch(workspace)
      remember(workspace)
    }
  }

  /// The folders worked in lately, so starting a chat somewhere is a menu rather than a panel.
  private(set) var recentWorkspaces: [URL] = []

  private func remember(_ url: URL?) {
    guard let url else { return }
    recentWorkspaces.removeAll { $0 == url }
    recentWorkspaces.insert(url, at: 0)
    if recentWorkspaces.count > 8 { recentWorkspaces.removeLast(recentWorkspaces.count - 8) }
    defaults.set(recentWorkspaces.map(\.path), forKey: "chat.recentWorkspaces")
  }

  /// Conversations gathered under the folder each was had in, newest folder first. The sidebar
  /// draws these instead of one flat list, which is what lets the window drop its folder bar.
  struct FolderGroup: Identifiable {
    var path: String?
    var chats: [SavedChat]

    var id: String { path ?? "\u{0}none" }
    var url: URL? { path.map { URL(filePath: $0) } }
    var name: String { url?.lastPathComponent ?? "No folder" }
  }

  var folders: [FolderGroup] {
    var order: [String] = []
    var grouped: [String: [SavedChat]] = [:]
    for chat in chats {
      let key = chat.workspace ?? "\u{0}none"
      if grouped[key] == nil { order.append(key) }
      grouped[key, default: []].append(chat)
    }
    return order.map { key in
      FolderGroup(path: key == "\u{0}none" ? nil : key, chats: grouped[key] ?? [])
    }
  }

  /// How hard the conversation on screen is asked to think. Fixed once it has had a turn; what
  /// is chosen then becomes the default for the next new one.
  var effort: ReasoningEffort {
    get { current.effort }
    set {
      defaultEffort = newValue
      defaults.set(newValue.rawValue, forKey: "chat.effort")
      guard current.isEmpty, newValue != current.effort else { return }
      update(current.id) { $0.effort = newValue }
      agents[current.id] = nil
    }
  }

  var isEffortLocked: Bool { !current.isEmpty }
  /// What a new conversation starts with: whatever was chosen last.
  private var defaultEffort: ReasoningEffort

  /// Which model answers a conversation: a pack on this Mac, or one of Apple's own.
  enum ModelPick: Hashable {
    case pack(String)
    case apple(AppleFoundationModel)
  }

  /// A question a running turn is waiting on the person to answer.
  struct Asked: Equatable {
    var chatID: UUID
    var question: TurnInbox.Question
  }

  private(set) var asked: Asked?

  /// What a turn's prompt came to against what the cache already held of it.
  struct Reuse: Equatable {
    var prompt: Int
    var cached: Int

    var fraction: Double { prompt > 0 ? Double(cached) / Double(prompt) : 0 }
  }

  private(set) var reuse: [UUID: Reuse] = [:]
  /// Bytes each conversation's archived prefixes take on disk.
  private(set) var diskUsage: [UUID: Int] = [:]
  /// The instructions and tool schemas every conversation starts from, archived once.
  private(set) var sharedDiskBytes = 0
  /// The conversation being read into the cache ahead of its next turn.
  private(set) var warmingChat: UUID?
  @ObservationIgnored private var warming: Task<Void, Never>?

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
    self.defaultEffort = effort

    let loaded = ChatStore().load()
    self.chats = loaded
    self.current = loaded.first ?? SavedChat(effort: effort)
    self.recentWorkspaces =
      (defaults.stringArray(forKey: "chat.recentWorkspaces") ?? []).map { URL(filePath: $0) }
    if let path = current.workspace ?? defaults.string(forKey: "chat.workspace") {
      self.workspace = URL(filePath: path)
      self.git.watch(self.workspace)
    }
    if chats.isEmpty { chats = [current] }
    var builder = RowBuilder()
    self.transcriptRows = builder.rows(from: current.transcript, notices: current.notes)
    rebuildSummaries()

    NotificationCenter.default.addObserver(
      forName: NSApplication.willTerminateNotification, object: nil, queue: nil
    ) { [weak self] _ in
      self?.persistAll()
      ShellJobs.stopEverything()
      self?.sandboxes.shutdownAll()
    }
  }

  // MARK: - Chats

  func startNewChat(in folder: URL? = nil) {
    persist()
    if let folder {
      workspace = folder
      defaults.set(folder.path, forKey: "chat.workspace")
    }
    var chat = SavedChat(workspace: (folder ?? workspace)?.path, effort: defaultEffort)
    stamp(&chat)
    chats.insert(chat, at: 0)
    open(chat)
  }

  /// A folder chosen from the panel, with a fresh conversation in it. The plus button in the
  /// sidebar is the one place a folder is picked now, so picking one starts a chat there.
  func startNewChatInChosenFolder() {
    guard let url = askForWorkspace() else { return }
    startNewChat(in: url)
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
    if let agent = agents[chat.id] {
      Task.detached { await agent.workspace.stopAllJobs() }
    }
    sandboxes.shutdown(chat.id)
    agents[chat.id] = nil
    parked[chat.id] = nil
    store.delete(chat.id)
    chats.removeAll { $0.id == chat.id }
    if let prefixes = server?.prefixStore {
      store.pruneCache(for: chat, keeping: chats, in: prefixes)
    }
    diskUsage[chat.id] = nil
    guard chat.id == current.id else { return }
    open(chats.first ?? SavedChat(workspace: workspace?.path, effort: defaultEffort))
  }

  /// Opening a conversation rebuilds its session from the transcript it was saved with, so the
  /// model picks up the thread rather than being told about it.
  private func open(_ chat: SavedChat) {
    park()
    current = chat
    if let path = chat.workspace { workspace = URL(filePath: path) }

    git.watch(workspace)
    attachments.removeAll()
    refusedDrop = nil

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
      transcriptRows = builder.rows(
        from: agents[chat.id]?.transcript ?? chat.transcript, notices: chat.notes)
      meta.removeAll()
      display.removeAll()
      pendingSteers = queuedSteers.removeValue(forKey: chat.id) ?? []
      failure = nil
    }
    if !chats.contains(where: { $0.id == chat.id }) { chats.insert(chat, at: 0) }
    turnSummaries = [:]
    rebuildSummaries()
    warm()
  }

  /// Set aside what the conversation being left looks like, but only while it is being
  /// answered: anything else is cheap enough to fold again from its transcript.
  private func park() {
    guard run?.chatID == current.id else { return }
    parked[current.id] = Parked(
      rows: transcriptRows, meta: meta, display: display, steers: pendingSteers,
      failure: failure)
  }

  // MARK: - Model lock

  /// What would answer a new conversation right now.
  var activePick: ModelPick? {
    guard let server else { return nil }
    if let apple = server.settings.appleModel { return .apple(apple) }
    let id = server.settings.activeModelID
    return id.isEmpty ? nil : .pack(id)
  }

  func pick(of chat: SavedChat) -> ModelPick? {
    if let apple = chat.appleModel { return .apple(apple) }
    return chat.model.map(ModelPick.pack)
  }

  func isAvailable(_ pick: ModelPick) -> Bool {
    guard let server else { return false }
    switch pick {
    case .pack(let id): return server.catalog[id] != nil
    case .apple(let apple): return server.offeredAppleModels.contains(apple)
    }
  }

  func name(of pick: ModelPick) -> String {
    switch pick {
    case .pack(let id): server?.catalog[id]?.displayName ?? id
    case .apple(let apple): apple.displayName
    }
  }

  /// The model the conversation on screen was started with, when it is not the one chosen now.
  var lockedElsewhere: ModelPick? {
    guard !current.isEmpty, let locked = pick(of: current), locked != activePick else {
      return nil
    }
    return locked
  }

  var lockedName: String { lockedElsewhere.map(name(of:)) ?? "" }

  private func use(_ pick: ModelPick) {
    switch pick {
    case .pack(let id): server?.activate(id)
    case .apple(let apple): server?.settings.appleModel = apple
    }
  }

  /// Fixes a conversation to whatever would answer it now, along with its effort.
  private func stamp(_ chat: inout SavedChat) {
    switch activePick {
    case .apple(let apple):
      chat.appleModelID = apple.rawValue
      chat.model = nil
    case .pack(let id):
      chat.appleModelID = nil
      chat.model = id
    case nil:
      chat.appleModelID = nil
      chat.model = nil
    }
  }

  /// Brings back the model the conversation on screen answers with, then sends what is typed.
  private func switchBack() {
    guard let locked = lockedElsewhere, isAvailable(locked) else { return }
    use(locked)
    agents[current.id] = nil
    if case .pack = locked, engine == nil {
      load()
    } else {
      Task { await self.sendOnceLoaded() }
    }
  }

  /// A copy of the conversation on screen that answers with the model chosen now. The original
  /// stays as it was, fixed to the model it started with.
  func branch() {
    guard !isRunningTurn else { return }
    persist()
    var chat = SavedChat(workspace: current.workspace, effort: current.effort)
    chat.transcript = current.transcript
    chat.notes = current.notes
    chat.parent = current.id
    chat.title = current.title
    stamp(&chat)
    chats.insert(chat, at: 0)
    store.save(chat)
    open(chat)
  }

  // MARK: - Readahead

  /// Reads the conversation on screen into the cache ahead of its next turn, when the pack that
  /// would answer it is the one loaded and nothing else is running.
  private func warm() {
    warming?.cancel()
    warming = nil
    warmingChat = nil
    refreshDiskUsage()
    guard !isRunningTurn, !current.isEmpty, engine != nil, server?.switching == nil,
      case .pack = pick(of: current), lockedElsewhere == nil,
      let agent = resolveAgent(for: current.id)
    else { return }
    let chatID = current.id
    warmingChat = chatID
    warming = Task { [weak self] in
      let read = await agent.readahead()
      guard let self, !Task.isCancelled, self.warmingChat == chatID else { return }
      self.warmingChat = nil
      if let read, read.tokens > 0 {
        self.reuse[chatID] = Reuse(prompt: read.tokens, cached: read.reused)
      }
      self.refreshDiskUsage()
    }
  }

  private func refreshDiskUsage() {
    guard let store = server?.prefixStore else { return }
    let usage = store.usage()
    var byChat: [UUID: Int] = [:]
    for (tag, held) in usage {
      if let id = UUID(uuidString: tag) { byChat[id] = held.bytes }
    }
    let shared = store.totalBytes - byChat.values.reduce(0, +)
    if byChat != diskUsage { diskUsage = byChat }
    if shared != sharedDiskBytes { sharedDiskBytes = max(0, shared) }
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
    if id == current.id { saved.workspace = workspace?.path }
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

  /// Where the context has gone, in the four parts worth telling apart. The instructions and
  /// the tool schemas are the same every turn and are the part that can actually be cut; the
  /// conversation is what the turns themselves have cost.
  struct ContextUse: Equatable {
    var instructions = 0
    var toolSchemas = 0
    var conversation = 0
    var used = 0
    var ceiling = 0

    var free: Int { max(0, ceiling - used) }
    var fraction: Double { ceiling > 0 ? min(1, Double(used) / Double(ceiling)) : 0 }
  }

  var contextUse: ContextUse {
    let used = readout?.context.peakTokens ?? 0
    let instructions = engine?.instructionTokens ?? 0
    let schemas = engine?.toolSchemaTokens ?? 0
    return ContextUse(
      instructions: min(instructions, used),
      toolSchemas: min(schemas, max(0, used - instructions)),
      conversation: max(0, used - instructions - schemas),
      used: used,
      ceiling: readout?.context.ceilingTokens ?? 0)
  }

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
    /// The turn is waiting on an answer from the person.
    case answer
    /// The conversation answers with a model other than the one loaded; sending loads it again.
    case switchBack
    /// It answers with a model that is no longer here, so only a branch can carry it on.
    case missingModel
    /// Another conversation is being answered. The pack takes one turn at a time, so this one
    /// waits rather than queueing into a busy engine.
    case busy
    case nothingToSay
  }

  var submission: Submission {
    if workspace == nil { return .chooseFolder }
    if isResponding { return question != nil ? .answer : .steer }
    if isRunningTurn { return .busy }
    if server?.switching != nil { return .loading }
    if let locked = lockedElsewhere { return isAvailable(locked) ? .switchBack : .missingModel }
    // One of Apple's own models needs nothing loaded: it answers whether or not a pack is
    // resident, so none of the pack's own gating applies while it is the one chosen.
    if server?.settings.appleModel == nil {
      if server?.phase.isBusy == true { return .loading }
      if engine == nil { return .load }
    }
    if !typed.isEmpty || !pendingSteers.isEmpty || !attachments.isEmpty { return .send }
    return .nothingToSay
  }

  var submissionLabel: String {
    switch submission {
    case .chooseFolder: "Choose"
    case .load: "Load"
    case .loading: "Loading"
    case .steer: "Steer"
    case .answer: "Answer"
    case .switchBack: "Load \(lockedName)"
    case .missingModel: "Branch"
    case .busy: "Busy"
    case .send, .nothingToSay: "Send"
    }
  }

  var canSubmit: Bool {
    switch submission {
    case .loading, .busy, .nothingToSay: false
    case .steer, .answer: !typed.isEmpty
    case .chooseFolder, .load, .send, .switchBack, .missingModel: true
    }
  }

  func submit() {
    switch submission {
    case .chooseFolder: chooseWorkspace()
    case .load: load()
    case .send: send()
    case .steer: steer()
    case .answer: answer()
    case .switchBack: switchBack()
    case .missingModel: branch()
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
    case .switchBack: "This conversation answers with \(lockedName). Send to load it again."
    case .missingModel: "\(lockedName) is no longer on this Mac. Branch to go on with another."
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
      if server.switching != nil {
        try? await Task.sleep(for: .milliseconds(150))
        continue
      }
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
    guard let url = askForWorkspace() else { return }
    workspace = url
    defaults.set(url.path, forKey: "chat.workspace")
    update(current.id) { $0.workspace = url.path }
    if let chat = saved(current.id) { store.save(chat) }
  }

  private func askForWorkspace() -> URL? {
    let panel = NSOpenPanel()
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    panel.prompt = "Work Here"
    panel.directoryURL = workspace
    guard panel.runModal() == .OK else { return nil }
    return panel.url
  }

  func send() {
    // Whatever was queued while the last turn ran goes out ahead of what was just typed, and
    // what was dropped on the composer goes out ahead of both: a picture is context for the
    // question, not an afterthought to it.
    let bundle = AttachmentBundle.build(attachments, workspace: workspace)
    let text = ([bundle.preamble] + pendingSteers.map(\.text) + [typed])
      .filter { !$0.isEmpty }
      .joined(separator: "\n\n")
    guard !text.isEmpty, !isRunningTurn else { return }
    draft = ""
    failure = nil
    refusedDrop = nil
    pendingSteers.removeAll()
    attachments.removeAll()
    send(text, in: current.id, images: bundle.images)
  }

  /// A turn started against a named conversation, which may not be the one on screen: the
  /// companion sends this way, and its rows fold into a parked copy until someone opens it,
  /// exactly as a conversation left mid-answer does.
  @discardableResult
  func send(_ text: String, in chatID: UUID, images: [URL] = []) -> Bool {
    let queued = queuedSteers.removeValue(forKey: chatID)?.map(\.text) ?? []
    let text = (queued + [text])
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }
      .joined(separator: "\n\n")
    guard !text.isEmpty, !isRunningTurn, let chat = saved(chatID) else { return false }
    if !chat.isEmpty, let locked = pick(of: chat), locked != activePick {
      guard isAvailable(locked) else { return false }
      use(locked)
    }
    guard let agent = resolveAgent(for: chatID) else { return false }
    if chat.isEmpty { update(chatID) { stamp(&$0) } }
    if chatID != current.id, parked[chatID] == nil {
      var builder = RowBuilder()
      let transcript = agent.transcript.isEmpty ? chat.transcript : agent.transcript
      parked[chatID] = Parked(
        rows: builder.rows(from: transcript, notices: chat.notes), meta: [:], display: [:],
        steers: [], failure: nil)
    }

    warming?.cancel()
    warming = nil
    warmingChat = nil
    let run = Run(chatID: chatID, agent: agent)
    run.totalsAtStart = server?.readout?.totals
    run.tokensAtStart = run.totalsAtStart?.generatedTokens ?? 0
    self.run = run
    runToken += 1
    startPolling(run)

    // The pictures travel on the front of the prompt, which is the one part of a turn a
    // session stores exactly as it was given and hands back unchanged.
    let payload = PromptAttachments.marker(for: images) + text

    run.task = Task { [weak self] in
      do {
        _ = try await agent.send(payload)
      } catch is CancellationError {
        // Stopping a turn is an ordinary thing to do, but it still leaves a mark where it
        // happened rather than a gap someone has to remember the reason for.
        self?.note("Stopped", tone: .stopped, for: run.chatID)
      } catch {
        self?.note(ChatController.plainly(error), tone: .failed, for: run.chatID)
        if AppleIntelligenceProbe.isMissingAssets(error) {
          self?.server?.setAppleIntelligenceUsable(false)
        }
      }
      guard let self else { return }
      self.finish(run)
    }
    return true
  }

  func stop() {
    run?.agent.inbox.cancelQuestion()
    run?.task?.cancel()
  }

  /// What became of a turn, written into the conversation that earned it — not into a bar
  /// under whichever one is being read when it lands. It sits after the last row there is, so
  /// a stop lands under the half-written answer it cut off.
  private func note(_ text: String, tone: ChatNotice.Tone, for chatID: UUID) {
    let notice = ChatNotice(tone: tone, text: text, after: rows(of: chatID).last?.id)
    update(chatID) { $0.notes.append(notice) }
    if let chat = saved(chatID) { store.save(chat) }
    refreshRows(of: chatID)
  }

  /// A framework error said in one line. A tool that throws arrives with its whole declaration
  /// printed into the message, which is a paragraph of Swift where a sentence would do.
  nonisolated static func plainly(_ error: Error) -> String {
    let whole = error.localizedDescription
    let message =
      whole.range(of: "Underlying error: ").map { String(whole[$0.upperBound...]) } ?? whole
    return
      message
      .replacing(/\s+/, with: " ")
      .trimmingCharacters(in: .whitespacesAndNewlines)
  }

  /// The rows of a conversation folded again, notices and all, wherever they are held.
  private func refreshRows(of chatID: UUID) {
    guard let chat = saved(chatID) else { return }
    var builder = RowBuilder()
    let rows = builder.rows(
      from: agents[chatID]?.transcript ?? chat.transcript, notices: chat.notes)
    if chatID == current.id {
      transcriptRows = Self.keeping(transcriptRows, with: rows)
    } else if var state = parked[chatID] {
      state.rows = Self.keeping(state.rows, with: rows)
      parked[chatID] = state
    }
  }

  /// Said to the turn on screen while it runs. It reaches the model with the next tool result.
  func steer() {
    let text = typed
    guard !text.isEmpty else { return }
    draft = ""
    let steer = TurnInbox.Steer(text: text)
    if isResponding { run?.agent.steer(steer) }
    pendingSteers.append(Row(id: steer.id, kind: .steer, text: text))
  }

  func drop(_ row: Row) {
    pendingSteers.removeAll { $0.id == row.id }
    if isResponding { run?.agent.inbox.remove(row.id) }
  }

  /// What the turn on screen is waiting on the person to answer.
  var question: TurnInbox.Question? {
    guard let asked, asked.chatID == current.id else { return nil }
    return asked.question
  }

  func isAsking(_ chat: SavedChat) -> Bool { asked?.chatID == chat.id }

  /// Answers the question the turn on screen is waiting on, with one of the options it offered
  /// or with whatever was typed.
  func answer(_ choice: String? = nil) {
    let reply = (choice ?? typed).trimmingCharacters(in: .whitespacesAndNewlines)
    guard !reply.isEmpty, let run, run.chatID == current.id else { return }
    if choice == nil { draft = "" }
    run.agent.inbox.answer(reply)
    asked = nil
  }

  // MARK: - Attachments

  func attach(_ intake: AttachmentIntake) {
    for attachment in intake.accepted
    where !attachments.contains(where: {
      $0.url == attachment.url
    }) {
      attachments.append(attachment)
    }
    refusedDrop = intake.refused.isEmpty ? nil : intake.refused.joined(separator: ", ")
  }

  func attach(_ urls: [URL]) {
    attach(AttachmentIntake.read(urls))
  }

  func pasteAttachment() {
    attach(AttachmentIntake.readPasteboard())
  }

  func detach(_ attachment: Attachment) {
    attachments.removeAll { $0.id == attachment.id }
    if attachment.isTemporary { try? FileManager.default.removeItem(at: attachment.url) }
  }

  func clearRefusedDrop() {
    refusedDrop = nil
  }

  /// Whether anything attached needs eyes the resident pack does not have, which is worth
  /// saying in the composer rather than discovering when the turn comes back short.
  var needsVision: Bool {
    attachments.contains { $0.kind == .image }
  }

  // MARK: - Worktrees

  /// A checkout of its own for this conversation, so a turn that builds and commits is not
  /// doing it in the same files someone else is editing. Only ever on request.
  func makeWorktree(named name: String? = nil) {
    guard let workspace else { return }
    do {
      let made = try Git.addWorktree(of: workspace, named: name ?? current.title)
      use(made)
    } catch {
      failure = error.localizedDescription
    }
  }

  func use(_ worktree: GitWorktree) {
    workspace = worktree.path
    defaults.set(worktree.path.path, forKey: "chat.workspace")
    update(current.id) { $0.workspace = worktree.path.path }
    if let chat = saved(current.id) { store.save(chat) }
    git.refresh()
  }

  func removeWorktree(_ worktree: GitWorktree) {
    guard let workspace else { return }
    do {
      try Git.removeWorktree(worktree, of: workspace)
      if workspace.standardizedFileURL == worktree.path.standardizedFileURL {
        self.workspace = git.status?.root
      }
      git.refresh()
    } catch {
      failure = error.localizedDescription
    }
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

  private func resolveAgent() -> CodingAgent? { resolveAgent(for: current.id) }

  /// The session for a conversation, made against that conversation's own folder rather than
  /// whichever one the window happens to be pointed at.
  private func resolveAgent(for chatID: UUID) -> CodingAgent? {
    if let agent = agents[chatID] { return agent }
    guard let chat = saved(chatID) else { return nil }
    let folder =
      (chatID == current.id ? workspace : nil)
      ?? chat.workspace.map { URL(filePath: $0) }

    let choiceModel: CodingAgent.ModelChoice
    switch chat.isEmpty ? activePick : pick(of: chat) ?? activePick {
    case .apple(let apple):
      // Reasoning level and guardrails live beside every pack's own sampler knobs, keyed the
      // same way — one settings sheet, whichever kind of model it is a sheet for.
      let sampler = server?.samplerSettings.settings(for: apple.id) ?? .default
      choiceModel = .apple(
        apple, reasoningLevel: sampler.resolvedAppleReasoningLevel,
        guardrails: sampler.resolvedAppleGuardrails)
    case .pack(let id):
      guard let engine else { return nil }
      choiceModel = .resident(engine, effort: chat.effort, model: id)
    case nil:
      guard let engine else { return nil }
      choiceModel = .resident(engine, effort: chat.effort, model: nil)
    }

    // A sandbox needs a folder to share in. Without one there is nothing to sandbox, so the
    // choice quietly becomes this Mac rather than failing on the first command.
    var choice = chat.sandbox ?? sandboxes.settings.defaultChoice
    if folder == nil { choice.kind = .native }
    let branch = folder == workspace && git.status?.detached == false ? git.status?.branch : nil

    let made = CodingAgent(
      model: choiceModel,
      workspace: Workspace(
        host: sandboxes.host(
          for: chatID, choice: choice,
          workspace: folder ?? FileManager.default.temporaryDirectory)),
      transcript: chat.transcript.isEmpty ? nil : chat.transcript,
      tag: chatID.uuidString,
      environment: PromptEnvironment.block(
        folder: choice.kind == .native ? folder : SandboxChoice.guestWorkspace, branch: branch))
    agents[chatID] = made
    return made
  }

  /// Where this conversation's commands run. Changing it rebuilds the session against the new
  /// shell; the transcript is what carries the conversation, so nothing is lost by that.
  var sandboxChoice: SandboxChoice {
    get { current.sandbox ?? sandboxes.settings.defaultChoice }
    set {
      guard newValue != sandboxChoice else { return }
      update(current.id) { $0.sandbox = newValue }
      sandboxes.settings.defaultChoice = newValue
      if newValue.kind != sandboxes.choice(of: current.id).kind
        || newValue.isSandboxed == false
      {
        sandboxes.shutdown(current.id)
      }
      agents[current.id] = nil
      store.save(current)
    }
  }

  var sandboxPhase: SandboxPhase { sandboxes.phase(of: current.id) }

  /// Gives the VM or the pod back. The conversation stays; its next command boots another one.
  func stopSandbox() {
    sandboxes.shutdown(current.id)
    agents[current.id] = nil
  }

  /// What the resident pack is holding, so the memory a VM may take is what is actually spare.
  var residentBytes: Int { server?.readout?.load.held ?? 0 }

  /// The transcript is read rather than mirrored: the executor fills it as the tokens land, so
  /// polling it is enough to show thinking, tool calls and the answer as they arrive.
  private func startPolling(_ run: Run) {
    run.poller?.cancel()
    // Start from the beginning: streamResponse may add the prompt after this poller starts.
    // The first non-empty transcript must reach disk before a stop or process failure can win.
    run.lastCheckpoint = .distantPast
    run.poller = Task { [weak self] in
      while !Task.isCancelled {
        self?.absorbTranscript(of: run)
        self?.checkpoint(run)
        try? await Task.sleep(for: .milliseconds(50))
      }
    }
  }

  /// A turn can run for minutes; writing only once it finishes means quitting or crashing
  /// mid-turn loses all of it. The short interval also catches the prompt immediately after
  /// streamResponse inserts it, before a stop or inference failure can skip finish().
  private func checkpoint(_ run: Run) {
    guard Date().timeIntervalSince(run.lastCheckpoint) >= 0.25 else { return }
    run.lastCheckpoint = Date()
    let transcript = run.agent.transcript
    guard !transcript.isEmpty, var saved = saved(run.chatID) else { return }
    // A turn that comes apart can hand back a transcript shorter than the one already on disk.
    // Whatever else that costs, it must not cost the conversation: the checkpoint waits for a
    // transcript that is at least as long as the one it would be writing over.
    guard transcript.count >= saved.transcript.count else { return }
    saved.transcript = transcript
    saved.updated = Date()
    if run.chatID == current.id { saved.workspace = workspace?.path }
    if !saved.titleIsCustom, let derived = SavedChat.title(from: transcript) {
      saved.title = derived
    }
    // Encoding a long transcript is not something a turn should stop for: the window is trying
    // to draw tokens while this runs.
    let directory = store.folder
    let sequence = ChatStore.reserveCheckpoint(for: saved.id)
    Task.detached(priority: .utility) {
      ChatStore.writeCheckpoint(saved, in: directory, sequence: sequence)
    }
  }

  private func finish(_ run: Run) {
    run.poller?.cancel()
    run.poller = nil
    absorbTranscript(of: run)
    _ = run.agent.inbox.take()
    if asked?.chatID == run.chatID { asked = nil }
    if self.run === run {
      self.run = nil
      runToken += 1
    }
    closeClocks(run)
    if run.chatID == current.id { rebuildSummaries() }
    recordTurnCost(run)
    let seconds = -run.started.timeIntervalSinceNow
    let generated = server?.readout.map { max(0, $0.totals.generatedTokens - run.tokensAtStart) }
    update(run.chatID) { chat in
      chat.meter.turns += 1
      chat.meter.seconds += seconds
      chat.meter.tokens += generated ?? 0
    }
    persist(run.chatID, prompt: true)

    if let readout = server?.readout {
      lastPrompt = readout.context.peakTokens
      lastCached = readout.prefix?.hits ?? 0
    }
    if run.chatID == current.id { warm() }
  }

  private func absorbTranscript(of run: Run) {
    absorb(
      run.builder.rows(from: run.agent.transcript, notices: saved(run.chatID)?.notes ?? []),
      for: run)
    absorbInbox(of: run)
  }

  /// What the person has said that the model has not heard yet, the question it is waiting on,
  /// and how much of its last prompt the cache already held.
  private func absorbInbox(of run: Run) {
    let waiting = run.agent.inbox.waiting.map { Row(id: $0.id, kind: .steer, text: $0.text) }
    if run.chatID == current.id {
      if waiting != pendingSteers { pendingSteers = waiting }
    } else if var state = parked[run.chatID], state.steers != waiting {
      state.steers = waiting
      parked[run.chatID] = state
    }

    let now = run.agent.inbox.pending.map { Asked(chatID: run.chatID, question: $0) }
    if now != asked {
      if now != nil, !NSApp.isActive { NSApp.requestUserAttention(.informationalRequest) }
      asked = now
    }

    if let request = readout?.inFlight.first(where: { $0.api == "chat" }),
      request.promptTokens > 0
    {
      let seen = Reuse(prompt: request.promptTokens, cached: request.cachedTokens)
      if reuse[run.chatID] != seen { reuse[run.chatID] = seen }
    }
  }

  /// A row's clock starts the first time it is seen, which is as close to when it happened as
  /// a transcript without timestamps allows.
  ///
  /// Where the rows land depends on whether the turn is the one being watched: the conversation
  /// on screen takes them through the observed properties, and one left running takes them into
  /// its parked copy, which draws nothing until it is opened again.
  private func absorb(_ folded: [Row], for run: Run) {
    let shown = run.chatID == current.id ? transcriptRows : (parked[run.chatID]?.rows ?? [])
    let rows = Self.keeping(shown, with: folded)
    if run.chatID == current.id {
      // Assigning an identical array would still be a change to everything watching it, and at
      // twenty polls a second that is a re-render of the whole transcript for nothing.
      guard rows != transcriptRows else { return }
      transcriptRows = rows
      stamp(rows, into: &meta, for: run)
    } else {
      guard var state = parked[run.chatID], rows != state.rows else { return }
      state.rows = rows
      stamp(rows, into: &state.meta, for: run)
      parked[run.chatID] = state
    }
  }

  /// The rows as they now stand, with nothing lost that was on screen a moment ago.
  ///
  /// The session's transcript is asked to survive a failed turn, but it is the framework's to
  /// keep and a conversation is too expensive to lose on that promise alone. When a fold comes
  /// back missing rows that were already drawn, the older ones stay and whatever is new is
  /// added to them; a fold that only grows replaces them outright, which is every ordinary
  /// poll.
  nonisolated static func keeping(_ shown: [Row], with folded: [Row]) -> [Row] {
    guard !shown.isEmpty else { return folded }
    let arrived = Set(folded.map(\.id))
    guard !shown.allSatisfy({ arrived.contains($0.id) }) else { return folded }

    let latest = Dictionary(folded.map { ($0.id, $0) }, uniquingKeysWith: { _, second in second })
    var out = shown.map { latest[$0.id] ?? $0 }
    let held = Set(shown.map(\.id))
    out.append(contentsOf: folded.filter { !held.contains($0.id) })
    return out
  }

  /// A row's clock starts when it appears and stops when the next one does. A transcript
  /// carries no timestamps, so this is the only account of what each step cost — and it is the
  /// one a badge beside a tool call is actually reporting.
  private func stamp(_ rows: [Row], into meta: inout [String: RowMeta], for run: Run) {
    let now = Date()
    for (index, row) in rows.enumerated() where meta[row.id] == nil {
      if index > 0, var earlier = meta[rows[index - 1].id], earlier.elapsed == nil {
        earlier.elapsed = max(0, now.timeIntervalSince(earlier.at))
        meta[rows[index - 1].id] = earlier
      }
      meta[row.id] = RowMeta(at: now, wasRead: row.kind.isInput)
      run.rowsThisTurn.insert(row.id)
    }
  }

  /// The last row of a turn has no row after it to stop its clock, so the turn's end does it —
  /// and leaves the turn's own total there, which is what the footer under an answer shows.
  private func closeClocks(_ run: Run) {
    let now = Date()
    let total = -run.started.timeIntervalSinceNow
    let last = (run.chatID == current.id ? transcriptRows : parked[run.chatID]?.rows ?? []).last
    withMeta(of: run.chatID) { meta in
      for id in run.rowsThisTurn {
        guard var record = meta[id] else { continue }
        if record.elapsed == nil { record.elapsed = max(0, now.timeIntervalSince(record.at)) }
        if id == last?.id { record.turnSeconds = total }
        meta[id] = record
      }
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
    } else if var state = parked[chatID] {
      change(&state.meta)
      parked[chatID] = state
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

  // MARK: - Reached from the companion

  /// A conversation by id, wherever it is held.
  func chat(_ id: UUID) -> SavedChat? { saved(id) }

  /// The rows of any conversation: the one on screen, one being answered in the background, or
  /// one that has only ever been on disk.
  func rows(of id: UUID) -> [Row] {
    if id == current.id { return transcriptRows }
    if let parked = parked[id] { return parked.rows }
    guard let chat = saved(id) else { return [] }
    var builder = RowBuilder()
    return builder.rows(from: agents[id]?.transcript ?? chat.transcript, notices: chat.notes)
  }

  func meta(of id: UUID) -> [String: RowMeta] {
    id == current.id ? meta : (parked[id]?.meta ?? [:])
  }

  func steers(of id: UUID) -> [Row] {
    id == current.id ? pendingSteers : (parked[id]?.steers ?? [])
  }

  func failure(of id: UUID) -> String? {
    id == current.id ? failure : parked[id]?.failure
  }

  /// A conversation made without opening it, so a phone starting one does not move the window
  /// off whatever is being read on the Mac.
  func makeChat(
    workspace folder: String?, model: String?, effort wanted: ReasoningEffort?
  )
    -> SavedChat
  {
    let chat = SavedChat(
      workspace: folder ?? workspace?.path,
      model: model ?? server?.settings.activeModelID,
      effort: wanted ?? defaultEffort)
    chats.insert(chat, at: 0)
    store.save(chat)
    return chat
  }

  func change(
    _ id: UUID, title: String?, workspace folder: String?, effort wanted: ReasoningEffort?
  ) {
    update(id) {
      if let title, !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        $0.title = title
        $0.titleIsCustom = true
      }
      if let folder { $0.workspace = folder }
      if let wanted, $0.isEmpty { $0.effort = wanted }
      $0.updated = Date()
    }
    if id == current.id, let folder { workspace = URL(filePath: folder) }
    if let chat = saved(id) { store.save(chat) }
  }

  /// Guidance for a conversation that may not be the one on screen. One being answered takes it
  /// into the running session; any other holds it until its next turn.
  func steer(_ text: String, in id: UUID) {
    let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { return }
    if let run, run.chatID == id {
      if asked?.chatID == id {
        run.agent.inbox.answer(text)
        asked = nil
        return
      }
      let steer = TurnInbox.Steer(text: text)
      run.agent.steer(steer)
      let row = Row(id: steer.id, kind: .steer, text: text)
      if id == current.id { pendingSteers.append(row) } else { parked[id]?.steers.append(row) }
      return
    }
    let row = Row(id: "steer-\(UUID().uuidString)", kind: .steer, text: text)
    if id == current.id {
      pendingSteers.append(row)
    } else if parked[id] != nil {
      parked[id]?.steers.append(row)
    } else {
      queuedSteers[id, default: []].append(row)
    }
  }

  func stop(_ id: UUID) {
    guard run?.chatID == id else { return }
    stop()
  }

  /// Read straight through, appending rather than replacing. The session is free to split a
  /// streamed reply across as many entries as it likes, so consecutive entries of the same
  /// kind are joined into one row: whatever was generated is shown, however it arrived.
  nonisolated private static func fold(
    _ entries: some Sequence<Transcript.Entry>, into existing: [Row]
  ) -> [Row] {
    var rows = existing

    func add(_ id: String, _ kind: Row.Kind, _ text: String, images: [String] = []) {
      guard !text.isEmpty || !images.isEmpty else { return }
      if let last = rows.last, last.kind.joins(kind) {
        rows[rows.count - 1].text += text
        return
      }
      rows.append(Row(id: id, kind: kind, text: text, images: images))
    }

    for entry in entries {
      switch entry {
      case .instructions(let instructions):
        add(instructions.id, .system, text(instructions.segments))
      case .prompt(let prompt):
        let split = PromptAttachments.split(text(prompt.segments))
        add(
          prompt.id, .prompt, PromptEnvironment.split(split.body).body,
          images: split.images)
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
        let split = SteerBlock.split(text(output.segments))
        add(Row.outputID(output.id), .toolOutput(name: output.toolName), split.output)
        for (index, said) in split.steers.enumerated() {
          rows.append(Row(id: Row.outputID(output.id) + "steer\(index)", kind: .steer, text: said))
        }
      @unknown default:
        continue
      }
    }
    return rows
  }

  /// Entries can still change after another entry is appended: the generation channel updates
  /// reasoning and responses by ID, including replacing their text when generation finishes.
  /// Always fold the current snapshot so an earlier row cannot retain a partial streamed value.
  /// `absorb` avoids publishing unchanged rows to the view.
  struct RowBuilder {
    func rows(from transcript: Transcript, notices: [ChatNotice] = []) -> [Row] {
      ChatController.weave(notices, into: ChatController.fold(transcript, into: []))
    }
  }

  /// A notice back into the rows at the point it happened, or at the end when the row it
  /// followed is no longer there.
  nonisolated static func weave(_ notices: [ChatNotice], into rows: [Row]) -> [Row] {
    guard !notices.isEmpty else { return rows }
    var out = rows
    for notice in notices.sorted(by: { $0.at < $1.at }) {
      let row = Row(
        id: notice.id.uuidString, kind: .notice(tone: notice.tone), text: notice.text)
      guard let anchor = notice.after,
        var index = out.lastIndex(where: { $0.id == anchor })
      else {
        out.append(row)
        continue
      }
      // Past anything already sitting under that row, so two notices keep their order.
      while index + 1 < out.count, case .notice = out[index + 1].kind { index += 1 }
      out.insert(row, at: index + 1)
    }
    return out
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
