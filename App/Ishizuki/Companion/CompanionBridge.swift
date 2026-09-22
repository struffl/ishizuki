// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The Mac's side of the companion protocol: the window's own controllers, spelled as the wire
// types a phone reads. Nothing here has an opinion about HTTP.

import Foundation
import IshizukiKit
import IshizukiLink

@available(macOS 27.0, *)
@MainActor
final class CompanionBridge {
  private let chat: ChatController
  private let server: ServerController
  private let settings: CompanionSettings

  /// The Mac's background commands, held here rather than on a shell that is built per
  /// request. A job outlives the call that started it, so its table has to as well.
  private let jobTable = ShellJobs()

  init(chat: ChatController, server: ServerController, settings: CompanionSettings) {
    self.chat = chat
    self.server = server
    self.settings = settings
  }

  var version: String {
    Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0"
  }

  func info(pairingOpen: Bool) -> ServerInfo {
    ServerInfo(
      name: settings.serviceName,
      version: version,
      model: server.settings.activeModelID.isEmpty ? nil : server.settings.activeModelID,
      modelLoaded: server.phase.isRunning,
      chats: chat.chats.count,
      pairingOpen: pairingOpen)
  }

  // MARK: - Conversations

  func summaries() -> [ChatSummary] {
    chat.chats.map { summary(of: $0) }
  }

  func summary(of saved: SavedChat) -> ChatSummary {
    let rows = chat.rows(of: saved.id)
    let preview =
      rows.last { $0.kind == .answer || $0.kind == .prompt }?
      .text
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .replacing(/\s+/, with: " ")
    return ChatSummary(
      id: saved.id,
      title: saved.title,
      created: saved.created,
      updated: saved.updated,
      workspace: saved.workspace,
      model: saved.model,
      effort: saved.effort.rawValue,
      rows: rows.count,
      isRunning: chat.isRunning(saved),
      preview: preview.map { String($0.prefix(140)) })
  }

  func detail(_ id: UUID) throws -> ChatDetail {
    guard let saved = chat.chat(id) else { throw missing(id) }
    let meta = chat.meta(of: id)
    return ChatDetail(
      summary: summary(of: saved),
      rows: chat.rows(of: id).map { row(from: $0, meta: meta[$0.id]) },
      pendingSteers: chat.steers(of: id).map { row(from: $0, meta: nil) },
      failure: chat.failure(of: id))
  }

  func create(_ new: NewChat) throws -> ChatSummary {
    if let folder = new.workspace { _ = try resolve(folder) }
    let made = chat.makeChat(
      workspace: new.workspace, model: new.model,
      effort: new.effort.flatMap(ReasoningEffort.init(rawValue:)))
    return summary(of: made)
  }

  func delete(_ id: UUID) throws {
    guard let saved = chat.chat(id) else { throw missing(id) }
    chat.delete(saved)
  }

  func change(_ id: UUID, _ change: ChatChange) throws -> ChatSummary {
    guard chat.chat(id) != nil else { throw missing(id) }
    if let folder = change.workspace { _ = try resolve(folder) }
    chat.change(
      id, title: change.title, workspace: change.workspace,
      effort: change.effort.flatMap(ReasoningEffort.init(rawValue:)))
    if let model = change.model, !model.isEmpty { server.activate(model) }
    guard let saved = chat.chat(id) else { throw missing(id) }
    return summary(of: saved)
  }

  func send(_ id: UUID, text: String) throws {
    guard let saved = chat.chat(id) else { throw missing(id) }
    guard saved.workspace != nil else {
      throw LinkFailure(
        code: "no_workspace", message: "this conversation has no folder to work in yet")
    }
    guard server.phase.isRunning else {
      // Nothing can be asked of a Mac whose pack is down, and a phone has no other way to bring
      // it up: the load is started here and the turn is asked for again once it is.
      if !server.phase.isBusy { server.start() }
      throw LinkFailure(
        code: "loading", message: "bringing the pack up on this Mac — ask again in a moment")
    }
    guard chat.send(text, in: id) else {
      throw LinkFailure(
        code: "busy", message: "this Mac is answering another conversation")
    }
  }

  func steer(_ id: UUID, text: String) throws {
    guard chat.chat(id) != nil else { throw missing(id) }
    chat.steer(text, in: id)
  }

  func stop(_ id: UUID) throws {
    guard chat.chat(id) != nil else { throw missing(id) }
    chat.stop(id)
  }

  /// One frame of a conversation's stream: everything a phone draws, gathered at once so the
  /// rows, the activity and the dials it shows all describe the same instant.
  func frame(_ id: UUID) throws -> TurnEvent {
    guard let saved = chat.chat(id) else { throw missing(id) }
    let meta = chat.meta(of: id)
    return TurnEvent(
      kind: .snapshot,
      rows: chat.rows(of: id).map { row(from: $0, meta: meta[$0.id]) },
      pendingSteers: chat.steers(of: id).map { row(from: $0, meta: nil) },
      activity: activity(of: saved),
      status: status(),
      summary: summary(of: saved),
      failure: chat.failure(of: id))
  }

  private func activity(of saved: SavedChat) -> LinkActivity {
    guard chat.isRunning(saved) else { return .idle }
    switch chat.activity {
    case .queued: return LinkActivity(phase: .queued, isRunning: true)
    case .reading(let fraction):
      return LinkActivity(phase: .reading, fraction: fraction, isRunning: true)
    case .writing: return LinkActivity(phase: .writing, isRunning: true)
    case .writingCommand:
      return LinkActivity(phase: .command, command: chat.writingCommand, isRunning: true)
    case .unknown: return LinkActivity(phase: .unknown, isRunning: true)
    }
  }

  func status() -> LinkStatus {
    let readout = server.readout
    return LinkStatus(
      model: server.settings.activeModelID.isEmpty ? nil : server.settings.activeModelID,
      loaded: readout?.isLoaded ?? false,
      contextTokens: readout?.context.peakTokens ?? 0,
      contextCeiling: readout?.context.ceilingTokens ?? 0,
      generatedTokens: readout?.totals.generatedTokens ?? 0,
      tokensPerSecond: readout?.totals.decodeRate ?? 0,
      cachedTokens: readout?.prefix?.hits ?? 0,
      heldBytes: readout?.load.held ?? 0,
      ceilingBytes: readout?.load.ceiling ?? 0,
      gpu: readout?.load.gpu,
      thermal: readout?.state.thermal)
  }

  func models() -> ModelList {
    ModelList(
      active: server.settings.activeModelID.isEmpty ? nil : server.settings.activeModelID,
      entries: server.catalog.entries.map { entry in
        ModelEntry(
          id: entry.id,
          name: entry.displayName,
          quantization: entry.quantization,
          contextTokens: entry.contextTokens,
          sizeBytes: entry.byteCount,
          loaded: entry.id == server.settings.activeModelID && server.phase.isRunning)
      })
  }

  func activate(model id: String) throws {
    guard server.catalog[id] != nil else {
      throw LinkFailure(code: "no_model", message: "no pack named \(id) on this Mac")
    }
    server.activate(id)
  }

  /// The folders a phone may open a conversation in: what the Mac has been told to share, and
  /// every folder a conversation is already working in.
  func roots() -> RootList {
    var seen = Set<String>()
    var roots: [WorkspaceRoot] = []
    for url in settings.allowedRoots where seen.insert(url.path).inserted {
      roots.append(WorkspaceRoot(path: url.path, name: url.lastPathComponent, isRecent: false))
    }
    for path in chat.chats.compactMap(\.workspace) where seen.insert(path).inserted {
      guard (try? resolve(path)) != nil else { continue }
      roots.append(
        WorkspaceRoot(
          path: path, name: URL(filePath: path).lastPathComponent, isRecent: true))
    }
    return RootList(roots: roots)
  }

  // MARK: - Files and shell

  func list(_ path: String) throws -> DirectoryListing {
    let url = try resolve(path)
    let contents =
      (try? FileManager.default.contentsOfDirectory(
        at: url,
        includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey, .contentModificationDateKey],
        options: [.skipsSubdirectoryDescendants])) ?? []
    let entries = contents.map { child -> DirectoryEntry in
      let values = try? child.resourceValues(forKeys: [
        .isDirectoryKey, .fileSizeKey, .contentModificationDateKey,
      ])
      return DirectoryEntry(
        name: child.lastPathComponent,
        path: child.path,
        isDirectory: values?.isDirectory ?? false,
        size: values?.fileSize ?? 0,
        modified: values?.contentModificationDate)
    }
    .sorted {
      $0.isDirectory == $1.isDirectory
        ? $0.name.localizedStandardCompare($1.name) == .orderedAscending
        : $0.isDirectory
    }
    let parent = url.deletingLastPathComponent()
    return DirectoryListing(
      path: url.path,
      parent: (try? resolve(parent.path)) != nil ? parent.path : nil,
      entries: entries)
  }

  func read(_ path: String, offset: Int, limit: Int) throws -> FileSlice {
    let url = try resolve(path)
    guard let data = FileManager.default.contents(atPath: url.path) else {
      throw LinkFailure(code: "no_file", message: "no such file: \(path)")
    }
    guard !data.prefix(1024).contains(0) else {
      return FileSlice(
        path: url.path, start: 0, end: 0, total: 0, lines: [], isBinary: true)
    }
    let lines = String(decoding: data, as: UTF8.self).components(separatedBy: "\n")
    let start = max(1, offset)
    guard start <= lines.count else {
      return FileSlice(
        path: url.path, start: start, end: start, total: lines.count, lines: [],
        isBinary: false)
    }
    let end = min(lines.count, start + max(1, limit) - 1)
    return FileSlice(
      path: url.path, start: start, end: end, total: lines.count,
      lines: Array(lines[(start - 1)..<end]), isBinary: false)
  }

  func write(_ write: FileWrite) throws {
    let url = try resolve(write.path)
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(write.contents.utf8).write(to: url, options: .atomic)
  }

  func shell(_ request: ShellRequest) async throws -> ShellOutcome {
    let started = Date()
    let result = try await host(for: request.cwd).run(
      request.command, cwd: try resolve(request.cwd ?? defaultDirectory().path),
      timeout: min(request.timeout ?? 120, 900), byteLimit: 256 * 1024)
    return ShellOutcome(
      stdout: result.stdout, stderr: result.stderr, exitCode: result.exitCode,
      truncated: result.truncated, seconds: -started.timeIntervalSinceNow)
  }

  func startJob(_ request: ShellRequest) async throws -> ShellJob {
    try await host(for: request.cwd).start(
      request.command, cwd: try resolve(request.cwd ?? defaultDirectory().path))
  }

  func jobs() async throws -> JobList {
    JobList(jobs: try await host(for: nil).jobs())
  }

  func jobOutput(_ id: String, wait: Double, limit: Int) async throws -> ShellJobOutput {
    do {
      return try await host(for: nil).read(
        job: id, wait: min(max(0, wait), 120), byteLimit: min(max(1024, limit), 256 * 1024))
    } catch ShellError.noSuchJob {
      throw missingJob(id)
    }
  }

  func stopJob(_ id: String, force: Bool) async throws -> ShellJob {
    do {
      return try await host(for: nil).stop(job: id, force: force)
    } catch ShellError.noSuchJob {
      throw missingJob(id)
    }
  }

  private func missingJob(_ id: String) -> LinkFailure {
    LinkFailure(code: "no_job", message: "there is no job called \(id) on this Mac")
  }

  /// A shell rooted wherever the phone is working, sharing one job table with every other
  /// request so a command started by one call is still there for the next.
  private func host(for cwd: String?) throws -> LocalShellHost {
    guard settings.allowShell else {
      throw LinkFailure(
        code: "shell_off", message: "this Mac is not sharing its shell")
    }
    let directory = try resolve(cwd ?? defaultDirectory().path)
    let root = try rootContaining(directory) ?? directory
    return LocalShellHost(workspace: root, jobs: jobTable)
  }

  private func defaultDirectory() -> URL {
    chat.workspace ?? settings.allowedRoots.first
      ?? FileManager.default.homeDirectoryForCurrentUser
  }

  // MARK: - What a phone is allowed to reach

  func resolve(_ path: String) throws -> URL {
    let expanded = (path as NSString).expandingTildeInPath
    let url = URL(filePath: expanded).standardizedFileURL
    guard try rootContaining(url) != nil else {
      throw LinkFailure(
        code: "outside_roots",
        message: "\(path) is outside the folders this Mac is sharing")
    }
    return url
  }

  private func rootContaining(_ url: URL) throws -> URL? {
    let resolved = url.resolvingSymlinksInPath().standardizedFileURL.path
    for root in settings.allowedRoots {
      let base = root.resolvingSymlinksInPath().standardizedFileURL.path
      if resolved == base || resolved.hasPrefix(base + "/") { return root }
    }
    return nil
  }

  private func missing(_ id: UUID) -> LinkFailure {
    LinkFailure(code: "no_chat", message: "no conversation \(id.uuidString) on this Mac")
  }

  private func row(from row: ChatController.Row, meta: ChatController.RowMeta?) -> TranscriptRow {
    let kind: TranscriptRow.Kind
    var tool: String?
    switch row.kind {
    case .system: kind = .system
    case .prompt: kind = .prompt
    case .steer: kind = .steer
    case .reasoning: kind = .reasoning
    case .answer: kind = .answer
    case .toolCall(let name):
      kind = .toolCall
      tool = name
    case .toolOutput(let name):
      kind = .toolOutput
      tool = name
    }
    return TranscriptRow(
      id: row.id, kind: kind, tool: tool, text: row.text, at: meta?.at,
      seconds: meta?.seconds, tokens: meta?.tokens, wasRead: meta?.wasRead ?? row.kind.isInput)
  }
}
