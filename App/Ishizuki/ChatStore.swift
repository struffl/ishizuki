// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Conversations on disk. The session's own transcript is Codable, so a saved chat is the whole
// of what was said rather than a retelling of it, and it goes back into a session unchanged.

import Foundation
import FoundationModels
import IshizukiKit

@available(macOS 27.0, *)
/// What a conversation's turns have cost: how long they took and what they wrote. Kept with
/// the conversation rather than the window, so the corner readout is about what is on screen.
struct TurnMeter: Codable, Equatable {
  var turns = 0
  var seconds = 0.0
  var tokens = 0

  var averageSeconds: Double { turns > 0 ? seconds / Double(turns) : 0 }
  var averageTokens: Int { turns > 0 ? tokens / turns : 0 }
}

struct SavedChat: Codable, Identifiable, Equatable {
  var id: UUID
  var title: String
  var created: Date
  var updated: Date
  /// The folder this conversation was working in, so reopening it picks up where it was.
  var workspace: String?
  /// The pack that answers it, fixed at its first turn along with `effort`: its cache is built
  /// on both, so changing either means branching.
  var model: String?
  /// One of Apple's own models, when that is what answers it instead of a pack.
  var appleModelID: String?
  /// The conversation this one was branched from.
  var parent: UUID?
  var effort: ReasoningEffort
  var transcript: Transcript
  /// The tokens of this conversation's last prompt. Kept so the archives holding its prefix
  /// can be found again: an archive belongs to a chat when the chat's prompt begins with it.
  var promptTokens: [Int] = []
  /// A title someone typed is never overwritten by one that was written for them.
  var titleIsCustom = false
  /// Where this conversation's commands run. Absent means whatever the app defaults to, which
  /// is what every conversation saved before there was a choice gets.
  var sandbox: SandboxChoice?
  /// What happened to a turn rather than in it. Optional so a chat written before there were
  /// any still decodes; read through `notes`.
  var notices: [ChatNotice]?

  var notes: [ChatNotice] {
    get { notices ?? [] }
    set { notices = newValue.isEmpty ? nil : newValue }
  }

  /// What this conversation's turns have cost. Optional so a chat written before there was a
  /// tally still decodes; read through `meter`.
  var turnMeter: TurnMeter?

  var meter: TurnMeter {
    get { turnMeter ?? TurnMeter() }
    set { turnMeter = newValue.turns > 0 ? newValue : nil }
  }

  init(
    id: UUID = UUID(), title: String = "New chat", workspace: String? = nil,
    model: String? = nil, effort: ReasoningEffort = .xhigh,
    transcript: Transcript = Transcript()
  ) {
    self.id = id
    self.title = title
    self.created = Date()
    self.updated = created
    self.workspace = workspace
    self.model = model
    self.effort = effort
    self.transcript = transcript
  }

  /// Whether an archive is a prefix of this conversation.
  func holds(_ tokens: [Int]) -> Bool {
    guard !tokens.isEmpty, tokens.count <= promptTokens.count else { return false }
    return Array(promptTokens.prefix(tokens.count)) == tokens
  }

  var isEmpty: Bool {
    !transcript.contains { if case .prompt = $0 { true } else { false } }
  }

  var appleModel: AppleFoundationModel? { appleModelID.flatMap(AppleFoundationModel.init) }

  /// A name taken from the first thing asked, until someone gives it a better one.
  static func title(from transcript: Transcript) -> String? {
    for entry in transcript {
      guard case .prompt(let prompt) = entry else { continue }
      let raw = prompt.segments
        .compactMap { if case .text(let t) = $0 { t.content } else { nil } }
        .joined(separator: " ")
      let text = PromptEnvironment.split(PromptAttachments.split(raw).body).body
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .replacing(/\s+/, with: " ")
      guard !text.isEmpty else { continue }
      return text.count > 42 ? String(text.prefix(42)) + "…" : text
    }
    return nil
  }
}

/// A turn that stopped or failed, kept beside the transcript.
///
/// The model's transcript has no entry for either — a stop leaves a half-written answer and a
/// failure leaves nothing at all — so the account of it lives here and is woven back into the
/// rows at the point it happened. It is shown in the conversation rather than in a bar under
/// it: what went wrong belongs where it went wrong, and it should still be there tomorrow.
struct ChatNotice: Codable, Equatable, Identifiable {
  enum Tone: String, Codable, Sendable {
    case stopped
    case failed
  }

  var id: UUID
  var tone: Tone
  var text: String
  var at: Date
  /// The row it follows, so it keeps its place when the transcript is folded again.
  var after: String?

  init(tone: Tone, text: String, after: String?, at: Date = Date()) {
    self.id = UUID()
    self.tone = tone
    self.text = text
    self.at = at
    self.after = after
  }
}

@available(macOS 27.0, *)
final class ChatStore {
  /// Checkpoints are made off the main actor. Keep only the newest one for each chat so a
  /// delayed disk write cannot put an earlier transcript back after a later checkpoint.
  private final class CheckpointWriter: @unchecked Sendable {
    private let lock = NSLock()
    private var issued: [UUID: Int] = [:]
    private var written: [UUID: Int] = [:]

    func reserve(for id: UUID) -> Int {
      lock.lock()
      defer { lock.unlock() }
      issued[id, default: 0] += 1
      return issued[id, default: 0]
    }

    func write(_ chat: SavedChat, in directory: URL, sequence: Int) {
      lock.lock()
      defer { lock.unlock() }
      guard sequence > written[chat.id, default: 0] else { return }
      written[chat.id] = sequence
      let encoder = JSONEncoder()
      encoder.dateEncodingStrategy = .iso8601
      guard let data = try? encoder.encode(chat) else { return }
      try? data.write(
        to: directory.appending(path: "\(chat.id.uuidString).json"), options: .atomic)
    }
  }

  private static let checkpointWriter = CheckpointWriter()
  private let directory: URL
  private let encoder = JSONEncoder()
  private let decoder = JSONDecoder()

  init() {
    directory = IshizukiPaths.models.deletingLastPathComponent().appending(path: "chats")
    try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    encoder.dateEncodingStrategy = .iso8601
    decoder.dateDecodingStrategy = .iso8601
  }

  /// Newest first, and a chat that will not decode is skipped rather than taking the list with
  /// it — a transcript written by an older build is not a reason to lose the others.
  func load() -> [SavedChat] {
    let files =
      (try? FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil)) ?? []
    return
      files
      .filter { $0.pathExtension == "json" }
      .compactMap { url in
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? decoder.decode(SavedChat.self, from: data)
      }
      .sorted { $0.updated > $1.updated }
  }

  func save(_ chat: SavedChat) {
    let sequence = Self.checkpointWriter.reserve(for: chat.id)
    Self.checkpointWriter.write(chat, in: directory, sequence: sequence)
  }

  /// A mid-turn checkpoint, written away from the main thread: encoding a long transcript is
  /// not something the window should stop drawing tokens for. It carries its own encoder,
  /// since the one above belongs to whoever is on the main thread.
  static func reserveCheckpoint(for id: UUID) -> Int {
    checkpointWriter.reserve(for: id)
  }

  static func writeCheckpoint(_ chat: SavedChat, in directory: URL, sequence: Int) {
    checkpointWriter.write(chat, in: directory, sequence: sequence)
  }

  var folder: URL { directory }

  func delete(_ id: UUID) {
    try? FileManager.default.removeItem(at: url(for: id))
  }

  /// Sheds the archives that belonged to a deleted conversation and to nothing else. Every
  /// chat begins with the same instructions, so an archive is only this one's if no other
  /// conversation still starts with it.
  func pruneCache(for chat: SavedChat, keeping others: [SavedChat], in store: PrefixStore) {
    store.remove(tag: chat.id.uuidString)
    guard !chat.promptTokens.isEmpty else { return }
    for entry in store.entries() where chat.holds(entry.tokens) {
      guard !others.contains(where: { $0.holds(entry.tokens) }) else { continue }
      _ = store.remove(entry.id)
    }
  }

  private func url(for id: UUID) -> URL {
    directory.appending(path: "\(id.uuidString).json")
  }
}
