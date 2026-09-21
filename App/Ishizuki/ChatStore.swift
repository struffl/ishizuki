// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Conversations on disk. The session's own transcript is Codable, so a saved chat is the whole
// of what was said rather than a retelling of it, and it goes back into a session unchanged.

import Foundation
import FoundationModels
import IshizukiKit

@available(macOS 27.0, *)
struct SavedChat: Codable, Identifiable, Equatable {
  var id: UUID
  var title: String
  var created: Date
  var updated: Date
  /// The folder this conversation was working in, so reopening it picks up where it was.
  var workspace: String?
  var model: String?
  var effort: ReasoningEffort
  var transcript: Transcript
  /// The tokens of this conversation's last prompt. Kept so the archives holding its prefix
  /// can be found again: an archive belongs to a chat when the chat's prompt begins with it.
  var promptTokens: [Int] = []
  /// A title someone typed is never overwritten by one that was written for them.
  var titleIsCustom = false

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

  /// A name taken from the first thing asked, until someone gives it a better one.
  static func title(from transcript: Transcript) -> String? {
    for entry in transcript {
      guard case .prompt(let prompt) = entry else { continue }
      let text = prompt.segments
        .compactMap { if case .text(let t) = $0 { t.content } else { nil } }
        .joined(separator: " ")
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .replacing(/\s+/, with: " ")
      guard !text.isEmpty else { continue }
      return text.count > 42 ? String(text.prefix(42)) + "…" : text
    }
    return nil
  }
}

@available(macOS 27.0, *)
final class ChatStore {
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
    guard let data = try? encoder.encode(chat) else { return }
    try? data.write(to: url(for: chat.id), options: .atomic)
  }

  /// A mid-turn checkpoint, written away from the main thread: encoding a long transcript is
  /// not something the window should stop drawing tokens for. It carries its own encoder,
  /// since the one above belongs to whoever is on the main thread.
  static func write(_ chat: SavedChat, in directory: URL) {
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    guard let data = try? encoder.encode(chat) else { return }
    try? data.write(
      to: directory.appending(path: "\(chat.id.uuidString).json"), options: .atomic)
  }

  var folder: URL { directory }

  func delete(_ id: UUID) {
    try? FileManager.default.removeItem(at: url(for: id))
  }

  /// Sheds the archives that belonged to a deleted conversation and to nothing else. Every
  /// chat begins with the same instructions, so an archive is only this one's if no other
  /// conversation still starts with it.
  func pruneCache(for chat: SavedChat, keeping others: [SavedChat], in store: PrefixStore) {
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
