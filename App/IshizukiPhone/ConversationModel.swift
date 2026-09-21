// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One conversation as the phone watches it: rows merged by id as they change, and a stream that
// comes back on its own when the network drops out from under it.

import Foundation
import IshizukiKit
import IshizukiLink
import Observation

@MainActor
@Observable
final class ConversationModel {
  let id: UUID

  private(set) var rows: [TranscriptRow] = []
  private(set) var steers: [TranscriptRow] = []
  private(set) var activity: LinkActivity = .idle
  private(set) var status: LinkStatus?
  private(set) var summary: ChatSummary?
  private(set) var failure: String?
  private(set) var isWatching = false

  var draft = ""

  private let store: LinkStore
  private var watcher: Task<Void, Never>?
  private var watchID = UUID()
  private var lastSaved = Date.distantPast
  private(set) var showingCached = false

  init(id: UUID, summary: ChatSummary?, store: LinkStore) {
    self.id = id
    self.summary = summary
    self.store = store
  }

  /// What the transcript draws: a steer has not been said yet, and the instructions are the
  /// Mac's business rather than something to read on a phone.
  var visibleRows: [TranscriptRow] {
    rows.filter { $0.kind != .steer && $0.kind != .system }
  }

  var isRunning: Bool { activity.isRunning }

  var canSend: Bool {
    !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && store.phase.isReady
  }

  func start() {
    guard watcher == nil else { return }
    let token = UUID()
    watchID = token
    watcher = Task { await watch(token) }
  }

  func stop() {
    watchID = UUID()
    watcher?.cancel()
    watcher = nil
    isWatching = false
  }

  private func watch(_ token: UUID) async {
    let serverID = store.known?.id
    let cache = store.cache
    if rows.isEmpty, let held = await cache?.load(ChatDetail.self, key: "chat-" + id.uuidString) {
      guard token == watchID, store.known?.id == serverID, !Task.isCancelled else { return }
      absorb(held)
      showingCached = true
    }
    while !Task.isCancelled && token == watchID && store.known?.id == serverID {
      guard let client = store.client else {
        await store.connect()
        guard !Task.isCancelled else { return }
        try? await Task.sleep(for: .seconds(2))
        continue
      }
      do {
        let detail = try await client.chat(id)
        guard token == watchID, store.known?.id == serverID, !Task.isCancelled else { return }
        absorb(detail)
        showingCached = false
        await cache?.save(detail, key: "chat-" + id.uuidString)
        guard token == watchID, store.known?.id == serverID, !Task.isCancelled else { return }
        isWatching = true
        for try await event in client.events(id) {
          guard token == watchID, store.known?.id == serverID, !Task.isCancelled else { return }
          apply(event)
          if event.kind == .rows || event.kind == .snapshot || event.kind == .done,
            Date().timeIntervalSince(lastSaved) >= 3 || event.kind == .done
          {
            if let summary {
              await cache?.save(
                ChatDetail(summary: summary, rows: rows, pendingSteers: steers, failure: failure),
                key: "chat-" + id.uuidString)
              lastSaved = Date()
            }
          }
        }
        guard token == watchID, !Task.isCancelled else { return }
        isWatching = false
        showingCached = !rows.isEmpty
      } catch {
        guard token == watchID, store.known?.id == serverID, !Task.isCancelled else { return }
        isWatching = false
        showingCached = !rows.isEmpty
        failure = error.localizedDescription
        await store.refresh()
      }
      guard !Task.isCancelled else { return }
      try? await Task.sleep(for: .seconds(2))
    }
  }

  private func absorb(_ detail: ChatDetail) {
    rows = detail.rows
    steers = detail.pendingSteers
    summary = detail.summary
    failure = detail.failure
  }

  private func apply(_ event: TurnEvent) {
    switch event.kind {
    case .snapshot:
      rows = event.rows ?? rows
      steers = event.pendingSteers ?? steers
      activity = event.activity ?? activity
      status = event.status ?? status
      summary = event.summary ?? summary
      failure = event.failure
    case .rows:
      merge(event.rows ?? [])
      steers = event.pendingSteers ?? steers
      summary = event.summary ?? summary
      failure = event.failure
    case .activity:
      activity = event.activity ?? .idle
    case .status:
      status = event.status ?? status
    case .failure:
      failure = event.failure
    case .done:
      activity = .idle
    case .ping:
      break
    }
  }

  /// Rows arrive as whichever of them changed. A transcript only ever grows, so a row already
  /// held is replaced in place and anything new goes on the end.
  private func merge(_ incoming: [TranscriptRow]) {
    guard !incoming.isEmpty else { return }
    var index = Dictionary(
      rows.enumerated().map { ($0.element.id, $0.offset) }, uniquingKeysWith: { first, _ in first })
    for row in incoming {
      if let at = index[row.id] {
        rows[at] = row
      } else {
        index[row.id] = rows.count
        rows.append(row)
      }
    }
  }

  func send() async {
    let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty, let client = store.client else { return }
    draft = ""
    failure = nil
    do {
      if isRunning {
        try await client.steer(id, text: text)
      } else {
        try await client.send(id, text: text)
      }
      start()
    } catch {
      failure = error.localizedDescription
      draft = text
    }
  }

  func stopTurn() async {
    guard let client = store.client else { return }
    try? await client.stop(id)
  }
}
