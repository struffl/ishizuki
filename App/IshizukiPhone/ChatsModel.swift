// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Every conversation the Mac holds, as the phone's list of them.

import Foundation
import IshizukiKit
import IshizukiLink
import Observation

@MainActor
@Observable
final class ChatsModel {
  private(set) var chats: [ChatSummary] = []
  private(set) var roots: [WorkspaceRoot] = []
  private(set) var models: ModelList?
  private(set) var failure: String?
  private(set) var isLoading = false

  private let store: LinkStore
  private var cacheServerID: String?

  init(store: LinkStore) {
    self.store = store
  }

  func refresh() async {
    guard !isLoading else { return }
    isLoading = true
    defer { isLoading = false }
    let serverID = store.known?.id
    if cacheServerID != serverID {
      chats = []
      roots = []
      models = nil
      failure = nil
      cacheServerID = serverID
    }
    let cache = store.cache
    if chats.isEmpty, let held = await cache?.load([ChatSummary].self, key: "chats") {
      guard store.known?.id == serverID, !Task.isCancelled else { return }
      chats = held
    }
    if roots.isEmpty, let held = await cache?.load([WorkspaceRoot].self, key: "roots") {
      guard store.known?.id == serverID, !Task.isCancelled else { return }
      roots = held
    }
    if models == nil, let held = await cache?.load(ModelList.self, key: "models") {
      guard store.known?.id == serverID, !Task.isCancelled else { return }
      models = held
    }
    if store.client == nil { await store.connect() }
    guard store.known?.id == serverID, !Task.isCancelled else { return }
    guard let client = store.client else {
      failure = "Offline — showing saved content. Pull to retry."
      return
    }
    do {
      let fresh = try await client.chats()
      guard store.known?.id == serverID, !Task.isCancelled else { return }
      chats = fresh
      await cache?.save(fresh, key: "chats")
      failure = nil
      // One unavailable section must not discard successfully loaded conversations.
      if let freshRoots = try? await client.roots() {
        guard store.known?.id == serverID, !Task.isCancelled else { return }
        roots = freshRoots.roots
        await cache?.save(roots, key: "roots")
      }
      if let freshModels = try? await client.models() {
        guard store.known?.id == serverID, !Task.isCancelled else { return }
        models = freshModels
        await cache?.save(freshModels, key: "models")
      }
    } catch {
      guard store.known?.id == serverID, !Task.isCancelled else { return }
      failure = "Showing saved content. " + error.localizedDescription
      await store.connect()
    }
  }

  func create(in folder: String?) async -> ChatSummary? {
    guard let client = store.client else { return nil }
    do {
      let made = try await client.create(NewChat(workspace: folder))
      chats.insert(made, at: 0)
      await store.cache?.save(chats, key: "chats")
      failure = nil
      return made
    } catch {
      failure = error.localizedDescription
      return nil
    }
  }

  func delete(_ chat: ChatSummary) async {
    guard let client = store.client else { return }
    do {
      try await client.delete(chat.id)
      chats.removeAll { $0.id == chat.id }
      await store.cache?.save(chats, key: "chats")
      await store.cache?.remove("chat-" + chat.id.uuidString)
    } catch {
      failure = error.localizedDescription
    }
  }

  func activate(model id: String) async {
    guard let client = store.client else { return }
    do {
      try await client.activate(model: id)
      models = try await client.models()
    } catch {
      failure = error.localizedDescription
    }
  }

  func absorb(_ summary: ChatSummary) {
    if let index = chats.firstIndex(where: { $0.id == summary.id }) {
      chats[index] = summary
    } else {
      chats.insert(summary, at: 0)
    }
  }
}
