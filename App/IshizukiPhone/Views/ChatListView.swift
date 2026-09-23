// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The Mac's conversations, newest first, and a way to start one in any folder it shares.

import IshizukiKit
import IshizukiLink
import SwiftUI

struct ChatListView: View {
  let store: LinkStore
  let chats: ChatsModel
  let local: LocalSession

  @Environment(\.scenePhase) private var scenePhase
  @State private var choosingFolder = false
  @State private var opened: ChatSummary?

  var body: some View {
    NavigationStack {
      List {
        if case .offline(let why) = store.phase {
          Section {
            VStack(alignment: .leading, spacing: 8) {
              Label(why, systemImage: "wifi.exclamationmark")
                .font(.body)
                .foregroundStyle(.secondary)
              Button("Try again") { Task { await chats.refresh() } }
                .buttonStyle(.borderless)
            }
          }
        }

        Section {
          NavigationLink {
            LocalChatView(local: local)
          } label: {
            Label {
              VStack(alignment: .leading, spacing: 2) {
                Text("Ask this iPhone")
                Text(local.blocker ?? Self.readyDetail(local))
                  .font(.footnote)
                  .foregroundStyle(.secondary)
              }
            } icon: {
              Image(systemName: "iphone.gen3")
            }
          }
        }

        Section(store.known?.name ?? "Mac") {
          ForEach(chats.chats) { chat in
            NavigationLink {
              ConversationView(
                model: ConversationModel(id: chat.id, summary: chat, store: store), chats: chats)
            } label: {
              ChatRow(chat: chat)
            }
          }
          .onDelete { offsets in
            let doomed = offsets.map { chats.chats[$0] }
            Task {
              for chat in doomed { await chats.delete(chat) }
            }
          }
          if chats.chats.isEmpty, !chats.isLoading {
            Text("No conversations on this Mac yet.")
              .foregroundStyle(.secondary)
          }
        }

        if let failure = chats.failure {
          Section {
            Text(failure).font(.body).foregroundStyle(.red)
          }
        }
      }
      .glassList()
      .navigationTitle("Conversations")
      .toolbar {
        ToolbarItem(placement: .topBarTrailing) {
          Button("New conversation", systemImage: "plus") { choosingFolder = true }
            .disabled(!store.phase.isReady)
        }
      }
      .refreshable { await chats.refresh() }
      .task {
        while !Task.isCancelled {
          await chats.refresh()
          do { try await Task.sleep(for: .seconds(15)) } catch { return }
        }
      }
      .onChange(of: scenePhase) { _, phase in
        if phase == .active { Task { await chats.refresh() } }
      }
      .sheet(isPresented: $choosingFolder) {
        FolderPicker(chats: chats) { folder in
          choosingFolder = false
          guard let folder else { return }
          Task {
            guard let made = await chats.create(in: folder) else { return }
            opened = made
          }
        }
      }
      .navigationDestination(item: $opened) { chat in
        ConversationView(
          model: ConversationModel(id: chat.id, summary: chat, store: store), chats: chats)
      }
    }
  }

  /// What the phone is answering with, named where the system names it.
  private static func readyDetail(_ local: LocalSession) -> String {
    guard #available(iOS 27.0, *) else { return "Apple's on-device model, no files, no shell" }
    return "\(local.variantName), no files, no shell"
  }
}

struct ChatRow: View {
  let chat: ChatSummary

  var body: some View {
    VStack(alignment: .leading, spacing: 3) {
      HStack(spacing: 6) {
        if chat.isRunning {
          AnimatedDots(size: 3, tint: Color.reading)
        }
        Text(chat.title)
          .font(.body)
          .lineLimit(1)
      }
      if let preview = chat.preview, !preview.isEmpty {
        Text(preview)
          .font(.footnote)
          .foregroundStyle(.secondary)
          .lineLimit(2)
      }
      HStack(spacing: 6) {
        if let workspace = chat.workspace {
          Text(URL(filePath: workspace).lastPathComponent)
            .font(.system(size: 11).monospacedDigit())
            .foregroundStyle(.tertiary)
        }
        Text(chat.updated, style: .relative)
          .font(.system(size: 11).monospacedDigit())
          .foregroundStyle(.tertiary)
      }
    }
  }
}

/// Which folder a new conversation works in. Only what the Mac offered: a phone never names a
/// path of its own.
struct FolderPicker: View {
  let chats: ChatsModel
  let chosen: (String?) -> Void

  var body: some View {
    NavigationStack {
      List {
        Section("Folders this Mac shares") {
          ForEach(chats.roots) { root in
            Button {
              chosen(root.path)
            } label: {
              VStack(alignment: .leading, spacing: 2) {
                Text(root.name)
                Text(root.path)
                  .font(.system(size: 11))
                  .foregroundStyle(.secondary)
                  .lineLimit(1)
                  .truncationMode(.head)
              }
            }
          }
          if chats.roots.isEmpty {
            Text("This Mac is not sharing any folder yet.")
              .foregroundStyle(.secondary)
          }
        }
      }
      .glassList()
      .navigationTitle("Work where?")
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          Button("Cancel") { chosen(nil) }
        }
      }
    }
  }
}
