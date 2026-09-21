// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Every conversation that has been had, and what each one was had with. A chat is saved with
// the pack that answered it and the folder it worked in, so reopening one is not a guess.

import IshizukiKit
import SwiftUI

@available(macOS 27.0, *)
struct ChatSidebar: View {
  @Bindable var chat: ChatController
  @Bindable var controller: ServerController

  @State private var renaming: SavedChat?
  @State private var draftTitle = ""

  var body: some View {
    List(selection: selection) {
      ForEach(chat.chats) { saved in
        row(saved)
          .tag(saved.id)
          .contextMenu {
            Button("Rename…") { beginRenaming(saved) }
            Button("Delete", role: .destructive) { chat.delete(saved) }
              .disabled(chat.isRunning(saved))
          }
      }
    }
    .listStyle(.sidebar)
    .safeAreaInset(edge: .top) {
      HStack {
        Button {
          chat.startNewChat()
        } label: {
          Label("New chat", systemImage: "square.and.pencil")
        }
        .buttonStyle(.plain)
        .font(.subheadline)
        .frame(minHeight: Metrics.hit)
        .contentShape(.rect)
        .help("Start a new conversation")
        Spacer()
      }
      .padding(.horizontal, 12)
      .padding(.vertical, 6)
    }
    .alert("Rename chat", isPresented: renamingBinding) {
      TextField("Title", text: $draftTitle)
      Button("Cancel", role: .cancel) { renaming = nil }
      Button("Rename") {
        if let renaming { chat.rename(renaming, to: draftTitle) }
        renaming = nil
      }
    }
  }

  private var selection: Binding<SavedChat.ID?> {
    Binding(
      get: { chat.current.id },
      set: { id in
        guard let id, let picked = chat.chats.first(where: { $0.id == id }) else { return }
        chat.select(picked)
      })
  }

  private var renamingBinding: Binding<Bool> {
    Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })
  }

  private func beginRenaming(_ saved: SavedChat) {
    draftTitle = saved.title
    renaming = saved
  }

  @ViewBuilder private func row(_ saved: SavedChat) -> some View {
    VStack(alignment: .leading, spacing: 2) {
      HStack(spacing: 5) {
        Text(saved.title)
          .font(.callout)
          .lineLimit(1)
        // A turn keeps going in the conversation it was started in, so the one still being
        // answered says so from the list rather than only from its own transcript.
        if chat.isRunning(saved) {
          AnimatedDots(size: 3, tint: .generating)
        }
      }
      HStack(spacing: 5) {
        Text(saved.updated, format: .relative(presentation: .numeric))
          .lineLimit(1)
        if let pack = saved.model, pack != controller.settings.activeModelID {
          // Said only when it differs: reopening it will answer with the pack that is loaded,
          // not the one that wrote it.
          Text("· \(short(pack))")
            .lineLimit(1)
            .foregroundStyle(Color.instructing)
        }
      }
      .font(.footnote)
      .foregroundStyle(.secondary)
    }
    .padding(.vertical, 2)
  }

  private func short(_ id: String) -> String {
    let name = (id as NSString).lastPathComponent
    return name.count > 22 ? String(name.prefix(22)) + "…" : name
  }
}
