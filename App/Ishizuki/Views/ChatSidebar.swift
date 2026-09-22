// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Every conversation that has been had, gathered under the folder it was had in. A chat is
// saved with the pack that answered it and the folder it worked in, so reopening one is not a
// guess — and the folder is picked here rather than from a bar over the transcript.

import IshizukiKit
import SwiftUI

@available(macOS 27.0, *)
struct ChatSidebar: View {
  @Bindable var chat: ChatController
  @Bindable var controller: ServerController

  @State private var renaming: SavedChat?
  @State private var draftTitle = ""
  /// Folders someone has shut. Kept by path rather than by index, so a folder that moves up the
  /// list as one of its chats is answered does not drag another one's state with it.
  @State private var collapsed: Set<String> = []

  var body: some View {
    List(selection: selection) {
      ForEach(chat.folders) { folder in
        Section {
          if !collapsed.contains(folder.id) {
            ForEach(folder.chats) { saved in
              row(saved)
                .tag(saved.id)
                .contextMenu {
                  Button("Rename…") { beginRenaming(saved) }
                  if let url = folder.url {
                    Button("Reveal in Finder") {
                      NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                  }
                  Divider()
                  Button("Delete", role: .destructive) { chat.delete(saved) }
                    .disabled(chat.isRunning(saved))
                }
            }
          }
        } header: {
          header(folder)
        }
      }
    }
    .listStyle(.sidebar)
    .safeAreaInset(edge: .top) { toolbar }
    .alert("Rename chat", isPresented: renamingBinding) {
      TextField("Title", text: $draftTitle)
      Button("Cancel", role: .cancel) { renaming = nil }
      Button("Rename") {
        if let renaming { chat.rename(renaming, to: draftTitle) }
        renaming = nil
      }
    }
  }

  /// One button, and everything a new conversation needs to decide: which folder it is in.
  /// The folder bar that used to sit over the transcript is this menu now.
  @ViewBuilder private var toolbar: some View {
    HStack(spacing: 6) {
      Button {
        chat.startNewChat()
      } label: {
        Label("New chat", systemImage: "square.and.pencil")
      }
      .buttonStyle(.plain)
      .font(.subheadline)
      .frame(minHeight: Metrics.hit)
      .contentShape(.rect)
      .help("Start a new conversation in \(chat.workspace?.lastPathComponent ?? "no folder")")

      Spacer()

      Menu {
        if let current = chat.workspace {
          Button("New chat in \(current.lastPathComponent)") { chat.startNewChat(in: current) }
          Divider()
        }
        ForEach(others, id: \.self) { url in
          Button(url.lastPathComponent) { chat.startNewChat(in: url) }
        }
        if !others.isEmpty { Divider() }
        Button("Other folder…") { chat.startNewChatInChosenFolder() }
      } label: {
        Image(systemName: "plus")
          .font(.subheadline)
          .hitTarget()
      }
      .menuStyle(.borderlessButton)
      .menuIndicator(.hidden)
      .fixedSize()
      .accessibilityLabel("New chat in a folder")
      .help("Start a conversation somewhere else")
    }
    .padding(.horizontal, 12)
    .padding(.vertical, 6)
  }

  /// Folders worth offering, which is the ones lately worked in minus the one already open.
  private var others: [URL] {
    chat.recentWorkspaces.filter { $0 != chat.workspace }.prefix(6).map { $0 }
  }

  @ViewBuilder private func header(_ folder: ChatController.FolderGroup) -> some View {
    let shut = collapsed.contains(folder.id)
    let isOpen = folder.path != nil && folder.path == chat.workspace?.path

    Button {
      if shut { collapsed.remove(folder.id) } else { collapsed.insert(folder.id) }
    } label: {
      HStack(spacing: 5) {
        Image(systemName: shut ? "chevron.right" : "chevron.down")
          .font(.system(size: 9, weight: .semibold))
          .foregroundStyle(.tertiary)
        Image(systemName: folder.path == nil ? "questionmark.folder" : "folder")
          .font(.footnote)
          .foregroundStyle(isOpen ? Color.reading : .secondary)
        Text(folder.name)
          .font(.system(.footnote, design: .monospaced, weight: .medium))
          .lineLimit(1)
          .truncationMode(.head)
        // Only the folder being worked in says which branch it is on: the others would each
        // cost a git call on every tick, and none of them is the one about to be changed.
        if isOpen, let status = chat.git.status {
          Text(status.summary)
            .font(.system(size: 10, design: .monospaced))
            .foregroundStyle(status.isClean ? Color.secondary : Color.instructing)
            .lineLimit(1)
        }
        Spacer(minLength: 0)
        Text("\(folder.chats.count)")
          .font(.system(size: 10, design: .monospaced))
          .foregroundStyle(.tertiary)
      }
      .frame(minHeight: Metrics.hit)
      .contentShape(.rect)
    }
    .buttonStyle(.plain)
    .accessibilityAddTraits(.isToggle)
    .help(folder.path ?? "Conversations with no folder of their own")
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
