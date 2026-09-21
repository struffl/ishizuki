// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The menu bar: every command the app has, where the platform says to look for it, whether
// or not there is a button for it somewhere in a window.

import IshizukiKit
import SwiftUI

@available(macOS 27.0, *)
struct IshizukiCommands: Commands {
  @Bindable var chat: ChatController
  @Bindable var controller: ServerController

  var body: some Commands {
    CommandGroup(replacing: .newItem) {
      Button("New Chat") { chat.startNewChat() }
        .keyboardShortcut("n")
      Button("Open Folder…") { chat.chooseWorkspace() }
        .keyboardShortcut("o", modifiers: [.command, .shift])
    }

    CommandMenu("Turn") {
      Button("Send") { chat.submit() }
        .keyboardShortcut(.return, modifiers: .command)
        .disabled(!chat.canSubmit)
      Button("Stop") { chat.stop() }
        .keyboardShortcut(".", modifiers: .command)
        .disabled(!chat.isResponding)
      Divider()
      Picker("Effort", selection: effort) {
        ForEach(ReasoningEffort.allCases, id: \.self) { level in
          Text(level.rawValue.capitalized).tag(level)
        }
      }
    }

    CommandMenu("Server") {
      Button(controller.phase.isRunning ? "Stop Server" : "Start Server") {
        controller.phase.isRunning ? controller.stop() : controller.start()
      }
      .keyboardShortcut("r", modifiers: [.command, .shift])
      .disabled(controller.phase.isBusy || controller.catalog.entries.isEmpty)
      Button("Restart Server") { controller.restart() }
        .disabled(!controller.phase.isRunning)
      Divider()
      Picker("Model", selection: model) {
        ForEach(controller.catalog.entries, id: \.id) { entry in
          Text(entry.displayName).tag(entry.id)
        }
      }
      .disabled(controller.catalog.entries.isEmpty)
      Button("Rescan for Packs") { controller.rescan() }
    }
  }

  private var effort: Binding<ReasoningEffort> {
    Binding(
      get: { chat.effort },
      set: {
        chat.effort = $0
        chat.saveEffort()
      })
  }

  private var model: Binding<String> {
    Binding(
      get: { controller.settings.activeModelID },
      set: { controller.activate($0) })
  }
}
