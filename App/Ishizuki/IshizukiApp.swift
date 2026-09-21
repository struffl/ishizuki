// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The menu bar app: a bonsai in the status bar, a dashboard behind it.

import IshizukiKit
import SwiftUI

@main
struct IshizukiApp: App {
  @State private var controller: ServerController
  @State private var chat: ChatController
  @State private var companion = CompanionServer()

  init() {
    let controller = ServerController()
    _controller = State(initialValue: controller)
    _chat = State(initialValue: ChatController(server: controller))
  }

  var body: some Scene {
    MenuBarExtra {
      MenuBarView(controller: controller)
    } label: {
      BonsaiGlyph()
    }
    .menuBarExtraStyle(.window)

    // Named for what it holds rather than for the app: the app's own name tells nobody which
    // of its windows they are looking at, here or in the Window menu.
    Window("Dashboard", id: "dashboard") {
      DashboardView(controller: controller, chat: chat, companion: companion)
        .task {
          controller.bootstrap()
          companion.attach(chat: chat, server: controller)
        }
        .onAppear { NSApp.setActivationPolicy(.regular) }
        .onDisappear { NSApp.setActivationPolicy(.accessory) }
    }
    .defaultSize(width: 760, height: 640)
    .defaultLaunchBehavior(.presented)
    .keyboardShortcut("0", modifiers: .command)
    .commands { IshizukiCommands(chat: chat, controller: controller) }

    // The same conversation, given room: a coding session wants more than a tab.
    Window("Agent", id: "agent") {
      ChatView(chat: chat, controller: controller)
        .task { controller.bootstrap() }
        .onAppear { NSApp.setActivationPolicy(.regular) }
    }
    .defaultSize(width: 720, height: 780)
    .keyboardShortcut("j", modifiers: [.command, .shift])

    // Its own window, reached from the app menu at Command-comma, rather than a tab behind
    // the dashboard's own content.
    Settings {
      SettingsView(controller: controller, companion: companion)
    }
  }
}
