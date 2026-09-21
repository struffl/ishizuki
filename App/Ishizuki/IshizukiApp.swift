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

  init() {
    let controller = ServerController()
    _controller = State(initialValue: controller)
    _chat = State(initialValue: ChatController(server: controller))
  }

  var body: some Scene {
    MenuBarExtra {
      MenuBarView(controller: controller)
        .tint(.accentSoft)
    } label: {
      BonsaiGlyph()
    }
    .menuBarExtraStyle(.window)

    Window("Ishizuki", id: "dashboard") {
      DashboardView(controller: controller, chat: chat)
        .tint(.accentSoft)
        .task { controller.bootstrap() }
        .onAppear { NSApp.setActivationPolicy(.regular) }
        .onDisappear { NSApp.setActivationPolicy(.accessory) }
    }
    .defaultSize(width: 760, height: 640)
    .defaultLaunchBehavior(.presented)

    // The same conversation, given room: a coding session wants more than a tab.
    Window("Agent", id: "agent") {
      ChatView(chat: chat, controller: controller)
        .tint(.accentSoft)
        .task { controller.bootstrap() }
        .onAppear { NSApp.setActivationPolicy(.regular) }
    }
    .defaultSize(width: 720, height: 780)
    .keyboardShortcut("j", modifiers: [.command, .shift])
  }
}
