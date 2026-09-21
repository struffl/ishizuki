// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The menu bar app: a bonsai in the status bar, a dashboard behind it.

import IshizukiKit
import SwiftUI

@main
struct IshizukiApp: App {
  @State private var controller = ServerController()

  var body: some Scene {
    MenuBarExtra {
      MenuBarView(controller: controller)
        .tint(.accentSoft)
    } label: {
      BonsaiGlyph()
    }
    .menuBarExtraStyle(.window)

    Window("Ishizuki", id: "dashboard") {
      DashboardView(controller: controller)
        .tint(.accentSoft)
        .task { controller.bootstrap() }
        .onAppear { NSApp.setActivationPolicy(.regular) }
        .onDisappear { NSApp.setActivationPolicy(.accessory) }
    }
    .defaultSize(width: 760, height: 640)
    .defaultLaunchBehavior(.presented)
  }
}
