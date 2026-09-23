// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The companion: a phone with one Mac in it. Pair once, then the Mac's conversations, its files
// and its shell are all here.

import IshizukiKit
import IshizukiLink
import SwiftUI
import UIKit

@main
struct PhoneApp: App {
  @State private var store = LinkStore()
  @State private var chats: ChatsModel
  @State private var local = LocalSession()

  init() {
    let store = LinkStore()
    _store = State(initialValue: store)
    _chats = State(initialValue: ChatsModel(store: store))
    Self.setSerifTitles()
  }

  /// The navigation bar draws its titles itself, out of reach of the serif set on the views.
  private static func setSerifTitles() {
    func serif(_ style: UIFont.TextStyle, _ weight: UIFont.Weight) -> UIFont {
      let base = UIFont.preferredFont(forTextStyle: style)
      let weighted = base.fontDescriptor.addingAttributes([
        .traits: [UIFontDescriptor.TraitKey.weight: weight]
      ])
      let descriptor = weighted.withDesign(.serif) ?? weighted
      return UIFont(descriptor: descriptor, size: 0)
    }
    let bar = UINavigationBar.appearance()
    bar.largeTitleTextAttributes = [.font: serif(.largeTitle, .bold)]
    bar.titleTextAttributes = [.font: serif(.headline, .semibold)]
  }

  var body: some Scene {
    WindowGroup {
      RootView(store: store, chats: chats, local: local)
        .tint(.moss)
        .fontDesign(.serif)
        .task { await store.connect() }
        .onOpenURL { url in
          guard let ticket = LinkTicket(url: url) else { return }
          Task { try? await store.pair(with: ticket) }
        }
    }
  }
}

struct RootView: View {
  let store: LinkStore
  let chats: ChatsModel
  let local: LocalSession

  var body: some View {
    ZStack {
      Color.paper.ignoresSafeArea()
      if store.known == nil {
        PairView(store: store, local: local)
      } else {
        TabView {
          Tab("Chats", systemImage: "bubble.left.and.text.bubble.right") {
            ChatListView(store: store, chats: chats, local: local)
          }
          Tab("Files", systemImage: "folder") {
            FilesView(store: store, chats: chats)
          }
          Tab("Shell", systemImage: "terminal") {
            ShellView(store: store, chats: chats)
          }
          Tab("Mac", systemImage: "desktopcomputer") {
            PhoneSettingsView(store: store, chats: chats, local: local)
          }
        }
      }
    }
  }
}
