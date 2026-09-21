// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The glass the panels are cut from — clear, since the window already lays down a material.

import AppKit
import SwiftUI

extension Color {
  /// A calmer stand-in for the system accent blue, which reads as too saturated on glass.
  static let accentSoft = Color(
    NSColor(name: nil) { appearance in
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        ? NSColor(red: 0.53, green: 0.73, blue: 0.96, alpha: 1)
        : NSColor(red: 0.22, green: 0.47, blue: 0.78, alpha: 1)
    })
}

extension View {
  /// The window's own backdrop, in one place so the dashboard and the chat cannot drift and so
  /// how clear the app reads is a single value to turn. Clear glass rather than a material:
  /// a material's blur is what reads as frost.
  func windowBackdrop() -> some View {
    containerBackground(for: .window) {
      Rectangle()
        .fill(.clear)
        .glassEffect(.clear, in: .rect(cornerRadius: 0))
        .ignoresSafeArea()
    }
  }
}

struct GlassCard<Content: View>: View {
  var padding: CGFloat = 12
  var radius: CGFloat = 14
  @ViewBuilder var content: Content

  @Environment(\.controlActiveState) private var activeState
  private var focused: Bool { activeState == .key }

  var body: some View {
    content
      .padding(padding)
      .frame(maxWidth: .infinity, alignment: .leading)
      .glassEffect(.clear, in: .rect(cornerRadius: radius))
      .overlay {
        RoundedRectangle(cornerRadius: radius)
          .strokeBorder(.white.opacity(focused ? 0.12 : 0.05), lineWidth: 0.5)
      }
      .shadow(
        color: .black.opacity(focused ? 0.12 : 0.04),
        radius: focused ? 8 : 3,
        y: focused ? 2 : 1
      )
      .opacity(focused ? 1 : 0.94)
      .animation(.easeOut(duration: 0.18), value: focused)
  }
}

struct GlassSection<Content: View>: View {
  let title: String
  @ViewBuilder var content: Content

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(title)
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(.secondary)
        .textCase(.uppercase)
        .padding(.leading, 4)
      GlassCard { content }
    }
  }
}
