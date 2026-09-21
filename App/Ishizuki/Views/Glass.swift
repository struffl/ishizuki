// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The glass the panels are cut from: the window is clear, and text sits on plates.

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

extension Color {
  /// Reading is blue and writing is green, held apart because the wait for one feels nothing
  /// like the wait for the other.
  static let generating = Color(
    NSColor(name: nil) { appearance in
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        ? NSColor(red: 0.45, green: 0.85, blue: 0.58, alpha: 1)
        : NSColor(red: 0.16, green: 0.55, blue: 0.31, alpha: 1)
    })
}

/// Dimming rather than frosting. The system draws glass inactive — greyer, more opaque — while
/// a window is not key, and lets it sample the live backdrop once it is, so a clear window that
/// read beautifully over a dark neighbour turned unreadable the moment it came forward over a
/// white page. A veil sits over the glass, because glass samples what is behind the whole
/// window and never sees a fill placed under it. What is behind still changes how the app
/// looks; it no longer decides whether it can be read.
enum GlassTuning {
  static let windowKey = "glass.windowVeil"
  static let plateKey = "glass.plateVeil"
  static let windowDefault = 0.72
  static let plateDefault = 0.88
}

/// Clear glass under a veil. Both veils are settings rather than constants, because how much
/// of the desktop is too much is a matter of the desktop and of taste.
private struct WindowVeil: View {
  @AppStorage(GlassTuning.windowKey) private var veil = GlassTuning.windowDefault

  var body: some View {
    ZStack {
      Rectangle()
        .fill(.clear)
        .glassEffect(.clear, in: .rect(cornerRadius: 0))
      Rectangle()
        .fill(.background.opacity(veil))
    }
    .ignoresSafeArea()
  }
}

/// The veil goes over the glass rather than under it: glass samples what is behind the whole
/// window, so a fill underneath is a fill it never sees.
private struct TextPlate: ViewModifier {
  @AppStorage(GlassTuning.plateKey) private var veil = GlassTuning.plateDefault
  let radius: CGFloat
  let horizontal: CGFloat
  let vertical: CGFloat

  func body(content: Content) -> some View {
    content
      .padding(.horizontal, horizontal)
      .padding(.vertical, vertical)
      .background {
        ZStack {
          RoundedRectangle(cornerRadius: radius)
            .fill(.clear)
            .glassEffect(.regular, in: .rect(cornerRadius: radius))
          RoundedRectangle(cornerRadius: radius)
            .fill(.background.opacity(veil))
        }
      }
  }
}

extension View {
  func windowBackdrop() -> some View {
    containerBackground(for: .window) { WindowVeil() }
  }

  /// Anything carrying text sits on this.
  func textPlate(
    radius: CGFloat = 10, horizontal: CGFloat = 10, vertical: CGFloat = 7
  ) -> some View {
    modifier(TextPlate(radius: radius, horizontal: horizontal, vertical: vertical))
  }
}

struct GlassCard<Content: View>: View {
  var padding: CGFloat = 12
  var radius: CGFloat = 14
  @ViewBuilder var content: Content

  @AppStorage(GlassTuning.plateKey) private var veil = GlassTuning.plateDefault
  @Environment(\.controlActiveState) private var activeState
  private var focused: Bool { activeState == .key }

  var body: some View {
    content
      .padding(padding)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background {
        ZStack {
          RoundedRectangle(cornerRadius: radius)
            .fill(.clear)
            .glassEffect(.regular, in: .rect(cornerRadius: radius))
          RoundedRectangle(cornerRadius: radius)
            .fill(.background.opacity(veil))
        }
      }
      .overlay {
        RoundedRectangle(cornerRadius: radius)
          .strokeBorder(.white.opacity(focused ? 0.14 : 0.06), lineWidth: 0.5)
      }
      .shadow(
        color: .black.opacity(focused ? 0.12 : 0.04),
        radius: focused ? 8 : 3,
        y: focused ? 2 : 1
      )
      .opacity(focused ? 1 : 0.97)
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

/// Three dots that pulse in place. Fixed width on purpose: a label that grows and shrinks with
/// its own ellipsis drags whatever sits beside it back and forth.
struct AnimatedDots: View {
  var size: CGFloat = 4
  var tint: Color = .secondary

  var body: some View {
    TimelineView(.periodic(from: .now, by: 0.32)) { context in
      let step = Int(context.date.timeIntervalSinceReferenceDate / 0.32) % 3
      HStack(spacing: size * 0.7) {
        ForEach(0..<3, id: \.self) { index in
          Circle()
            .fill(tint)
            .opacity(index == step ? 1 : 0.25)
            .frame(width: size, height: size)
        }
      }
    }
    .frame(width: size * 3 + size * 1.4, alignment: .leading)
  }
}
