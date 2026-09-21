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

extension Color {
  /// The instructions the model is handed every turn, which is most of what it reads before it
  /// can say anything.
  static let instructing = Color(
    NSColor(name: nil) { appearance in
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        ? NSColor(red: 0.91, green: 0.74, blue: 0.35, alpha: 1)
        : NSColor(red: 0.62, green: 0.45, blue: 0.05, alpha: 1)
    })
}

extension Color {
  /// The person's own bubble. Fixed rather than the system accent, because white text has to
  /// stay readable on it whatever accent someone has chosen.
  static let mine = Color(
    NSColor(name: nil) { appearance in
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        ? NSColor(red: 0.13, green: 0.47, blue: 0.96, alpha: 1)
        : NSColor(red: 0.04, green: 0.42, blue: 0.94, alpha: 1)
    })
}

/// A message bubble with a tail on the side it came from. Only the two ends of the
/// conversation get one; a thought or a tool call is not something anybody said.
///
/// Drawn as one continuous outline rather than a rounded rectangle with a tail added to it.
/// The first attempt did the latter, and the body's own corner cut across the tail and left a
/// spur hanging off the bottom of every bubble.
struct Bubble: Shape {
  var mine: Bool
  var radius: CGFloat = 13
  var tail: CGFloat = 6

  func path(in rect: CGRect) -> Path {
    let r = min(radius, min(rect.width, rect.height) / 2)
    let body = CGRect(
      x: rect.minX, y: rect.minY, width: max(0, rect.width - tail), height: rect.height)
    let (left, top, right, bottom) = (body.minX, body.minY, body.maxX, body.maxY)

    var path = Path()
    path.move(to: CGPoint(x: left + r, y: top))
    path.addLine(to: CGPoint(x: right - r, y: top))
    path.addQuadCurve(
      to: CGPoint(x: right, y: top + r), control: CGPoint(x: right, y: top))
    path.addLine(to: CGPoint(x: right, y: bottom - r * 0.85))
    // Out to the tip and back in along the bottom edge, so the tail is part of the outline.
    path.addQuadCurve(
      to: CGPoint(x: right + tail, y: bottom),
      control: CGPoint(x: right, y: bottom - r * 0.15))
    path.addQuadCurve(
      to: CGPoint(x: right - r * 0.85, y: bottom),
      control: CGPoint(x: right - r * 0.2, y: bottom))
    path.addLine(to: CGPoint(x: left + r, y: bottom))
    path.addQuadCurve(
      to: CGPoint(x: left, y: bottom - r), control: CGPoint(x: left, y: bottom))
    path.addLine(to: CGPoint(x: left, y: top + r))
    path.addQuadCurve(to: CGPoint(x: left + r, y: top), control: CGPoint(x: left, y: top))
    path.closeSubpath()

    guard !mine else { return path }
    // The model's side is the same shape seen in a mirror.
    return path.applying(
      CGAffineTransform(translationX: rect.width, y: 0).scaledBy(x: -1, y: 1))
  }
}

extension View {
  /// A plate cut to a shape of its own, for the bubbles that are not rectangles.
  func textPlate(
    _ shape: some Shape, horizontal: CGFloat = 12, vertical: CGFloat = 8
  ) -> some View {
    padding(.horizontal, horizontal)
      .padding(.vertical, vertical)
      .glassEffect(.clear, in: shape)
  }

  /// The window lays down a material, which is what makes the panels on it read as glass.
  func windowBackdrop() -> some View {
    containerBackground(.ultraThinMaterial, for: .window)
  }

  /// Anything carrying text sits on this: clear glass, since the window's material is already
  /// doing the work of separating it from whatever is behind the window.
  func textPlate(
    radius: CGFloat = 10, horizontal: CGFloat = 10, vertical: CGFloat = 7
  ) -> some View {
    padding(.horizontal, horizontal)
      .padding(.vertical, vertical)
      .glassEffect(.clear, in: .rect(cornerRadius: radius))
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
          .strokeBorder(.white.opacity(focused ? 0.18 : 0.06), lineWidth: 1)
      }
      .shadow(
        color: .black.opacity(focused ? 0.16 : 0.05),
        radius: focused ? 10 : 4,
        y: focused ? 3 : 1
      )
      .opacity(focused ? 1 : 0.82)
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
