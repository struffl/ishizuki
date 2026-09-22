// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The glass the panels are cut from: the window is clear, controls are glass, and content
// sits on standard materials — the layering the platform draws everything else with.

import AppKit
import SwiftUI

extension Color {
  /// Reading, held apart from writing because the wait for one feels nothing like the wait
  /// for the other. Fixed rather than the system accent, which is the person's to spend.
  static let reading = Color(
    NSColor(name: nil) { appearance in
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        ? NSColor(red: 0.53, green: 0.73, blue: 0.96, alpha: 1)
        : NSColor(red: 0.16, green: 0.37, blue: 0.68, alpha: 1)
    })

  /// Writing.
  static let generating = Color(
    NSColor(name: nil) { appearance in
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        ? NSColor(red: 0.45, green: 0.85, blue: 0.58, alpha: 1)
        : NSColor(red: 0.11, green: 0.44, blue: 0.24, alpha: 1)
    })

  /// The instructions the model is handed every turn, which is most of what it reads before
  /// it can say anything. Darkened in the light appearance to clear 4.5:1 on a plate.
  static let instructing = Color(
    NSColor(name: nil) { appearance in
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        ? NSColor(red: 1.0, green: 0.82, blue: 0.40, alpha: 1)
        : NSColor(red: 0.46, green: 0.29, blue: 0.0, alpha: 1)
    })

  /// The person's own bubble. Fixed rather than the system accent, because white text has to
  /// stay readable on it whatever accent someone has chosen.
  static let mine = Color(
    NSColor(name: nil) { appearance in
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        ? NSColor(red: 0.13, green: 0.47, blue: 0.96, alpha: 1)
        : NSColor(red: 0.04, green: 0.42, blue: 0.94, alpha: 1)
    })

  /// The system's own separator, for the borders that used to be a guess at white.
  static let hairline = Color(nsColor: .separatorColor)

  /// A line an edit added, and the line it took out: subdued next to a real git diff, since
  /// this one sits on glass rather than a terminal's flat background.
  static let diffAdded = Color(
    NSColor(name: nil) { appearance in
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        ? NSColor(red: 0.45, green: 0.85, blue: 0.58, alpha: 1)
        : NSColor(red: 0.11, green: 0.44, blue: 0.24, alpha: 1)
    })
  static let diffRemoved = Color(
    NSColor(name: nil) { appearance in
      appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        ? NSColor(red: 0.98, green: 0.55, blue: 0.53, alpha: 1)
        : NSColor(red: 0.75, green: 0.20, blue: 0.18, alpha: 1)
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
  /// A bubble's padding plus the tail's own width on the side it hangs off. The shape takes
  /// the tail out of the rect it is given, so without this the text loses exactly as much room
  /// as the tail occupies and its last letter ends up against the edge.
  func bubble(mine: Bool, horizontal: CGFloat = 13, vertical: CGFloat = 8) -> some View {
    let shape = Bubble(mine: mine)
    return padding(.vertical, vertical)
      .padding(mine ? .leading : .trailing, horizontal)
      .padding(mine ? .trailing : .leading, horizontal + shape.tail)
      .background(Color.mine.opacity(mine ? 1 : 0), in: shape)
  }

  /// The model's side of the conversation. Content, so it takes a standard material rather
  /// than glass: glass belongs to the controls that float over the content, not to the
  /// content itself.
  func plateBubble(horizontal: CGFloat = 13, vertical: CGFloat = 8) -> some View {
    let shape = Bubble(mine: false)
    return padding(.vertical, vertical)
      .padding(.trailing, horizontal)
      .padding(.leading, horizontal + shape.tail)
      .background(.thinMaterial, in: shape)
      .gloss(shape)
  }

  /// A plate cut to a shape of its own, for the bubbles that are not rectangles.
  func textPlate(
    _ shape: some Shape, horizontal: CGFloat = 12, vertical: CGFloat = 8
  ) -> some View {
    padding(.horizontal, horizontal)
      .padding(.vertical, vertical)
      .background(.thinMaterial, in: shape)
      .gloss(shape)
  }

  /// The window lays down the backdrop photo, which is what the panels on top read as layered
  /// glass against, rather than the desktop the old flat material used to blur.
  func windowBackdrop() -> some View {
    containerBackground(for: .window) { WindowBackdrop() }
  }

  /// Anything carrying text sits on this. A thin material rather than clear glass, so small
  /// type keeps its contrast against whatever is behind the window.
  func textPlate(
    radius: CGFloat = 10, horizontal: CGFloat = 10, vertical: CGFloat = 7
  ) -> some View {
    padding(.horizontal, horizontal)
      .padding(.vertical, vertical)
      .background(.thinMaterial, in: .rect(cornerRadius: radius))
      .gloss(.rect(cornerRadius: radius))
  }

  /// The sheen that tells the eye "glass" rather than "blurred photo": a highlight pooling at
  /// the top of the shape, and a rim that catches the light the same way a real edge would.
  func gloss(_ shape: some Shape) -> some View {
    overlay {
      shape
        .fill(
          LinearGradient(
            colors: [.white.opacity(0.4), .white.opacity(0)],
            startPoint: .top, endPoint: UnitPoint(x: 0.5, y: 0.65))
        )
        .blendMode(.overlay)
        .allowsHitTesting(false)
    }
    .overlay {
      shape
        .stroke(
          LinearGradient(
            colors: [.white.opacity(0.6), .white.opacity(0.05)],
            startPoint: .top, endPoint: .bottom),
          lineWidth: 1)
        .allowsHitTesting(false)
    }
  }
}

/// A panel in the content layer: a material, a hairline, and nothing that fights the window
/// state the system already draws for us.
struct GlassCard<Content: View>: View {
  var padding: CGFloat = 12
  var radius: CGFloat = 14
  @ViewBuilder var content: Content

  var body: some View {
    content
      .padding(padding)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(.regularMaterial, in: .rect(cornerRadius: radius))
      .overlay {
        RoundedRectangle(cornerRadius: radius)
          .strokeBorder(Color.hairline, lineWidth: 1)
      }
      .gloss(.rect(cornerRadius: radius))
  }
}

struct GlassSection<Content: View>: View {
  let title: String
  @ViewBuilder var content: Content

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text(title)
        .font(.subheadline.weight(.semibold))
        .foregroundStyle(.secondary)
        .padding(.leading, 4)
      GlassCard { content }
    }
  }
}
