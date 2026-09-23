// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The surfaces the panels are cut from: the window's own background, grouped sections the way
// Settings draws them, and glass left to the controls the system gives it to.

import AppKit
import SwiftUI

extension Color {
  init(light: (Double, Double, Double), dark: (Double, Double, Double)) {
    self = Color(
      NSColor(name: nil) { appearance in
        let (r, g, b) = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? dark : light
        return NSColor(red: r, green: g, blue: b, alpha: 1)
      })
  }

  /// The window: parchment, or a warm charcoal after dark.
  static let paper = Color(light: (0.957, 0.937, 0.894), dark: (0.114, 0.106, 0.094))
  /// The sidebar, a sheet further down the stack.
  static let paperDeep = Color(light: (0.925, 0.898, 0.843), dark: (0.086, 0.080, 0.071))
  /// What a section, a plate or a chip is cut from: a fresher sheet laid on the page.
  static let surface = Color(light: (0.988, 0.976, 0.949), dark: (0.160, 0.149, 0.133))
  /// A pencil line rather than a system separator.
  static let hairline = Color(light: (0.839, 0.800, 0.722), dark: (0.290, 0.271, 0.239))
  /// Ink for text that is not the model's or the person's.
  static let ink = Color(light: (0.200, 0.176, 0.141), dark: (0.906, 0.878, 0.827))

  /// The bonsai's own green: the accent, and the person's bubble.
  static let moss = Color(light: (0.310, 0.431, 0.259), dark: (0.529, 0.667, 0.447))
  /// Fired clay, for what needs a look.
  static let clay = Color(light: (0.639, 0.333, 0.200), dark: (0.851, 0.557, 0.408))

  /// Reading, held apart from writing because the wait for one feels nothing like the wait
  /// for the other: a faded blue-black ink.
  static let reading = Color(light: (0.220, 0.330, 0.470), dark: (0.600, 0.710, 0.820))
  /// Writing: new growth.
  static let generating = Color(light: (0.310, 0.431, 0.259), dark: (0.580, 0.740, 0.490))
  /// The instructions the model is handed every turn: an ochre margin note.
  static let instructing = Color(light: (0.500, 0.350, 0.110), dark: (0.890, 0.740, 0.450))
  /// The person's own bubble. Fixed so white text stays readable on it.
  static let mine = Color(light: (0.290, 0.404, 0.239), dark: (0.290, 0.404, 0.239))

  static let diffAdded = Color(light: (0.290, 0.420, 0.220), dark: (0.580, 0.760, 0.490))
  static let diffRemoved = Color(light: (0.640, 0.250, 0.180), dark: (0.900, 0.560, 0.480))
}

/// A message bubble with a tail on the side it came from. Only the two ends of the
/// conversation get one; a thought or a tool call is not something anybody said.
///
/// Drawn as one continuous outline rather than a rounded rectangle with a tail added to it.
/// The first attempt did the latter, and the body's own corner cut across the tail and left a
/// spur hanging off the bottom of every bubble.
struct Bubble: Shape {
  var mine: Bool
  var radius: CGFloat = 14
  var tail: CGFloat = 7

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
  func bubble(mine: Bool, horizontal: CGFloat = 14, vertical: CGFloat = 9) -> some View {
    let shape = Bubble(mine: mine)
    return padding(.vertical, vertical)
      .padding(mine ? .leading : .trailing, horizontal)
      .padding(mine ? .trailing : .leading, horizontal + shape.tail)
      .background(Color.mine.opacity(mine ? 1 : 0), in: shape)
  }

  /// The model's side of the conversation: the same quiet fill a grouped form row uses.
  func plateBubble(horizontal: CGFloat = 14, vertical: CGFloat = 9) -> some View {
    let shape = Bubble(mine: false)
    return padding(.vertical, vertical)
      .padding(.trailing, horizontal)
      .padding(.leading, horizontal + shape.tail)
      .background(Color.surface, in: shape)
  }

  /// Paper for the window itself.
  func paperBackground() -> some View {
    containerBackground(for: .window) { Color.paper }
      .tint(.moss)
      .fontDesign(.serif)
  }

  /// A sheet laid on the page: fresh paper with a pencil edge.
  func paperCard(radius: CGFloat = Radius.card) -> some View {
    background(Color.surface, in: .rect(cornerRadius: radius))
      .overlay {
        RoundedRectangle(cornerRadius: radius)
          .strokeBorder(Color.hairline, lineWidth: 0.5)
      }
      .shadow(color: Color.ink.opacity(0.05), radius: 1.5, y: 1)
  }

  func textPlate(
    _ shape: some Shape, horizontal: CGFloat = 13, vertical: CGFloat = 9
  ) -> some View {
    padding(.horizontal, horizontal)
      .padding(.vertical, vertical)
      .background(Color.surface, in: shape)
  }

  func textPlate(
    radius: CGFloat = Radius.control, horizontal: CGFloat = 11, vertical: CGFloat = 8
  ) -> some View {
    padding(.horizontal, horizontal)
      .padding(.vertical, vertical)
      .background(Color.surface, in: .rect(cornerRadius: radius))
  }

  /// Something short that sits on content: a row's own header, a chip, the line over the
  /// composer.
  func chipPlate(
    radius: CGFloat = Radius.chip, horizontal: CGFloat = 9, vertical: CGFloat = 2
  ) -> some View {
    padding(.horizontal, horizontal)
      .padding(.vertical, vertical)
      .background(Color.surface, in: .rect(cornerRadius: radius))
  }
}

/// A panel in the content layer, drawn the way a grouped form draws its sections.
struct GlassCard<Content: View>: View {
  var padding: CGFloat = Spacing.m
  var radius: CGFloat = Radius.card
  @ViewBuilder var content: Content

  var body: some View {
    content
      .padding(padding)
      .frame(maxWidth: .infinity, alignment: .leading)
      .paperCard(radius: radius)
  }
}

/// A section's title, set as Settings sets its own.
struct SectionHeader: View {
  let title: String

  var body: some View {
    Text(title)
      .font(.system(size: 17, weight: .semibold, design: .serif))
      .foregroundStyle(Color.ink)
      .padding(.leading, Spacing.xs)
  }
}

struct GlassSection<Content: View>: View {
  let title: String
  @ViewBuilder var content: Content

  var body: some View {
    VStack(alignment: .leading, spacing: Spacing.s) {
      SectionHeader(title: title)
      GlassCard { content }
    }
  }
}
