// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The phone's colours, named exactly as the Mac names them, so a view that moves between the two
// keeps its meaning.

import SwiftUI
import UIKit

extension Color {
  /// The page: parchment, or warm charcoal after dark.
  static let paper = Color(
    light: Color(red: 0.957, green: 0.937, blue: 0.894),
    dark: Color(red: 0.114, green: 0.106, blue: 0.094))

  /// A fresher sheet laid on the page, for a row, a plate or a chip.
  static let surface = Color(
    light: Color(red: 0.988, green: 0.976, blue: 0.949),
    dark: Color(red: 0.16, green: 0.149, blue: 0.133))

  /// A pencil line.
  static let hairline = Color(
    light: Color(red: 0.839, green: 0.8, blue: 0.722),
    dark: Color(red: 0.29, green: 0.271, blue: 0.239))

  /// Text that is not the model's or the person's.
  static let ink = Color(
    light: Color(red: 0.2, green: 0.176, blue: 0.141),
    dark: Color(red: 0.906, green: 0.878, blue: 0.827))

  /// The bonsai's own green: the accent.
  static let moss = Color(
    light: Color(red: 0.31, green: 0.431, blue: 0.259),
    dark: Color(red: 0.529, green: 0.667, blue: 0.447))

  /// Fired clay, for what needs a look.
  static let clay = Color(
    light: Color(red: 0.639, green: 0.333, blue: 0.2),
    dark: Color(red: 0.851, green: 0.557, blue: 0.408))

  /// Reading, held apart from writing: a faded blue-black ink.
  static let reading = Color(
    light: Color(red: 0.22, green: 0.33, blue: 0.47),
    dark: Color(red: 0.6, green: 0.71, blue: 0.82))

  /// Writing: new growth.
  static let generating = Color(
    light: Color(red: 0.31, green: 0.431, blue: 0.259),
    dark: Color(red: 0.58, green: 0.74, blue: 0.49))

  /// The instructions the model is handed every turn.
  static let instructing = Color(
    light: Color(red: 0.5, green: 0.35, blue: 0.11),
    dark: Color(red: 0.89, green: 0.74, blue: 0.45))

  /// The person's own bubble.
  static let mine = Color(
    light: Color(red: 0.29, green: 0.404, blue: 0.239),
    dark: Color(red: 0.29, green: 0.404, blue: 0.239))

  static let diffAdded = Color(
    light: Color(red: 0.29, green: 0.42, blue: 0.22),
    dark: Color(red: 0.58, green: 0.76, blue: 0.49))

  static let diffRemoved = Color(
    light: Color(red: 0.64, green: 0.25, blue: 0.18),
    dark: Color(red: 0.9, green: 0.56, blue: 0.48))

  init(light: Color, dark: Color) {
    self = Color(
      UIColor { traits in
        traits.userInterfaceStyle == .dark ? UIColor(dark) : UIColor(light)
      })
  }
}
