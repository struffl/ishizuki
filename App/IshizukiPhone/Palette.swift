// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The phone's colours, named exactly as the Mac names them, so a view that moves between the two
// keeps its meaning. Worth folding into one cross-platform palette once the Mac's own settles.

import SwiftUI
import UIKit

extension Color {
  /// Reading, held apart from writing because the wait for one feels nothing like the wait for
  /// the other.
  static let reading = Color(
    light: Color(red: 0.16, green: 0.37, blue: 0.68),
    dark: Color(red: 0.53, green: 0.73, blue: 0.96))

  /// Writing.
  static let generating = Color(
    light: Color(red: 0.11, green: 0.44, blue: 0.24),
    dark: Color(red: 0.45, green: 0.85, blue: 0.58))

  /// The instructions the model is handed every turn.
  static let instructing = Color(
    light: Color(red: 0.46, green: 0.29, blue: 0.0),
    dark: Color(red: 1.0, green: 0.82, blue: 0.40))

  /// The person's own bubble.
  static let mine = Color(
    light: Color(red: 0.04, green: 0.42, blue: 0.94),
    dark: Color(red: 0.13, green: 0.47, blue: 0.96))

  static let hairline = Color(uiColor: .separator)

  static let diffAdded = Color(
    light: Color(red: 0.11, green: 0.44, blue: 0.24),
    dark: Color(red: 0.45, green: 0.85, blue: 0.58))
  static let diffRemoved = Color(
    light: Color(red: 0.75, green: 0.20, blue: 0.18),
    dark: Color(red: 0.98, green: 0.55, blue: 0.53))

  init(light: Color, dark: Color) {
    self = Color(
      UIColor { traits in
        traits.userInterfaceStyle == .dark ? UIColor(dark) : UIColor(light)
      })
  }
}
