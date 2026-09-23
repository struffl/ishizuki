// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The smallest a control is allowed to be, and the modifier that enforces it. Both apps draw the
// same inline chevrons, so both read this.

import SwiftUI

/// The sizes the platform asks for, kept in one place so no control drifts under them.
enum Metrics {
  /// The floor for anything clickable, and what the inline chevrons and dismissers use.
  static let hit: CGFloat = 24
}

/// The gaps between things, on one scale.
enum Spacing {
  static let xs: CGFloat = 4.5
  static let s: CGFloat = 9
  static let m: CGFloat = 13
  static let l: CGFloat = 18
  static let xl: CGFloat = 26
}

extension Font {
  /// Body text, a step above the platform's 13 points.
  static let base = Font.system(size: 14.5)
}

enum Radius {
  static let chip: CGFloat = 7
  static let control: CGFloat = 9
  static let card: CGFloat = 11
}

extension View {
  /// A hit region no smaller than the platform asks for, whatever the glyph inside measures.
  func hitTarget(_ side: CGFloat = Metrics.hit) -> some View {
    frame(minWidth: side, minHeight: side)
      .contentShape(.rect)
  }
}
