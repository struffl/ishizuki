// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The smallest a control is allowed to be, and the modifier that enforces it. Both apps draw the
// same inline chevrons, so both read this.

import SwiftUI

/// The sizes the platform asks for, kept in one place so no control drifts under them.
enum Metrics {
  /// The floor for anything clickable. The platform's own default is 28; 22 is the smallest
  /// the guidelines allow room for, and what the inline chevrons and dismissers use.
  static let hit: CGFloat = 22
}

extension View {
  /// A hit region no smaller than the platform asks for, whatever the glyph inside measures.
  func hitTarget(_ side: CGFloat = Metrics.hit) -> some View {
    frame(minWidth: side, minHeight: side)
      .contentShape(.rect)
  }
}
