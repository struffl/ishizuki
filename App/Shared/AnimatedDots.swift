// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The one waiting indicator both apps draw, kept where both can reach it.

import SwiftUI

/// Three dots that pulse in place. Fixed width on purpose: a label that grows and shrinks with
/// its own ellipsis drags whatever sits beside it back and forth. Decorative, so it is hidden
/// from assistive technologies and holds still when motion is turned down.
struct AnimatedDots: View {
  var size: CGFloat = 4
  var tint: Color = .secondary

  @Environment(\.accessibilityReduceMotion) private var reduceMotion

  var body: some View {
    Group {
      if reduceMotion {
        dots(lit: nil)
      } else {
        TimelineView(.periodic(from: .now, by: 0.32)) { context in
          dots(lit: Int(context.date.timeIntervalSinceReferenceDate / 0.32) % 3)
        }
      }
    }
    .frame(width: size * 3 + size * 1.4, alignment: .leading)
    .accessibilityHidden(true)
  }

  private func dots(lit: Int?) -> some View {
    HStack(spacing: size * 0.7) {
      ForEach(0..<3, id: \.self) { index in
        Circle()
          .fill(tint)
          .opacity(lit == nil ? 0.6 : (index == lit ? 1 : 0.25))
          .frame(width: size, height: size)
      }
    }
  }
}
