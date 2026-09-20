// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The status bar mark: a tree over rock, drawn to stay legible at 18 points.

import AppKit
import SwiftUI

struct BonsaiGlyph: View {
  var body: some View {
    Image(nsImage: BonsaiGlyph.image)
  }

  /// A template image rather than a drawn view: the status bar tints these itself, and it
  /// renders SwiftUI canvases in a label inconsistently.
  static let image: NSImage = {
    let side = 18.0
    let image = NSImage(size: NSSize(width: side, height: side), flipped: true) { _ in
      let unit = side / 18
      func point(_ x: Double, _ y: Double) -> NSPoint {
        NSPoint(x: x * unit, y: y * unit)
      }

      NSColor.black.setStroke()
      NSColor.black.setFill()

      let trunk = NSBezierPath()
      trunk.move(to: point(9, 13.2))
      trunk.curve(to: point(9.6, 7.6), controlPoint1: point(9.3, 11.2), controlPoint2: point(7.7, 9.6))
      trunk.lineWidth = unit * 1.3
      trunk.lineCapStyle = .round
      trunk.stroke()

      let branch = NSBezierPath()
      branch.move(to: point(9.0, 10.4))
      branch.curve(to: point(5.6, 8.9), controlPoint1: point(7.8, 10.5), controlPoint2: point(6.4, 9.9))
      branch.lineWidth = unit
      branch.lineCapStyle = .round
      branch.stroke()

      for pad in [
        NSRect(x: 5.9 * unit, y: 4.0 * unit, width: 7.8 * unit, height: 4.4 * unit),
        NSRect(x: 2.6 * unit, y: 6.2 * unit, width: 5.4 * unit, height: 3.4 * unit),
      ] {
        NSBezierPath(ovalIn: pad).fill()
      }

      let pot = NSBezierPath()
      pot.move(to: point(4.4, 13.0))
      pot.line(to: point(13.6, 13.0))
      pot.line(to: point(12.0, 16.6))
      pot.line(to: point(6.0, 16.6))
      pot.close()
      pot.fill()

      return true
    }
    image.isTemplate = true
    return image
  }()
}
