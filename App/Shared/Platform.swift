// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The two or three places a view has to know which platform it is drawing on, in one file so the
// views themselves do not.

import SwiftUI

#if os(macOS)
  import AppKit

  typealias PlatformFont = NSFont
#else
  import UIKit

  typealias PlatformFont = UIFont
#endif

enum Clipboard {
  static func copy(_ text: String) {
    #if os(macOS)
      NSPasteboard.general.clearContents()
      NSPasteboard.general.setString(text, forType: .string)
    #else
      UIPasteboard.general.string = text
    #endif
  }
}

extension PlatformFont {
  static func mono(_ size: Double) -> PlatformFont {
    monospacedSystemFont(ofSize: size, weight: .regular)
  }
}
