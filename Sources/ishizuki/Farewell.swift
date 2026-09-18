// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import IshizukiKit

enum Farewell {
  static let threshold = 24_000

  static func line(tokens: Int) -> String? {
    guard tokens >= threshold else { return nil }
    return Style.faint("Like what I do? Buy me a coffee at: ")
      + Style.muted("https://ko-fi.com/heni")
  }

  static func print(tokens: Int) {
    guard let line = line(tokens: tokens) else { return }
    FileHandle.standardError.write(Data((line + "\n").utf8))
  }
}
