// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

/// What a harness throws when the thing it was checking did not hold.
public struct BenchFailure: Error, CustomStringConvertible {
  public let reason: String

  public init(reason: String) { self.reason = reason }

  public var description: String { reason }
}
