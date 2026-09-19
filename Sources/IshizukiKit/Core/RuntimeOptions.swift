// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

public enum ANEOffload: Equatable, Sendable {
  // Time both halves on this machine and split them where they finish together.
  case automatic
  case fraction(Double)
}

public enum BonsaiRuntime {
  public nonisolated(unsafe) static var useFusedHadamard = false

  public nonisolated(unsafe) static var useQMVWide = false

  // Share of the wide projections to prefill on the Neural Engine, or nil for none.
  public nonisolated(unsafe) static var aneOffload: ANEOffload?

  // Set once the pack's slices are found, and read by the projections that can be split.
  public nonisolated(unsafe) static var aneBank: ANEBank?
}
