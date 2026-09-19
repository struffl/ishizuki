// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation

/// A size ishizuki is willing to stand behind.
///
/// Named for what it costs rather than for a bit width, because no module in these packs is
/// actually at the nominal width: a base width carries most of the model and a measured few
/// are lifted off it. The name that matters is the average.
public struct QuantProfile: Sendable, Equatable {
  public let name: String
  public let baseBits: Int
  /// Widths the allocator may lift a module to, cheapest first.
  public let boostBits: [Int]
  public let targetBpw: Double
  public let groupSize: Int
  public let summary: String

  public init(
    name: String, baseBits: Int, boostBits: [Int], targetBpw: Double,
    groupSize: Int = 64, summary: String
  ) {
    self.name = name
    self.baseBits = baseBits
    self.boostBits = boostBits
    self.targetBpw = targetBpw
    self.groupSize = groupSize
    self.summary = summary
  }

  public static let tiny = QuantProfile(
    name: "tiny", baseBits: 2, boostBits: [3, 4], targetBpw: 2.9,
    summary: "smallest that still keeps vision and the draft head")

  public static let small = QuantProfile(
    name: "small", baseBits: 2, boostBits: [3, 4, 5], targetBpw: 3.3,
    summary: "2-bit base, heavily boosted where it counts")

  public static let balanced = QuantProfile(
    name: "balanced", baseBits: 3, boostBits: [4, 5], targetBpw: 3.6,
    summary: "3-bit base with 4/5-bit boosts — the house default")

  public static let quality = QuantProfile(
    name: "quality", baseBits: 3, boostBits: [4, 5], targetBpw: 3.9,
    summary: "3-bit base, loosened target, closest to oQ4e")

  public static let all: [QuantProfile] = [tiny, small, balanced, quality]

  public static func named(_ name: String) -> QuantProfile? {
    all.first { $0.name == name }
  }
}
