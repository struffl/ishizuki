// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

public enum ANEOffload: Equatable, Sendable {
  // Time both halves on this machine and split them where they finish together.
  case automatic
  case fraction(Double)

  public enum Refusal: Error, CustomStringConvertible {
    case noSlices
    case cutElsewhere(cut: Double, wanted: Double)

    public var description: String {
      switch self {
      case .noSlices:
        "This pack carries no Neural Engine slices."
      case .cutElsewhere(let cut, let wanted):
        String(
          format: "This pack's slices are cut at %.2f, not %.2f — re-export to change the split",
          cut, wanted)
      }
    }
  }

  /// The channel split is settled when the slices are cut, so the pack decides it and this only
  /// says whether to use them — and refuses a fraction the pack was not cut at.
  public static func apply(_ setting: ANEOffload?, pack: URL) throws {
    BonsaiRuntime.aneOffload = setting
    BonsaiRuntime.aneBank = nil
    guard let setting else { return }

    guard let bank = ANEBank(pack: pack) else { throw Refusal.noSlices }
    if case .fraction(let wanted) = setting, let cut = bank.fraction, abs(cut - wanted) > 0.02 {
      throw Refusal.cutElsewhere(cut: cut, wanted: wanted)
    }
    BonsaiRuntime.aneBank = bank
  }
}

public enum BonsaiRuntime {
  public nonisolated(unsafe) static var useFusedHadamard = false

  public nonisolated(unsafe) static var useQMVWide = false

  public nonisolated(unsafe) static var useVerifyMatmul = true

  /// Routed experts held in memory per sparse layer, when the model keeps them on disk. More
  /// slots means fewer reads and more resident bytes; this is the whole memory dial for a
  /// streamed model.
  public nonisolated(unsafe) static var expertSlots = 16

  /// Rows of the n-gram table a single fetch may ask for. A token costs one row per head, so
  /// this is the prefill chunk times the head count, and the buffer it sizes is small enough
  /// that being generous costs nothing.
  public nonisolated(unsafe) static var engramRows = 16384

  // Share of the wide projections to prefill on the Neural Engine, or nil for none.
  public nonisolated(unsafe) static var aneOffload: ANEOffload?

  // Set once the pack's slices are found, and read by the projections that can be split.
  public nonisolated(unsafe) static var aneBank: ANEBank?

  // What the Neural Engine is asked to hold at once, kept under the ~4.5 GB where it stops
  // holding slices and starts paging them.
  public nonisolated(unsafe) static var aneBudgetBytes = 4 << 30
}
