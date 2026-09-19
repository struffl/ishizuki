// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

/// Spends a bits-per-weight budget across the model's modules.
///
/// Everything starts at the profile's base width. Lifting a module costs bytes and buys back
/// some of the error it was measured to suffer, so the allocator repeatedly takes whichever
/// lift buys the most error per byte, until the budget is spent. A module can be lifted more
/// than once, each step judged on its own merits, which is what lets a badly behaved
/// projection climb two widths while its neighbours stay at the base.
public struct BitAllocator {
  public struct Result: Sendable {
    public let bits: [String: Int]
    public let achievedBpw: Double
    public let boosted: Int
    public let total: Int

    public var histogram: [Int: Int] {
      bits.values.reduce(into: [:]) { $0[$1, default: 0] += 1 }
    }
  }

  public let profile: QuantProfile

  /// Below this width the vocabulary embedding and the head that reads back from it stop
  /// degrading gracefully and start producing garbage completions. The greedy auction below
  /// judges every module purely by error removed per byte spent, and that measure is
  /// deliberately size-invariant — which means these two, being a large fraction of the whole
  /// model's parameters, can lose every round to a crowd of cheaper wins and never get lifted
  /// at all. Pinning them sidesteps the auction instead of hoping it favours them.
  private static let pinnedMinimumBits = 4

  private static func isPinned(_ path: String) -> Bool {
    path.hasSuffix("embed_tokens.weight") || path.hasSuffix("lm_head.weight")
  }

  public init(profile: QuantProfile) {
    self.profile = profile
  }

  public func allocate(_ measurements: [ModuleMeasurement]) -> Result {
    let groupSize = profile.groupSize
    let widths = ([profile.baseBits] + profile.boostBits).sorted()

    var bits: [String: Int] = [:]
    var elements: [String: Int] = [:]
    var measured: [String: ModuleMeasurement] = [:]
    var totalElements = 0
    var totalBytes = 0.0

    for measurement in measurements {
      let floor =
        Self.isPinned(measurement.path)
        ? max(profile.baseBits, Self.pinnedMinimumBits) : profile.baseBits
      bits[measurement.path] = floor
      elements[measurement.path] = measurement.elements
      measured[measurement.path] = measurement
      totalElements += measurement.elements
      totalBytes += measurement.bytes(bits: floor, groupSize: groupSize)
    }
    guard totalElements > 0 else {
      return Result(bits: [:], achievedBpw: 0, boosted: 0, total: 0)
    }

    let budgetBytes = Double(totalElements) * profile.targetBpw / 8

    /// How much error a lift removes, per byte it costs. Larger is a better buy.
    func value(_ path: String, from: Int, to: Int) -> Double? {
      guard let measurement = measured[path] else { return nil }
      let cost =
        measurement.bytes(bits: to, groupSize: groupSize)
        - measurement.bytes(bits: from, groupSize: groupSize)
      guard cost > 0 else { return nil }
      // Scaling the gain by size is what makes this a per-byte figure: a module twice as big
      // removes twice as much error for twice the bytes, so size cancels and the comparison is
      // purely how steeply a module improves for what it costs.
      let gain =
        (measurement.error(from) - measurement.error(to)) * Double(measurement.elements)
      guard gain > 0 else { return nil }
      return gain / cost
    }

    // A heap would be tidier, but the candidate list is one entry per module and is rebuilt
    // only for the module that just moved.
    var candidates: [(path: String, next: Int, value: Double)] = []
    for path in bits.keys {
      let current = bits[path] ?? profile.baseBits
      guard let next = widths.first(where: { $0 > current }),
        let score = value(path, from: current, to: next)
      else { continue }
      candidates.append((path, next, score))
    }

    var boosted: Set<String> = []
    while totalBytes < budgetBytes, !candidates.isEmpty {
      guard
        let bestIndex = candidates.indices.max(by: { candidates[$0].value < candidates[$1].value })
      else { break }
      let best = candidates[bestIndex]
      let current = bits[best.path] ?? profile.baseBits
      guard let measurement = measured[best.path] else {
        candidates.remove(at: bestIndex)
        continue
      }

      let cost =
        measurement.bytes(bits: best.next, groupSize: groupSize)
        - measurement.bytes(bits: current, groupSize: groupSize)
      // Stop rather than overspend: the target is a ceiling, not something to land near.
      if totalBytes + cost > budgetBytes {
        candidates.remove(at: bestIndex)
        continue
      }

      bits[best.path] = best.next
      boosted.insert(best.path)
      totalBytes += cost

      if let further = widths.first(where: { $0 > best.next }),
        let score = value(best.path, from: best.next, to: further)
      {
        candidates[bestIndex] = (best.path, further, score)
      } else {
        candidates.remove(at: bestIndex)
      }
    }

    return Result(
      bits: bits,
      achievedBpw: totalBytes * 8 / Double(totalElements),
      boosted: boosted.count,
      total: bits.count)
  }
}
