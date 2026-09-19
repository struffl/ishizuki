// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// Turning a run of tokens into the rows of an n-gram table.

import Foundation

/// What a token's context hashes to, before any table is consulted.
///
/// One multiplier per position, mixed by exclusive or, and the same mixed value handed to every
/// head of that order — the heads differ only in the prime they fold it by. Two orders share the
/// table, the pairs first and then the triples, which is why a sixteen-head table is eight heads
/// counted twice.
///
/// The arithmetic wraps on purpose: the multipliers are large and odd, and the overflow is what
/// spreads a vocabulary of a quarter of a million across twenty million rows.
public struct NgramHasher: Sendable {
  public let multipliers: [Int]
  public let ngramSize: Int
  public let headsPerNgram: Int
  public let eosTokenId: Int

  public var heads: Int { (ngramSize - 1) * headsPerNgram }

  public init(multipliers: [Int], ngramSize: Int, headsPerNgram: Int, eosTokenId: Int) {
    self.multipliers = multipliers
    self.ngramSize = ngramSize
    self.headsPerNgram = headsPerNgram
    self.eosTokenId = eosTokenId
  }

  /// The history shifted back by `shift`, without reaching over the end of a document. A
  /// position too near the start of its segment reads the separator instead, so the first tokens
  /// of a document are not given a context belonging to the one before it.
  func shifted(_ history: [Int], by shift: Int) -> [Int] {
    guard shift > 0 else { return history }
    var out = [Int](repeating: eosTokenId, count: history.count)
    var previousEos = -1
    var seen = -1
    for position in 0..<history.count {
      let inSegment = position - (previousEos + 1)
      let source = position - shift
      if inSegment >= shift, source >= 0 { out[position] = history[source] }
      if history[position] == eosTokenId { seen = position }
      previousEos = seen
    }
    return out
  }

  /// One row of hashes per position, in head order: every head of the shorter order first.
  /// `history` is the context the model has seen; only the last `count` positions come back.
  public func hashes(_ history: [Int], last count: Int) -> [[Int]] {
    guard count > 0, !history.isEmpty else { return [] }
    let shifts = (0..<ngramSize).map { shifted(history, by: $0) }
    let start = max(0, history.count - count)

    var rows: [[Int]] = []
    rows.reserveCapacity(history.count - start)
    for position in start..<history.count {
      var row: [Int] = []
      row.reserveCapacity(heads)
      for order in 2...ngramSize {
        var mixed = shifts[0][position] &* multipliers[0]
        for place in 1..<order {
          mixed ^= shifts[place][position] &* multipliers[place]
        }
        // Every head of an order sees the same mix; its own prime is what separates them.
        row.append(contentsOf: [Int](repeating: mixed, count: headsPerNgram))
      }
      rows.append(row)
    }
    return rows
  }
}
