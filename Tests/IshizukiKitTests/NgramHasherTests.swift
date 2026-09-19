// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// Tokens into table rows, against values worked out from the reference by hand.

import Foundation
import Testing

@testable import IshizukiKit

/// A hash is right or it is noise, and noise reads exactly like a working model that has been
/// handed the wrong rows. So the values here come from the reference transcribed independently,
/// with the wrapping arithmetic kept in sixty-four bits: a history with two separators in it, so
/// the shifts have segments to respect.
@Suite("N-gram hasher")
struct NgramHasherTests {
  private let eos = 9
  private let primes = [1009, 1013, 1019, 1021]

  private func hasher() -> NgramHasher {
    NgramHasher(
      multipliers: Golden.multipliers, ngramSize: 3, headsPerNgram: 2, eosTokenId: eos)
  }

  /// A position too near the start of its document reads the separator, not the tail of the
  /// document before it.
  @Test("shifts without reaching over a separator")
  func shifting() {
    let hasher = hasher()
    #expect(hasher.shifted(Golden.history, by: 0) == Golden.history)
    #expect(hasher.shifted(Golden.history, by: 1) == Golden.shift1)
    #expect(hasher.shifted(Golden.history, by: 2) == Golden.shift2)
  }

  @Test("mixes each order once and hands it to every head of that order")
  func hashes() {
    let rows = hasher().hashes(Golden.history, last: Golden.history.count)
    #expect(rows.count == Golden.history.count)
    #expect(rows[0].count == 4)
    for (position, want) in Golden.hashes.enumerated() {
      #expect(rows[position] == want, "row \(position)")
    }
  }

  /// The heads of one order share a mix and differ only by their prime, which is what makes a
  /// collision in one head independent of the others.
  @Test("heads of an order share a mix, orders do not")
  func headsShareTheirOrder() {
    let rows = hasher().hashes(Golden.history, last: 3)
    for row in rows {
      #expect(row[0] == row[1], "the pair heads should see one mix")
      #expect(row[2] == row[3], "the triple heads should see one mix")
      #expect(row[0] != row[2], "the two orders should not collapse together")
    }
  }

  /// The whole path: tokens to the rows a fetch would ask for.
  @Test("folds each head's mix into that head's block")
  func addresses() throws {
    let layout = EngramLayout.blocked(
      ngramSize: 3, headDim: 4, vocabSizes: primes, parts: 3, dtype: "float32")
    let rows = hasher().hashes(Golden.history, last: Golden.history.count)
    for (position, want) in Golden.addresses.enumerated() {
      let got = (0..<4).map { layout.address(head: $0, hash: rows[position][$0]) }
      #expect(got == want, "position \(position)")
    }
  }

  @Test("hands back only the positions asked for")
  func tail() {
    let all = hasher().hashes(Golden.history, last: Golden.history.count)
    let tail = hasher().hashes(Golden.history, last: 3)
    #expect(tail.count == 3)
    #expect(tail == Array(all.suffix(3)))
    #expect(hasher().hashes(Golden.history, last: 0).isEmpty)
  }

  private enum Golden {
    static let multipliers = [5700357411248241081, 2685821657736338717, -7046029254386353131]
    static let history = [5, 7, 9, 3, 4, 9, 8, 2, 6]
    static let shift1 = [9, 5, 7, 9, 3, 4, 9, 8, 2]
    static let shift2 = [9, 9, 5, 9, 9, 3, 9, 9, 8]
    static let hashes: [[Int]] = [
      [-4251634123836840296, -4251634123836840296, 5407819088512219685, 5407819088512219685],
      [-7809262793317415522, -7809262793317415522, 2048200325650340131, 2048200325650340131],
      [-4390996171359171510, -4390996171359171510, -3169007326850113501, -3169007326850113501],
      [-6762524772196308178, -6762524772196308178, 3302801031023032211, 3302801031023032211],
      [6034593460459893171, 6034593460459893171, -2572565550936543986, -2572565550936543986],
      [5975838563864949749, 5975838563864949749, -8626613546947491894, -8626613546947491894],
      [4010596150823881677, 4010596150823881677, -5162470467269404816, -5162470467269404816],
      [-5469378548021584998, -5469378548021584998, 4317714770994546471, 4317714770994546471],
      [-8057638970490265492, -8057638970490265492, 7031888990752374980, 7031888990752374980],
    ]
    static let addresses: [[Int]] = [
      [684, 1600, 2187, 3135], [669, 1046, 2393, 3610], [175, 1495, 2824, 3860],
      [95, 1755, 2951, 3915], [119, 1205, 2032, 3472], [211, 1040, 2754, 3249],
      [568, 1445, 2420, 3928], [990, 1381, 2518, 3637], [956, 1023, 2800, 3837],
    ]
  }
}
