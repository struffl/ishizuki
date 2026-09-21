// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Picking the tokens a query may attend to, against values worked out from the reference.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// Sparse attention fails quietly: attend to the wrong two thousand tokens and the model still
/// answers, just worse, and no shape ever disagrees. So the ranking is checked against the
/// reference transcribed independently — pooled means, the relu before the sum across heads, and
/// the tail that is kept without ever being ranked.
@Suite("QSA indexer")
struct QSAIndexerTests {
  private let headDim = 4
  private let compressRatio = 2
  private let budget = 4
  private let visible = 9

  private func selector() -> QSASelector {
    QSASelector(headDim: headDim, compressRatio: compressRatio, budget: budget)
  }

  private func keys() -> MLXArray {
    MLXArray(Golden.keys.flatMap { $0 }, [Golden.keys.count, headDim])
  }

  private func queries() -> MLXArray {
    MLXArray(Golden.q.flatMap { $0 }, [Golden.q.count, headDim])
  }

  @Test("pools each whole block and leaves the tail out of it")
  func pools() {
    let pooled = selector().pooled(keys())
    eval(pooled)

    #expect(pooled.shape == [visible / compressRatio, headDim], "the odd token forms no block")
    for (block, row) in Golden.pooled.enumerated() {
      for (i, want) in row.enumerated() {
        #expect(abs(pooled[block, i].item(Float.self) - want) < 1e-6, "pooled[\(block)][\(i)]")
      }
    }
  }

  /// A head that dislikes a block contributes nothing rather than voting it down, so one
  /// enthusiastic head is enough to keep a block.
  @Test("scores a block by what the heads that want it say")
  func scores() {
    let selector = selector()
    let scores = selector.scores(queries: queries(), blockKeys: selector.pooled(keys()))
    eval(scores)

    #expect(scores.shape == [visible / compressRatio])
    for (block, want) in Golden.scores.enumerated() {
      #expect(abs(scores[block].item(Float.self) - want) < 1e-6, "score \(block)")
    }
    #expect(Golden.scores.contains(0), "a block no head wants should score nothing")
  }

  @Test("spends the budget on the best blocks")
  func chooses() {
    let selector = selector()
    let scores = selector.scores(queries: queries(), blockKeys: selector.pooled(keys()))
    #expect(selector.chooseBlocks(scores).sorted() == Golden.chosen)
  }

  @Test("keeps the tail whether the budget reached it or not")
  func keepsTheTail() {
    let selector = selector()
    #expect(selector.tokens(blocks: Golden.chosen, visible: visible) == Golden.tokens)
    // Token 8 is the tail: it forms no whole block, and it is in the answer regardless.
    #expect(selector.tokens(blocks: [], visible: visible) == [8])
  }

  @Test("selects the same tokens end to end")
  func selects() {
    #expect(selector().select(queries: queries(), keys: keys(), visible: visible) == Golden.tokens)
  }

  /// Under the budget there is nothing to rank, and everything visible should come back.
  @Test("attends to everything when the budget covers it")
  func underBudget() {
    let roomy = QSASelector(headDim: headDim, compressRatio: compressRatio, budget: 64)
    #expect(roomy.select(queries: queries(), keys: keys(), visible: visible) == Array(0..<visible))
    // And a context too short to form one block is all tail.
    #expect(selector().select(queries: queries(), keys: keys(), visible: 1) == [0])
  }

  private enum Golden {
    static let keys: [[Float]] = [
      [0.1875, -0.5625, 0.125, -0.625], [-0.375, 0.3125, -0.4375, 0.25],
      [0.5, -0.25, 0.4375, -0.3125], [-0.0625, 0.625, -0.125, 0.5625],
      [-0.625, 0.0625, -0.6875, 0.0], [0.25, -0.5, 0.1875, -0.5625],
      [-0.3125, 0.375, -0.375, 0.3125], [0.5625, -0.1875, 0.5, -0.25],
      [0.0, 0.6875, -0.0625, 0.625],
    ]
    static let q: [[Float]] = [
      [-0.625, 0.0625, -0.6875, 0.0], [0.25, -0.5, 0.1875, -0.5625],
      [-0.3125, 0.375, -0.375, 0.3125],
    ]
    static let pooled: [[Float]] = [
      [-0.09375, -0.125, -0.15625, -0.1875], [0.21875, 0.1875, 0.15625, 0.125],
      [-0.1875, -0.21875, -0.25, -0.28125], [0.125, 0.09375, 0.0625, 0.03125],
    ]
    static let scores: [Float] = [0.13671875, 0.0, 0.224609375, 0.0]
    static let chosen = [0, 2]
    static let tokens = [0, 1, 4, 5, 8]
  }
}
