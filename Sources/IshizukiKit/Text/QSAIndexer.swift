// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// Choosing the few thousand tokens a query is allowed to attend to.

import Foundation
import MLX
import MLXNN

/// Which blocks of keys a query looks at, out of everything it could.
///
/// Attention over a quarter of a million tokens is not affordable and mostly not wanted, so the
/// keys are pooled a few at a time into blocks, each block scored against the query by a small
/// separate head, and only the best of them are attended to. The budget is a token count; the
/// blocks are what the budget is actually spent in.
///
/// The tail — whatever does not fill a whole block — is always kept. It is the most recent
/// context, it is never worth ranking, and dropping it would cost the model the token it just saw.
public struct QSASelector: Sendable {
  public let headDim: Int
  public let compressRatio: Int
  public let budget: Int

  public init(headDim: Int, compressRatio: Int, budget: Int) {
    self.headDim = headDim
    self.compressRatio = compressRatio
    self.budget = budget
  }

  /// How many blocks the budget pays for.
  public var blockTopK: Int { budget / compressRatio }

  /// One key per whole block, the mean of the keys in it. Pooling happens in float32: these are
  /// averaged and then normalised, and doing that at the keys' own width loses the small
  /// differences the ranking is about.
  ///
  /// What comes back is only the mean. The caller normalises it and rotates it to the position
  /// its block starts at before scoring — a block stands where it begins, not where it ends.
  public func pooled(_ keys: MLXArray) -> MLXArray {
    let blocks = keys.dim(-2) / compressRatio
    guard blocks > 0 else { return MLXArray.zeros([0, headDim], dtype: keys.dtype) }
    let whole = keys[..<(blocks * compressRatio), 0...].asType(.float32)
    return whole.reshaped([blocks, compressRatio, headDim]).mean(axis: 1).asType(keys.dtype)
  }

  /// What each block is worth to this query. A head that dislikes a block says nothing rather
  /// than voting against it, so a block is kept when any head wants it.
  public func scores(queries: MLXArray, blockKeys: MLXArray) -> MLXArray {
    let hits = relu(matmul(queries.asType(.float32), blockKeys.asType(.float32).transposed()))
    return hits.sum(axis: -2) / sqrt(Float(headDim))
  }

  /// The blocks worth spending the budget on, best first not guaranteed — a mask does not care.
  public func chooseBlocks(_ scores: MLXArray) -> [Int] {
    let blocks = scores.dim(-1)
    let keep = min(blockTopK, blocks)
    guard keep > 0 else { return [] }
    guard keep < blocks else { return Array(0..<blocks) }
    let ordered = argPartition(scores, kth: blocks - keep, axis: -1)
    return ordered[(blocks - keep)...].asArray(Int32.self).map(Int.init)
  }

  /// The tokens those blocks stand for, plus the tail the budget never had to pay for.
  public func tokens(blocks: [Int], visible: Int) -> [Int] {
    let whole = visible / compressRatio
    var chosen: [Int] = []
    chosen.reserveCapacity(blocks.count * compressRatio + compressRatio)
    for block in blocks.sorted() where block < whole {
      for offset in 0..<compressRatio { chosen.append(block * compressRatio + offset) }
    }
    chosen.append(contentsOf: (whole * compressRatio)..<visible)
    return chosen
  }

  /// Everything a query may attend to: the pooled ranking, spent, and the tail kept.
  public func select(queries: MLXArray, keys: MLXArray, visible: Int) -> [Int] {
    let usable = keys[..<visible, 0...]
    let blocks = visible / compressRatio
    guard blocks > 0 else { return Array(0..<visible) }
    let chosen = chooseBlocks(scores(queries: queries, blockKeys: pooled(usable)))
    return tokens(blocks: chosen, visible: visible)
  }
}
