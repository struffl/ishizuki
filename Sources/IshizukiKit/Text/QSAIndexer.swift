// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
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

/// The indexer as a layer holds it: one small projection that reads the same activation the
/// attention does, and produces the mask saying which keys each query may see.
///
/// It is a separate head geometry from the attention's — its own width, its own count, its own
/// pair of norms — and it keeps its own keys, unnormalised and unrotated, because a block's key
/// is the mean of its tokens' and a mean of rotated keys is not the rotation of their mean.
public struct QSAIndexer: @unchecked Sendable {
  let qkProj: any Projection
  let qNorm: MLXArray
  let kNorm: MLXArray
  let rope: RotaryEmbedding
  let heads: Int
  let kvHeads: Int
  let headDim: Int
  let selector: QSASelector
  let eps: Float

  public init(
    config: BonsaiConfig.TextConfig, module: String, factory: PackedModuleFactory,
    store: WeightStore, rope: RotaryEmbedding
  ) throws {
    let prefix = factory.tensorPrefix + module + ".indexer"
    guard let heads = config.indexerNumHeads, let headDim = config.indexerHeadDim,
      let budget = config.indexerBudget, let ratio = config.indexerCompressRatio
    else {
      throw BonsaiError.unsupportedModel("\(module) indexes attention but says nothing of how")
    }
    self.qkProj = try factory.projection(module + ".indexer.index_qk_proj")
    self.qNorm = try store(prefix + ".q_layernorm.weight")
    self.kNorm = try store(prefix + ".k_layernorm.weight")
    self.rope = rope
    self.heads = heads
    self.kvHeads = config.indexerKVHeads ?? 1
    self.headDim = headDim
    self.selector = QSASelector(headDim: headDim, compressRatio: ratio, budget: budget)
    self.eps = config.rmsNormEps
  }

  /// Whether a context this long can spend the whole budget, in which case every block is
  /// selected and the mask would say nothing the causal one does not. The check is worth making
  /// — at the budgets these models ship, it is the only case that ever arises.
  public func selectsEverything(upTo length: Int) -> Bool {
    length / selector.compressRatio <= selector.blockTopK
  }

  /// One row per query, holding the key positions it may attend to. Nil when the budget
  /// reaches the whole context and the ordinary causal mask already says it.
  public func callAsFunction(
    _ x: MLXArray, cache: AttentionKVCache?, offset: Int
  ) -> MLXArray? {
    let length = x.dim(1)
    let seen = offset + length

    // The keys are recorded whatever the answer turns out to be. A short context selects
    // everything and needs no mask, but the tokens it saw are still what a later step pools —
    // skipping the record here is a decode that indexes against the current token alone.
    let projected = qkProj(x)
    let queryWidth = heads * headDim
    var queries = projected[.ellipsis, 0..<queryWidth]
      .reshaped([x.dim(0), length, heads, headDim])
    let fresh = projected[.ellipsis, queryWidth..<(queryWidth + kvHeads * headDim)]
      .reshaped([x.dim(0), length, headDim])

    var keys = fresh
    if var cache { keys = cache.appendIndexerKeys(fresh, upTo: seen) }
    eval(keys)
    guard !selectsEverything(upTo: seen) else { return nil }

    queries = MLXFast.rmsNorm(queries, weight: qNorm.asType(queries.dtype), eps: eps)
    queries = rope(queries.transposed(0, 2, 1, 3), positions: Self.axes(Array(offset..<seen)))

    // The selection is per query and the ranking is small, so it is done a query at a time and
    // the answer is a list of positions rather than a tensor of them.
    var rows: [[Int]] = []
    rows.reserveCapacity(length)
    let flatKeys = keys[0]
    for query in 0..<length {
      let visible = offset + query + 1
      rows.append(
        select(queries: queries[0, 0..., query, 0...], keys: flatKeys, visible: visible))
    }
    return mask(rows: rows, keyLength: seen, dtype: x.dtype)
  }

  /// Which positions one query keeps. The pooled block keys are normalised and rotated to the
  /// position each block begins at, which is where its first token sat.
  func select(queries: MLXArray, keys: MLXArray, visible: Int) -> [Int] {
    let blocks = visible / selector.compressRatio
    guard blocks > 0 else { return Array(0..<visible) }

    var pooled = selector.pooled(keys[0..<visible, 0...])
    pooled = MLXFast.rmsNorm(pooled, weight: kNorm.asType(pooled.dtype), eps: eps)
    let starts = (0..<blocks).map { $0 * selector.compressRatio }
    pooled = rope(pooled.reshaped([1, 1, blocks, headDim]), positions: Self.axes(starts))
      .reshaped([blocks, headDim])

    let chosen = selector.chooseBlocks(selector.scores(queries: queries, blockKeys: pooled))
    return selector.tokens(blocks: chosen, visible: visible)
  }

  /// Rope here is the model's own, which is an mrope: it wants a position per axis, and text
  /// puts the same one on all three.
  static func axes(_ positions: [Int]) -> MLXArray {
    let row = positions.map { Int32($0) }
    return MLXArray(row + row + row, [3, positions.count])
  }

  /// The rows turned into something attention can add to its causal mask: zero where a key was
  /// selected, and the floor of the dtype where it was not.
  func mask(rows: [[Int]], keyLength: Int, dtype: DType) -> MLXArray {
    var flat = [Bool](repeating: false, count: rows.count * keyLength)
    for (query, kept) in rows.enumerated() {
      let base = query * keyLength
      for key in kept where key < keyLength { flat[base + key] = true }
    }
    let keep = MLXArray(flat, [1, 1, rows.count, keyLength])
    return MLX.where(keep, MLXArray(Float(0)), MLXArray(-Float.greatestFiniteMagnitude))
      .asType(dtype)
  }
}
