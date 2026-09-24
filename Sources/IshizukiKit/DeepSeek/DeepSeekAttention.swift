// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// V4.1 attention: a sliding window every layer keeps, and a compressed global KV a few layers own.

import Foundation
import MLX
import MLXNN
import MLXFast

/// What the layers of one forward pass hand down to each other. An index source leaves its
/// top-k for the reuse layers below it; the candidate source leaves its block pool for the
/// reindexers below it. Each is written before anything below reads it.
public final class DeepSeekSelection: @unchecked Sendable {
  var indices: MLXArray?
  var candidates: MLXArray?

  public init() {}
}

/// Pools `ratio` consecutive tokens into one latent through a learned softmax gate, per channel.
/// A group that the chunk leaves unfinished waits in the cache for the tokens that finish it.
struct DeepSeekCompressor: @unchecked Sendable {
  let ratio: Int
  let kv: any Projection
  let gate: (any Projection)?
  let norm: MLXArray
  let eps: Float

  /// The latents of every group this chunk completes, before any rotation.
  func callAsFunction(_ x: MLXArray, cache: DeepSeekLayerCache) -> MLXArray {
    guard ratio > 1, let gate else {
      return MLXFast.rmsNorm(kv(x), weight: norm.asType(x.dtype), eps: eps)
    }
    let wide = x.asType(.float32)
    var rows = kv(wide).asType(.float32)
    var scores = gate(wide).asType(.float32)
    if let pendingKV = cache.pendingKV, let pendingScore = cache.pendingScore {
      rows = concatenated([pendingKV, rows], axis: 1)
      scores = concatenated([pendingScore, scores], axis: 1)
    }
    let b = rows.dim(0)
    let d = rows.dim(2)
    let total = rows.dim(1)
    let whole = total / ratio * ratio
    cache.pendingKV = whole < total ? rows[0..., whole...] : nil
    cache.pendingScore = whole < total ? scores[0..., whole...] : nil
    guard whole > 0 else { return MLXArray.zeros([b, 0, d], dtype: x.dtype) }
    let grouped = rows[0..., ..<whole].reshaped([b, whole / ratio, ratio, d])
    let weights = softmax(
      scores[0..., ..<whole].reshaped([b, whole / ratio, ratio, d]), axis: 2, precise: true)
    let pooled = (grouped * weights).sum(axis: 2).asType(x.dtype)
    return MLXFast.rmsNorm(pooled, weight: norm.asType(pooled.dtype), eps: eps)
  }
}

/// Picks, for every query, the `topK` compressed positions worth attending to.
///
/// A small side attention of its own: fp4 query heads against one key per compressed position,
/// rectified and summed under per-head weights the query also predicts. The candidate source
/// additionally keeps the best blocks of positions as a pool, and the reindexers below it score
/// only inside that pool.
struct DeepSeekIndexer: @unchecked Sendable {
  let queries: any Projection
  let weights: any Projection
  let keyProjection: (any Projection)?
  let keyNorm: MLXArray?
  let heads: Int
  let headDim: Int
  let topK: Int
  let isCandidateSource: Bool
  let usesCandidates: Bool
  let blockSize: Int
  let topBlocks: Int
  let eps: Float

  /// Index keys for newly compressed latents, taken before the latents are rotated.
  func keys(_ latent: MLXArray, rope: DeepSeekRope, table: (cos: MLXArray, sin: MLXArray),
    fakeQuant: Bool
  ) -> MLXArray? {
    guard let keyProjection, let keyNorm else { return nil }
    var k = MLXFast.rmsNorm(keyProjection(latent), weight: keyNorm.asType(latent.dtype), eps: eps)
    k = rope(k, table)
    return fakeQuant ? DeepSeekQuant.fp4(k) : k
  }

  /// Every query's chosen positions, `[b, s, min(topK, count)]`, -1 where fewer are visible.
  /// Scored a block of queries at a time: the scores against every compressed position are the
  /// widest thing a long context builds, and a whole chunk of them at once does not fit.
  func select(
    _ x: MLXArray, qr: MLXArray, rope: DeepSeekRope, table: (cos: MLXArray, sin: MLXArray),
    keys: MLXArray, visible: [Int], selection: DeepSeekSelection, fakeQuant: Bool
  ) -> MLXArray {
    let b = x.dim(0)
    let s = x.dim(1)
    let count = keys.dim(1)
    var q = queries(qr).reshaped([b, s, heads, headDim])
    q = rope(q, table)
    if fakeQuant { q = DeepSeekQuant.fp4(q) }
    let scale = Float(pow(Double(headDim), -0.5) * pow(Double(heads), -0.5))
    let perHead = weights(x).asType(.float32) * scale
    let wideKeys = keys.asType(.float32).expandedDimensions(axis: 1).swappedAxes(-1, -2)
    let positions = MLXArray(0..<Int32(count)).reshaped([1, 1, count])
    let negative = MLXArray(-Float.infinity)
    let k = min(topK, count)

    var chosen: [MLXArray] = []
    var pools: [MLXArray] = []
    var from = 0
    while from < s {
      let to = min(from + DeepSeekAttention.queryBlock, s)
      let dots = matmul(q[0..., from..<to].asType(.float32), wideKeys)
      var scores = (relu(dots) * perHead[0..., from..<to].expandedDimensions(axis: -1)).sum(axis: 2)
      let reach = MLXArray(visible[from..<to].map { Int32($0) }).reshaped([1, to - from, 1])
      scores = MLX.where(positions .< reach, scores, negative)
      if isCandidateSource {
        let pool = candidates(scores, visible: Array(visible[from..<to]))
        pools.append(pool)
      } else if usesCandidates, let pool = selection.candidates {
        scores = MLX.where(pool[0..., from..<to], scores, negative)
      }
      let top = sorted(argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k], axis: -1)
        .asType(.int32)
      let picked = MLX.where(top .< reach, top, MLXArray(Int32(-1)))
      if s > DeepSeekAttention.queryBlock { eval(picked) }
      chosen.append(picked)
      from = to
    }
    if isCandidateSource {
      selection.candidates = pools.count == 1 ? pools[0] : concatenated(pools, axis: 1)
    }
    return chosen.count == 1 ? chosen[0] : concatenated(chosen, axis: 1)
  }

  /// The best `topBlocks` blocks of positions for each query, the block holding its newest
  /// position always among them, as a mask over positions.
  private func candidates(_ scores: MLXArray, visible: [Int]) -> MLXArray {
    let b = scores.dim(0)
    let s = scores.dim(1)
    let count = scores.dim(2)
    let blocks = (count + blockSize - 1) / blockSize
    var padded = scores
    if blocks * blockSize > count {
      padded = concatenated(
        [scores, MLXArray.full([b, s, blocks * blockSize - count], values: MLXArray(-Float.infinity))],
        axis: -1)
    }
    var best = padded.reshaped([b, s, blocks, blockSize]).max(axis: -1)
    let newest = MLXArray(visible.map { Int32($0 > 0 ? ($0 - 1) / blockSize : -1) })
      .reshaped([1, s, 1])
    let index = MLXArray(0..<Int32(blocks)).reshaped([1, 1, blocks])
    best = MLX.where(index .== newest, MLXArray(Float.infinity), best)
    let kept = min(topBlocks, blocks)
    let threshold = sorted(best, axis: -1)[.ellipsis, (blocks - kept)..<(blocks - kept + 1)]
    let keep = (best .>= threshold) .&& (best .> MLXArray(-Float.infinity))
    return repeated(keep, count: blockSize, axis: -1)[.ellipsis, ..<count]
  }
}

public final class DeepSeekAttention: @unchecked Sendable {
  let layer: Int
  let heads: Int
  let headDim: Int
  let window: Int
  let ratio: Int
  let eps: Float
  let scale: Float
  let qA: any Projection
  let qB: any Projection
  let qNorm: MLXArray
  let kvProjection: any Projection
  let kvNorm: MLXArray
  let outA: GroupedProjection
  let outB: any Projection
  let sink: MLXArray
  let rope: DeepSeekRope
  let compressor: DeepSeekCompressor?
  let indexer: DeepSeekIndexer?
  let kvSource: Int?
  var fakeQuant = true

  init(
    layer: Int, prefix: String, config: DeepSeekConfig, weights: DeepSeekWeights,
    rope: DeepSeekRope
  ) throws {
    self.layer = layer
    self.heads = config.heads
    self.headDim = config.headDim
    self.window = config.window
    self.ratio = config.compressRatio(layer: layer)
    self.eps = config.normEps
    self.scale = Float(pow(Double(config.headDim), -0.5))
    self.rope = rope
    self.qA = try weights.linear(prefix + ".wq_a")
    self.qB = try weights.linear(prefix + ".wq_b")
    self.qNorm = try weights.array(prefix + ".q_norm.weight")
    self.kvProjection = try weights.linear(prefix + ".wkv")
    self.kvNorm = try weights.array(prefix + ".kv_norm.weight")
    self.outA = try weights.grouped(prefix + ".wo_a", groups: config.oGroups)
    self.outB = try weights.linear(prefix + ".wo_b")
    self.sink = try weights.float32(prefix + ".attn_sink")

    let backbone = layer < config.layers
    if backbone, ratio > 0, config.kvSourceLayers.contains(layer) {
      self.compressor = DeepSeekCompressor(
        ratio: ratio, kv: try weights.linear(prefix + ".compressor.wkv", wide: ratio > 1),
        gate: ratio > 1 ? try weights.linear(prefix + ".compressor.wgate", wide: true) : nil,
        norm: try weights.array(prefix + ".compressor.norm.weight"), eps: config.normEps)
    } else {
      self.compressor = nil
    }
    if backbone, ratio > 0, config.indexSourceLayers.contains(layer) {
      let owns = config.kvSourceLayers.contains(layer)
      self.indexer = DeepSeekIndexer(
        queries: try weights.linear(prefix + ".indexer.wq_b"),
        weights: try weights.linear(prefix + ".indexer.weights_proj"),
        keyProjection: owns ? try weights.linear(prefix + ".indexer.wk") : nil,
        keyNorm: owns ? try weights.array(prefix + ".indexer.k_norm.weight") : nil,
        heads: config.indexHeads, headDim: config.indexHeadDim, topK: config.indexTopK,
        isCandidateSource: layer == config.candidateSourceLayer,
        usesCandidates: config.candidateSourceLayer >= 0 && layer > config.candidateSourceLayer,
        blockSize: config.candidateBlockSize, topBlocks: config.candidateTopKBlocks,
        eps: config.normEps)
    } else {
      self.indexer = nil
    }
    self.kvSource = backbone && ratio > 0 ? config.kvSource(of: layer) : nil
  }

  /// Queries attended to at once. A prefill chunk is walked in blocks of this many, each with
  /// only the window keys and the gathered latents its own queries can reach, so what a layer
  /// holds grows with the block rather than with the square of the chunk.
  nonisolated(unsafe) static var queryBlock = 128

  /// One block of queries, `q` `[b, n, heads, headDim]` starting `from` positions into the
  /// chunk, against the window rows it can see and the latents its indices chose. The sinks sit
  /// in the softmax's denominator only. Returns `[b, heads, n, headDim]`.
  private func attend(
    _ q: MLXArray, keys: MLXArray, held: Int, from: Int, latents: MLXArray?, indices: MLXArray?
  ) -> MLXArray {
    let b = q.dim(0)
    let n = q.dim(1)
    let first = max(0, held + from - (window - 1))
    let last = held + from + n
    let windowKeys = keys[0..., first..<last]
    let query = MLXArray(0..<Int32(n)).reshaped([n, 1]) + Int32(held + from)
    let position = MLXArray(Int32(first)..<Int32(last)).reshaped([1, last - first])
    let allowed = (position .<= query) .&& (position .> query - Int32(window))
    let mask = MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity))

    var logits = [
      matmul(q.transposed(0, 2, 1, 3), windowKeys.expandedDimensions(axis: 1).swappedAxes(-1, -2))
        * scale + mask
    ]
    var gathered: MLXArray?
    if let latents, let indices, indices.dim(-1) > 0 {
      let k = indices.dim(-1)
      let flat = latents.reshaped([-1, latents.dim(-1)])
      let base = (MLXArray(0..<Int32(b)) * Int32(latents.dim(1))).reshaped([b, 1, 1])
      let rows = take(flat, (maximum(indices, 0) + base).reshaped([-1]), axis: 0)
        .reshaped([b, n, k, latents.dim(-1)]).asType(q.dtype)
      gathered = rows
      let invalid = MLX.where(indices .< 0, MLXArray(-Float.infinity), MLXArray(Float(0)))
      logits.append(
        (matmul(q, rows.swappedAxes(-1, -2)) * scale).transposed(0, 2, 1, 3)
          + invalid.expandedDimensions(axis: 1))
    }
    let sinks = broadcast(
      sink.reshaped([1, heads, 1, 1]).asType(logits[0].dtype), to: [b, heads, n, 1])
    let probabilities = softmax(concatenated(logits + [sinks], axis: -1), axis: -1, precise: true)
    let windowed = windowKeys.dim(1)
    var out = matmul(
      probabilities[.ellipsis, ..<windowed],
      windowKeys.expandedDimensions(axis: 1).asType(probabilities.dtype))
    if let gathered {
      let k = gathered.dim(2)
      let share = probabilities[.ellipsis, windowed..<(windowed + k)].transposed(0, 2, 1, 3)
      out = out + matmul(share, gathered.asType(share.dtype)).transposed(0, 2, 1, 3)
    }
    return out
  }

  /// Only the global KV a source layer owes the layers after it, for a stretch of prompt the
  /// layer itself is not run over: the compressor's latents and their index keys. The window is
  /// left empty — the replay that follows starts it again — and the offset moves on.
  func compressOnly(_ x: MLXArray, cache: DeepSeekLayerCache) {
    if let compressor {
      let latent = compressor(x, cache: cache)
      let count = latent.dim(1)
      if count > 0 {
        let groups = rope.table(from: cache.compressedCount * ratio, count: count, stride: ratio)
        let indexKeys = indexer?.keys(latent, rope: rope, table: groups, fakeQuant: fakeQuant)
        var roped = rope(latent, groups)
        if fakeQuant { roped = DeepSeekQuant.fp4E4M3(roped) }
        cache.append(latents: roped, keys: indexKeys)
      }
    }
    cache.window = nil
    cache.advance(x.dim(1))
  }

  /// The window keys a draft layer holds: the backbone's own positions, projected through this
  /// layer from what the backbone's last layers read, and nothing of the drafts themselves.
  func observe(_ main: MLXArray, cache: DeepSeekLayerCache, start: Int) {
    let table = rope.table(from: start, count: main.dim(1))
    var kv = rope(
      MLXFast.rmsNorm(kvProjection(main), weight: kvNorm.asType(main.dtype), eps: eps), table)
    if fakeQuant { kv = DeepSeekQuant.fp8(kv) }
    let held = cache.window.map { concatenated([$0.asType(kv.dtype), kv], axis: 1) } ?? kv
    cache.window = held[0..., max(0, held.dim(1) - window)...]
    cache.advance(main.dim(1))
  }

  /// A draft block at `start`: every position sees the whole window and every other draft
  /// position, both ways, which is what lets one pass propose them all at once.
  func draft(_ x: MLXArray, cache: DeepSeekLayerCache, start: Int) -> MLXArray {
    let b = x.dim(0)
    let s = x.dim(1)
    let table = rope.table(from: start, count: s)
    let qr = MLXFast.rmsNorm(qA(x), weight: qNorm.asType(x.dtype), eps: eps)
    let q = rope(qB(qr).reshaped([b, s, heads, headDim]), table)
    var kv = rope(MLXFast.rmsNorm(kvProjection(x), weight: kvNorm.asType(x.dtype), eps: eps), table)
    if fakeQuant { kv = DeepSeekQuant.fp8(kv) }
    let keys = cache.window.map { concatenated([$0.asType(kv.dtype), kv], axis: 1) } ?? kv
    let logits = matmul(q.transposed(0, 2, 1, 3), keys.expandedDimensions(axis: 1).swappedAxes(-1, -2))
      * scale
    let sinks = broadcast(sink.reshaped([1, heads, 1, 1]).asType(logits.dtype), to: [b, heads, s, 1])
    let probabilities = softmax(concatenated([logits, sinks], axis: -1), axis: -1, precise: true)
    let out = matmul(
      probabilities[.ellipsis, ..<keys.dim(1)],
      keys.expandedDimensions(axis: 1).asType(probabilities.dtype))
    let attended = rope(out.transposed(0, 2, 1, 3).asType(x.dtype), table, inverse: true)
    let grouped = outA(attended.reshaped([b * s, outA.groups, -1]))
    return outB(grouped.reshaped([b, s, -1]))
  }

  func callAsFunction(
    _ x: MLXArray, caches: [DeepSeekLayerCache], selection: DeepSeekSelection, start: Int
  ) -> MLXArray {
    let b = x.dim(0)
    let s = x.dim(1)
    let cache = caches[layer]
    let table = rope.table(from: start, count: s)

    let qr = MLXFast.rmsNorm(qA(x), weight: qNorm.asType(x.dtype), eps: eps)
    let q = rope(qB(qr).reshaped([b, s, heads, headDim]), table)
    var kv = rope(MLXFast.rmsNorm(kvProjection(x), weight: kvNorm.asType(x.dtype), eps: eps), table)
    if fakeQuant { kv = DeepSeekQuant.fp8(kv) }

    let previous = cache.window?.asType(kv.dtype)
    let keys = previous.map { concatenated([$0, kv], axis: 1) } ?? kv
    let held = previous?.dim(1) ?? 0
    let keep = max(window - 1, 0)
    cache.window = keep > 0 ? keys[0..., max(0, held + s - keep)...] : nil

    var latents: MLXArray?
    var indices: MLXArray?
    if ratio > 0, let kvSource {
      if let compressor {
        let latent = compressor(x, cache: cache)
        let count = latent.dim(1)
        if count > 0 {
          let groups = rope.table(from: cache.compressedCount * ratio, count: count, stride: ratio)
          let indexKeys = indexer?.keys(latent, rope: rope, table: groups, fakeQuant: fakeQuant)
          var roped = rope(latent, groups)
          if fakeQuant { roped = DeepSeekQuant.fp4E4M3(roped) }
          cache.append(latents: roped, keys: indexKeys)
        }
      }
      let source = caches[kvSource]
      let visible = (0..<s).map { (start + $0 + 1) / ratio }
      if source.compressedCount > 0, let owned = source.latents {
        if let indexer, let keys = source.keys {
          indices = indexer.select(
            x, qr: qr, rope: rope, table: table, keys: keys, visible: visible,
            selection: selection, fakeQuant: fakeQuant)
          selection.indices = indices
        } else {
          indices = selection.indices
        }
        latents = owned
      } else if indexer != nil {
        selection.indices = MLXArray.zeros([b, s, 0], dtype: .int32)
      }
    }

    var pieces: [MLXArray] = []
    var from = 0
    while from < s {
      let to = min(from + Self.queryBlock, s)
      pieces.append(
        attend(q[0..., from..<to], keys: keys, held: held, from: from, latents: latents,
          indices: indices?[0..., from..<to]))
      from = to
    }
    let out = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 2)
    let attended = rope(out.transposed(0, 2, 1, 3).asType(x.dtype), table, inverse: true)
    let grouped = outA(attended.reshaped([b * s, outA.groups, -1]))
    cache.advance(s)
    return outB(grouped.reshaped([b, s, -1]))
  }
}
