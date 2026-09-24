// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DeepSeek-V4.1 assembled: embed, widen, forty mixed blocks, collapse, head.

import Foundation
import MLX
import MLXFast

/// One layer: attention then experts, each read out of the residual copies and mixed back in.
///
/// The collapse a sub-block reads with is the one the sub-block before it predicted — the
/// attention uses the previous layer's feed-forward `pre`, the feed-forward uses this layer's
/// attention `pre` — which is what lets the whole mix run in one pass over the streams.
final class DeepSeekBlock: @unchecked Sendable {
  let layer: Int
  let attention: DeepSeekAttention
  let moe: DeepSeekMoE
  let engram: DeepSeekEngramBlock?
  let attentionNorm: MLXArray
  let feedNorm: MLXArray
  let attentionMixer: SinkhornMixer
  let feedMixer: SinkhornMixer
  let eps: Float

  /// `engram` false builds the layer without its n-gram block, for a probe that compares one
  /// block against a reference that runs it without one.
  init(
    layer: Int, prefix: String, config: DeepSeekConfig, weights: DeepSeekWeights,
    rope: DeepSeekRope, engram: Bool = true
  ) throws {
    self.layer = layer
    self.eps = config.normEps
    self.attention = try DeepSeekAttention(
      layer: layer, prefix: prefix + ".attn", config: config, weights: weights, rope: rope)
    self.moe = try DeepSeekMoE(layer: layer, prefix: prefix + ".ffn", config: config, weights: weights)
    if engram, let index = config.engramLayers.firstIndex(of: layer), layer < config.layers {
      self.engram = try DeepSeekEngramBlock(
        index: index, prefix: prefix + ".engram", config: config, weights: weights)
    } else {
      self.engram = nil
    }
    self.attentionNorm = try weights.array(prefix + ".attn_norm.weight")
    self.feedNorm = try weights.array(prefix + ".ffn_norm.weight")
    func mixer(_ name: String) throws -> SinkhornMixer {
      SinkhornMixer(
        projection: try weights.float32(prefix + ".hc_\(name)_fn"),
        scale: try weights.float32(prefix + ".hc_\(name)_scale"),
        base: try weights.float32(prefix + ".hc_\(name)_base"),
        copies: config.hcMult, iterations: config.hcSinkhornIterations, eps: config.hcEps,
        normEps: config.normEps)
    }
    self.attentionMixer = try mixer("attn")
    self.feedMixer = try mixer("ffn")
  }

  /// The part of this layer a prompt's early tokens still need when the layer itself is skipped
  /// for them: its attention input, handed to the compressor.
  func compressOnly(_ streams: MLXArray, pre: MLXArray, cache: DeepSeekLayerCache, compute: DType) {
    var input = SinkhornMixer.collapse(streams, pre: pre).asType(compute)
    input = MLXFast.rmsNorm(input, weight: attentionNorm.asType(compute), eps: eps)
    attention.compressOnly(input, cache: cache)
  }

  /// `draft` runs a DSpark block: `caches` is then just this block's own, and the attention
  /// reads the backbone's window rather than a cache of the block's own positions.
  func callAsFunction(
    _ streams: MLXArray, pre: MLXArray, caches: [DeepSeekLayerCache],
    selection: DeepSeekSelection, start: Int, compute: DType, image: MLXArray? = nil,
    draft: Bool = false
  ) -> (MLXArray, MLXArray) {
    let attentionMix = attentionMixer(streams)
    var input = SinkhornMixer.collapse(streams, pre: pre).asType(compute)
    input = MLXFast.rmsNorm(input, weight: attentionNorm.asType(compute), eps: eps)
    let attended =
      draft
      ? attention.draft(input, cache: caches[0], start: start)
      : attention(input, caches: caches, selection: selection, start: start)
    let mixed = SinkhornMixer.expand(
      attended, residual: streams, post: attentionMix.post, comb: attentionMix.comb)

    let feedMix = feedMixer(mixed)
    var feed = SinkhornMixer.collapse(mixed, pre: attentionMix.pre).asType(compute)
    feed = MLXFast.rmsNorm(feed, weight: feedNorm.asType(compute), eps: eps)
    let answered = moe(feed, image: image)
    let out = SinkhornMixer.expand(
      answered, residual: mixed, post: feedMix.post, comb: feedMix.comb)
    return (out, feedMix.pre)
  }
}

public final class DeepSeekModel: @unchecked Sendable {
  public let config: DeepSeekConfig
  let embedding: MLXArray
  let blocks: [DeepSeekBlock]
  let norm: MLXArray
  let head: any Projection
  public let hasher: DeepSeekNgramHasher?
  let tables: [any DeepSeekEngramTable]
  public let compute: DType
  /// DSpark, when the checkpoint ships it and the experts are held: a verify reads the union of
  /// its tokens' experts, which off a disk costs more than the tokens it saves, so a streamed
  /// release does not load the head at all.
  public let draft: DeepSeekDraft?

  /// Whether the caches go through the rounding they were trained with. A golden test turns it
  /// off to compare the continuous arithmetic on its own.
  public var fakeQuant = true {
    didSet {
      for block in blocks { block.attention.fakeQuant = fakeQuant }
      draft?.fakeQuant = fakeQuant
    }
  }

  public init(
    weights: DeepSeekWeights, tokenMap: [Int32]? = nil, tables: [any DeepSeekEngramTable]? = nil
  ) throws {
    let config = weights.config
    self.config = config
    self.compute = weights.dense ? .float32 : weights.compute
    let window = DeepSeekRope(
      dims: config.ropeDim, base: config.ropeTheta, originalContext: 0, factor: 1,
      betaFast: config.betaFast, betaSlow: config.betaSlow)
    let compressed = DeepSeekRope(
      dims: config.ropeDim, base: config.compressRopeTheta, originalContext: config.originalContext,
      factor: config.ropeFactor, betaFast: config.betaFast, betaSlow: config.betaSlow)
    self.blocks = try (0..<config.layers).map { layer in
      try DeepSeekBlock(
        layer: layer, prefix: "layers.\(layer)", config: config, weights: weights,
        rope: config.compressRatio(layer: layer) > 0 ? compressed : window)
    }
    self.embedding = try weights.array("embed.weight")
    self.norm = try weights.array("norm.weight")
    self.head = try weights.linear("head", wide: weights.dense)
    let streamed = weights.expertSlots != nil && !weights.dense
    self.draft =
      config.hasDraft && !streamed && weights.has("mtp.0.main_proj.weight")
      ? try DeepSeekDraft(weights: weights, rope: window, compute: compute) : nil

    if config.hasEngram {
      guard let tokenMap else {
        throw BonsaiError.missingComponent("an engram model needs its compressed token map")
      }
      self.hasher = try DeepSeekNgramHasher(tokenMap: tokenMap, config: config)
      self.tables =
        try tables
        ?? config.engramLayers.map { layer in
          let prefix = "layers.\(layer).engram.embed"
          let rows = try weights.checkpoint.entry(prefix + ".weight").byteCount
          if rows <= 256 << 20 {
            return ResidentEngramTable(
              codes: try weights.checkpoint.tensor(prefix + ".weight"),
              scales: try weights.checkpoint.tensor(prefix + ".scale"))
          }
          return try CheckpointEngramTable(checkpoint: weights.checkpoint, prefix: prefix)
        }
    } else {
      self.hasher = nil
      self.tables = []
    }
  }

  /// Whether the routed experts are read into slots rather than held.
  public var streamsExperts: Bool { blocks.first?.moe.experts is StreamedExpertBank }

  /// What the streamed layers have read so far, summed; nil when the experts are held.
  public var expertTraffic: ExpertStore.Summary? {
    ExpertStore.Summary(
      layers: blocks.compactMap { ($0.moe.experts as? StreamedExpertBank)?.store })
  }

  /// What a release holds in memory before any expert is read — everything but the routed
  /// experts, the n-gram tables, the tower and the draft head — and what one expert weighs.
  public static func footprint(of checkpoint: DeepSeekCheckpoint) -> (resident: Int, expert: Int) {
    var resident = 0
    var perExpert = 0
    for (name, entry) in checkpoint.entries {
      if name.hasPrefix("layers.0.ffn.experts.0.") { perExpert += entry.byteCount }
      let routed = name.contains(".ffn.experts.")
      let table = name.contains(".engram.embed.")
      let tower = name.hasPrefix("vision.") || name.hasPrefix("aligner.") || name.hasPrefix("mtp.")
      if !routed, !table, !tower { resident += entry.byteCount }
    }
    return (resident, perExpert)
  }

  /// Slots per layer for a release whose experts stream: `BonsaiRuntime.expertSlots` when it
  /// is set, otherwise as many as fit beside the resident weights, up to a quarter of the bank.
  public static func expertSlots(
    for checkpoint: DeepSeekCheckpoint, ceiling: Int = ResidencyManager.gpuCeiling
  ) -> Int {
    let config = checkpoint.config
    if BonsaiRuntime.expertSlots > 0 {
      return min(max(BonsaiRuntime.expertSlots, config.activatedExperts), config.routedExperts)
    }
    let (resident, perExpert) = footprint(of: checkpoint)
    let free = max(ceiling - resident - StreamedPlan.reserve, 0)
    let fit = free / max(config.layers * perExpert, 1)
    let wanted = min(fit, max(config.routedExperts / StreamedPlan.bankShare, config.activatedExperts))
    return max(wanted / 8 * 8, config.activatedExperts)
  }

  /// What a release occupies once it is open: the resident weights and every layer's slots.
  public static func residentBytes(in directory: URL) -> Int? {
    guard let checkpoint = try? DeepSeekCheckpoint(directory: directory) else { return nil }
    let (resident, perExpert) = footprint(of: checkpoint)
    let slots = expertSlots(for: checkpoint)
    return resident + checkpoint.config.layers * slots * perExpert
  }

  /// The last layer that makes global KV, which every layer above it reads: in the release the
  /// first decoder layer, projecting it from the encoder's last state. Nothing a prompt's early
  /// tokens owe the layers above it is made anywhere else.
  public var decoderStart: Int? { config.kvSourceLayers.max() }

  /// Whether a prompt runs the decoder over its last window of tokens only, as DeepSeek serves
  /// the model and post-trains it to expect. Off, every layer reads every token.
  public var boundedReplay = BonsaiRuntime.deepseekBoundedReplay

  /// A stretch of prompt run through the encoder only. The decoder's global KV is still made
  /// for it, from the encoder's output, but the decoder layers are not run: only a prompt's
  /// last window of tokens is, and the decoder's windows start again there. That is DeepSeek's
  /// bounded replay, which is close rather than exact and halves what a long prompt costs.
  public func encode(_ tokens: MLXArray, cache: ModelCache, embeddings: MLXArray? = nil) {
    guard let decoderStart else {
      eval(forward(tokens, cache: cache, embeddings: embeddings).trunk)
      return
    }
    let caches = layerCaches(cache)
    let s = tokens.dim(1)
    let (streams, pre, _) = run(
      tokens, caches: caches, embeddings: embeddings, through: decoderStart, observe: nil)
    blocks[decoderStart].compressOnly(
      streams, pre: pre, cache: caches[decoderStart], compute: compute)
    for layer in (decoderStart + 1)..<config.layers {
      caches[layer].window = nil
      caches[layer].advance(s)
    }
    eval(caches[decoderStart].latents ?? MLXArray(0), caches[decoderStart].keys ?? MLXArray(0))
  }

  public func makeCache(kvConfig: KVCacheConfig = KVCacheConfig()) -> ModelCache {
    ModelCache(layers: (0..<config.layers).map { _ in DeepSeekLayerCache() }, kvConfig: kvConfig)
  }

  private func layerCaches(_ cache: ModelCache) -> [DeepSeekLayerCache] {
    cache.layers.map { layer in
      guard let own = layer as? DeepSeekLayerCache else {
        fatalError("a DeepSeek-V4.1 model runs only on the caches it made")
      }
      return own
    }
  }

  /// Each engram layer's fetched rows for this chunk, `[b, s, columns * headDim]`. A dead
  /// position — part of a picture — takes no part in any n-gram.
  func engramRows(
    _ tokens: MLXArray, caches: [DeepSeekLayerCache], dead: [Bool]? = nil
  ) -> [MLXArray] {
    guard let hasher else { return [] }
    let ids = tokens.reshaped([-1]).asArray(Int32.self)
    let compressed = hasher.compress(ids, dead: dead)
    let history = caches[0].ngramTokens
    let hashes = hasher.hashes(compressed, history: history)
    caches[0].ngramTokens = hasher.history(after: history + compressed)
    return zip(hashes, tables).map { indices, table in
      do {
        return try table.rows(indices).reshaped([tokens.dim(0), tokens.dim(1), -1])
      } catch {
        fatalError("the engram table could not be read: \(error)")
      }
    }
  }

  /// The collapsed stream before the final norm. `observe` sees every layer's streams on the
  /// way, for a test that has to pin a difference to the layer that made it.
  public func trunk(
    _ tokens: MLXArray, cache: ModelCache, observe: ((Int, MLXArray) -> Void)? = nil
  ) -> MLXArray {
    forward(tokens, cache: cache, observe: observe).trunk
  }

  /// The trunk, and what the draft head reads: each target layer's input averaged over the
  /// copies, side by side. `embeddings` stands in for the token embeddings when a prompt carries
  /// pictures; every position of a picture's span holds the image token, which is how its
  /// positions are told apart from the words around them.
  public func forward(
    _ tokens: MLXArray, cache: ModelCache, embeddings: MLXArray? = nil,
    observe: ((Int, MLXArray) -> Void)? = nil
  ) -> (trunk: MLXArray, draftHidden: MLXArray?) {
    let (streams, pre, targets) = run(
      tokens, caches: layerCaches(cache), embeddings: embeddings, through: config.layers,
      observe: observe)
    let trunk = SinkhornMixer.collapse(streams, pre: pre).asType(compute)
    return (trunk, targets.isEmpty ? nil : concatenated(targets, axis: -1))
  }

  /// The streams and the next collapse after the first `through` blocks.
  private func run(
    _ tokens: MLXArray, caches: [DeepSeekLayerCache], embeddings: MLXArray?, through: Int,
    observe: ((Int, MLXArray) -> Void)?
  ) -> (streams: MLXArray, pre: MLXArray, targets: [MLXArray]) {
    let start = caches[0].offset
    let b = tokens.dim(0)
    let s = tokens.dim(1)
    var image: MLXArray?
    var dead: [Bool]?
    if embeddings != nil, let imageToken = config.imageTokenId {
      let spans = tokens.asType(.int32) .== Int32(imageToken)
      image = spans
      dead = spans.reshaped([-1]).asArray(Bool.self)
    }
    let rows = engramRows(tokens, caches: caches, dead: dead)

    let embedded =
      embeddings ?? take(embedding, tokens.reshaped([-1]), axis: 0).reshaped([b, s, -1])
    var streams = broadcast(
      embedded.asType(.float32).expandedDimensions(axis: 2),
      to: [b, s, config.hcMult, config.dim])
    var pre = SinkhornMixer.identity(batch: b, length: s, copies: config.hcMult)
    let selection = DeepSeekSelection()
    var targets: [MLXArray] = []
    let live = image.map { .!$0 }
    for block in blocks[..<through] {
      if let engram = block.engram {
        streams = engram(streams, rows: rows[engram.index], live: live)
      }
      if draft != nil, config.draftTargetLayers.contains(block.layer) {
        targets.append(streams.mean(axis: 2))
      }
      (streams, pre) = block(
        streams, pre: pre, caches: caches, selection: selection, start: start, compute: compute,
        image: image)
      observe?(block.layer, streams)
    }
    return (streams, pre, targets)
  }

  public func normed(_ h: MLXArray) -> MLXArray {
    MLXFast.rmsNorm(h.asType(compute), weight: norm.asType(compute), eps: config.normEps)
  }

  /// The collapsed stream the head reads, normalised.
  public func hidden(
    _ tokens: MLXArray, cache: ModelCache, observe: ((Int, MLXArray) -> Void)? = nil
  ) -> MLXArray {
    normed(trunk(tokens, cache: cache, observe: observe))
  }

  public func logits(_ hidden: MLXArray) -> MLXArray {
    let width = (head as? DenseLinear)?.weight.dtype ?? hidden.dtype
    return head(hidden.asType(width)).asType(.float32)
  }

  public func callAsFunction(_ tokens: MLXArray, cache: ModelCache) -> MLXArray {
    logits(hidden(tokens, cache: cache))
  }
}
