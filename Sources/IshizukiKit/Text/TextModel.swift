// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXFast

public final class DecoderLayer: @unchecked Sendable {
  public let isLinear: Bool
  private let linearAttention: GatedDeltaNet?
  private let selfAttention: Attention?
  private let inputLayerNorm: MLXArray?
  private let postAttentionLayerNorm: MLXArray?
  /// A widened residual replaces both layer norms: the streams are normalised by the gate that
  /// mixes them, so a hyper-connected layer ships no `input_layernorm` at all.
  private let attnResidual: GatedResidual?
  private let mlpResidual: GatedResidual?
  private let ple: PLEBlock?
  private let mlp: any FeedForward
  private let eps: Float

  public init(
    config: BonsaiConfig.TextConfig, layer: Int, isFullAttention: Bool,
    factory: PackedModuleFactory, store: WeightStore, rope: RotaryEmbedding,
    isSparse: Bool = false, path: String? = nil
  ) throws {
    self.isLinear = !isFullAttention
    self.eps = config.rmsNormEps

    if isFullAttention {
      self.selfAttention = try Attention(
        config: config, layer: layer, factory: factory, store: store, rope: rope,
        path: path)
      self.linearAttention = nil
    } else {
      self.linearAttention = try GatedDeltaNet(
        config: config, layer: layer, factory: factory, store: store)
      self.selfAttention = nil
    }

    let prefix = factory.tensorPrefix + (path ?? "model.layers.\(layer)")
    let module = path ?? "model.layers.\(layer)"
    if config.usesHyperConnections {
      self.inputLayerNorm = nil
      self.postAttentionLayerNorm = nil
      let count = config.hcCount ?? 1
      self.attnResidual = try DecoderLayer.residual(
        module + ".attn_hyper_connection", config: config, count: count, factory: factory,
        store: store)
      self.mlpResidual = try DecoderLayer.residual(
        module + ".mlp_hyper_connection", config: config, count: count, factory: factory,
        store: store)
      self.ple =
        layer == config.pleLayer
        ? try PLEBlock(config: config, module: module, factory: factory, store: store) : nil
    } else {
      self.inputLayerNorm = try store(prefix + ".input_layernorm.weight")
      self.postAttentionLayerNorm = try store(prefix + ".post_attention_layernorm.weight")
      self.attnResidual = nil
      self.mlpResidual = nil
      self.ple = nil
    }
    if isSparse {
      self.mlp = try MoEBlock(
        config: config, layer: layer, factory: factory, store: store, path: path)
    } else {
      self.mlp = try MLP(layer: layer, factory: factory, path: path)
    }
  }

  /// The gate that opens and closes a widened residual, read from one of a layer's two
  /// hyper-connection modules.
  static func residual(
    _ module: String, config: BonsaiConfig.TextConfig, count: Int,
    factory: PackedModuleFactory, store: WeightStore
  ) throws -> GatedResidual {
    // The streams are float32, and a dense weight left narrower is widened again on every call.
    func projection(_ path: String) throws -> any Projection {
      let built = try factory.projection(module + path)
      guard let dense = built as? DenseLinear, dense.weight.dtype != .float32 else { return built }
      let weight = dense.weight.asType(.float32)
      let bias = dense.bias?.asType(.float32)
      eval([weight] + (bias.map { [$0] } ?? []))
      return DenseLinear(weight: weight, bias: bias)
    }
    return GatedResidual(
      norm: try store(factory.tensorPrefix + module + ".hc_norm.weight"),
      down: try projection(".input_mix_weight_down"),
      up: try projection(".input_mix_weight_up"),
      inject: store.has(factory.tensorPrefix + module + ".block_inject_weight.weight")
        ? try projection(".block_inject_weight") : nil,
      count: count, width: config.hiddenSize, eps: config.rmsNormEps)
  }

  /// The block itself: attention and feed-forward, each read out of the streams and written
  /// back into them. `engrams` is this chunk's n-gram rows, needed only by the PLE layer.
  private func attend(
    _ normed: MLXArray, mask: MLXArray?, cache: LayerCache?, positions: MLXArray?
  ) -> MLXArray {
    if let linearAttention {
      return linearAttention(normed, cache: cache as? GatedDeltaNetCache)
    }
    if let selfAttention {
      return selfAttention(
        normed, mask: mask, cache: cache as? AttentionKVCache, positions: positions)
    }
    return normed
  }

  private func hyper(
    _ x: MLXArray, attn: GatedResidual, feed: GatedResidual, mask: MLXArray?,
    cache: LayerCache?, positions: MLXArray?, compute: DType, engrams: MLXArray?
  ) -> MLXArray {
    var streams = x
    if let ple, let engrams {
      let recurrent = cache as? GatedDeltaNetCache
      var state = recurrent?.pleConvState
      streams =
        streams + ple(engrams, streams: streams, state: &state).asType(streams.dtype)
      recurrent?.pleConvState = state
    }

    let opened = attn(streams)
    let attended = attend(
      opened.mixed.asType(compute), mask: mask, cache: cache, positions: positions)
    streams = GatedResidual.close(opened, with: attended.asType(streams.dtype))

    let second = feed(streams)
    return GatedResidual.close(
      second, with: mlp(second.mixed.asType(compute)).asType(streams.dtype))
  }

  public func callAsFunction(
    _ x: MLXArray, mask: MLXArray?, cache: LayerCache?, positions: MLXArray?,
    compute: DType? = nil, engrams: MLXArray? = nil
  ) -> MLXArray {
    if let attnResidual, let mlpResidual {
      return hyper(
        x, attn: attnResidual, feed: mlpResidual, mask: mask, cache: cache,
        positions: positions, compute: compute ?? x.dtype, engrams: engrams)
    }
    guard let inputLayerNorm, let postAttentionLayerNorm else {
      fatalError("a layer with neither layer norms nor a widened residual cannot run")
    }
    // The residual carries float32 while the modules run in the pack's own width: sixty-four
    // layers of bf16 addition is where this runtime drifts from the reference, and a wider
    // accumulator costs a cast rather than a wider matmul.
    // Not the norm weight's own dtype: a pack may store its norms wider than its projections,
    // and running the projections at the norm's width is how this got three times slower.
    let compute = compute ?? x.dtype
    let normed = MLXFast.rmsNorm(
      x, weight: inputLayerNorm.asType(x.dtype), eps: eps
    ).asType(compute)

    let attended = attend(normed, mask: mask, cache: cache, positions: positions)

    let h = x + attended.asType(x.dtype)
    let postNormed = MLXFast.rmsNorm(
      h, weight: postAttentionLayerNorm.asType(h.dtype), eps: eps
    ).asType(compute)
    return h + mlp(postNormed).asType(h.dtype)
  }
}

public final class TextModel: @unchecked Sendable {
  public let config: BonsaiConfig.TextConfig
  public let embedTokens: PackedEmbedding
  public let layers: [DecoderLayer]
  private let norm: MLXArray?
  /// A hyper-connected model has no final norm of its own: the mixer that folds the streams
  /// back into one width normalises them on the way, and the head reads what it returns.
  private let mixer: GatedResidual?
  private let engrams: EngramStore?
  private let hasher: NgramHasher?
  public let lmHead: PackedLinear
  public let rope: RotaryEmbedding
  private let eps: Float

  public init(
    config: BonsaiConfig, factory: PackedModuleFactory, store: WeightStore,
    ropeScaling: RopeScaling = .none
  ) throws {
    let text = config.textConfig
    self.config = text
    self.eps = text.rmsNormEps

    let rope = RotaryEmbedding(
      dimensions: text.ropeDimensions,
      base: text.ropeParameters.ropeTheta,
      mropeSection: text.ropeParameters.mropeSection,
      interleaved: text.ropeParameters.mropeInterleaved ?? false,
      scaling: ropeScaling)

    self.rope = rope
    self.embedTokens = try factory.embedding("model.embed_tokens")

    let fullAttention = text.isFullAttention
    let sparse = text.isSparse
    var built: [DecoderLayer] = []
    built.reserveCapacity(text.numHiddenLayers)
    for layer in 0..<text.numHiddenLayers {
      built.append(
        try DecoderLayer(
          config: text, layer: layer, isFullAttention: fullAttention[layer],
          factory: factory, store: store, rope: rope, isSparse: sparse[layer]))
    }
    self.layers = built

    if text.usesHyperConnections {
      self.norm = nil
      self.mixer = try DecoderLayer.residual(
        "model.hyper_connection_mixer", config: text, count: text.hcCount ?? 1,
        factory: factory, store: store)
    } else {
      self.norm = try store(factory.tensorPrefix + "model.norm.weight")
      self.mixer = nil
    }

    if text.pleLayer != nil, let table = store.engrams {
      self.engrams = table
      self.hasher = NgramHasher(
        multipliers: table.layout.multipliers, ngramSize: table.layout.ngramSize,
        headsPerNgram: table.layout.headsPerNgram, eosTokenId: table.layout.eosTokenId)
    } else {
      self.engrams = nil
      self.hasher = nil
    }

    if text.tieWordEmbeddings, !store.has(factory.tensorPrefix + "lm_head.weight") {
      self.lmHead = try factory.tiedHead("model.embed_tokens")
    } else {
      self.lmHead = try factory.linear("lm_head")
    }
  }

  public func hidden(
    inputs: MLXArray?, inputEmbeddings: MLXArray? = nil,
    cache: ModelCache? = nil, positions: MLXArray? = nil
  ) -> MLXArray {
    normed(
      trunk(
        inputs: inputs, inputEmbeddings: inputEmbeddings, cache: cache,
        positions: positions))
  }

  public func normed(_ h: MLXArray) -> MLXArray {
    if let mixer { return mixer(h).mixed.asType(h.dtype) }
    return MLXFast.rmsNorm(h, weight: norm!.asType(h.dtype), eps: eps)
  }

  /// The last layer's activation before the final norm. An MTP head fuses this, not the
  /// normalized hidden the head reads.
  public func trunk(
    inputs: MLXArray?, inputEmbeddings: MLXArray? = nil,
    cache: ModelCache? = nil, positions: MLXArray? = nil
  ) -> MLXArray {
    var h: MLXArray
    if let inputEmbeddings {
      h = inputEmbeddings
    } else if let inputs {
      h = embedTokens(inputs)
    } else {
      fatalError("hidden(inputs:) requires token ids or embeddings")
    }

    let compute = h.dtype
    let offset = cache?.offset ?? 0
    let mask = causalMask(length: h.dim(1), offset: offset, dtype: compute)

    let engramRows = fetchEngrams(inputs: inputs, cache: cache)

    h = h.asType(.float32)
    // Every stream starts as a copy of the embedding, and they only diverge once a block has
    // been gated into them.
    if let count = config.hcCount, count > 1 {
      h = concatenated(Array(repeating: h, count: count), axis: -1)
    }
    for (index, layer) in layers.enumerated() {
      h = layer(
        h, mask: mask, cache: cache?.layers[index], positions: positions, compute: compute,
        engrams: engramRows)
    }
    return h.asType(compute)
  }

  /// This chunk's n-gram rows, and the tokens the next chunk will need to reach back over.
  ///
  /// The addresses are the tokens themselves, so the whole fetch is known before the first
  /// layer runs — which is the only reason a table this size can sit on disk.
  private func fetchEngrams(inputs: MLXArray?, cache: ModelCache?) -> MLXArray? {
    guard let engrams, let hasher, let layer = config.pleLayer else { return nil }
    guard let inputs else {
      fatalError("a per-layer-embedding model needs token ids, not embeddings alone")
    }
    let recurrent = cache?.layers[layer] as? GatedDeltaNetCache
    let context = hasher.ngramSize - 1
    let previous =
      recurrent?.pleTokens?.asArray(Int32.self).map(Int.init)
      ?? Array(repeating: hasher.eosTokenId, count: context)
    let current = inputs.reshaped([-1]).asArray(Int32.self).map(Int.init)
    let history = previous + current
    recurrent?.pleTokens = MLXArray(history.suffix(context).map(Int32.init))

    do {
      let hashes = hasher.hashes(history, last: current.count)
      // A fetch is bounded by the buffer the store holds, so a chunk longer than that is read
      // in pieces rather than refused.
      let stride = max(1, engrams.capacity / engrams.layout.heads)
      var pieces: [MLXArray] = []
      var start = 0
      while start < hashes.count {
        let end = min(start + stride, hashes.count)
        // What the store hands back is a view onto the buffer it fetched into, and there are
        // only two of those. Widening is what takes a copy of it; without one, the next fetch
        // — the next piece, or the next token — rewrites these rows before they are read.
        let piece = try engrams.embeddings(hashes: Array(hashes[start..<end]))
          .asType(.float32)
        eval(piece)
        pieces.append(piece)
        start = end
      }
      let rows = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0)
      return rows.reshaped([1, current.count, engrams.layout.width])
    } catch {
      fatalError("the n-gram table could not be read: \(error)")
    }
  }

  public func callAsFunction(
    _ inputs: MLXArray?, inputEmbeddings: MLXArray? = nil,
    cache: ModelCache? = nil, positions: MLXArray? = nil
  ) -> MLXArray {
    let h = hidden(
      inputs: inputs, inputEmbeddings: inputEmbeddings, cache: cache,
      positions: positions)
    return lmHead(h)
  }

  public func lastLogits(_ h: MLXArray) -> MLXArray {
    lmHead(h[0..., -1, 0...]).expandedDimensions(axis: 1)
  }

  public func lastLogits(
    inputs: MLXArray?, inputEmbeddings: MLXArray? = nil,
    cache: ModelCache? = nil, positions: MLXArray? = nil
  ) -> MLXArray {
    lastLogits(
      hidden(
        inputs: inputs, inputEmbeddings: inputEmbeddings, cache: cache,
        positions: positions))
  }

  public func makeCache(kvConfig: KVCacheConfig = KVCacheConfig()) -> ModelCache {
    ModelCache(config: config, kvConfig: kvConfig)
  }
}

extension TextModel {
  /// The n-gram rows a chunk of tokens resolves to. A caller that drives the layers itself —
  /// a test walking them one at a time, a bench — has to fetch these the way `trunk` does.
  public func engramRows(inputs: MLXArray, cache: ModelCache?) -> MLXArray? {
    fetchEngrams(inputs: inputs, cache: cache)
  }
}
