// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

public final class DecoderLayer: @unchecked Sendable {
  public let isLinear: Bool
  private let linearAttention: GatedDeltaNet?
  private let selfAttention: Attention?
  private let inputLayerNorm: MLXArray
  private let postAttentionLayerNorm: MLXArray
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
    self.inputLayerNorm = try store(prefix + ".input_layernorm.weight")
    self.postAttentionLayerNorm = try store(prefix + ".post_attention_layernorm.weight")
    if isSparse {
      self.mlp = try MoEBlock(
        config: config, layer: layer, factory: factory, store: store, path: path)
    } else {
      self.mlp = try MLP(layer: layer, factory: factory, path: path)
    }
  }

  public func callAsFunction(
    _ x: MLXArray, mask: MLXArray?, cache: LayerCache?, positions: MLXArray?,
    compute: DType? = nil
  ) -> MLXArray {
    // The residual carries float32 while the modules run in the pack's own width: sixty-four
    // layers of bf16 addition is where this runtime drifts from the reference, and a wider
    // accumulator costs a cast rather than a wider matmul.
    // Not the norm weight's own dtype: a pack may store its norms wider than its projections,
    // and running the projections at the norm's width is how this got three times slower.
    let compute = compute ?? x.dtype
    let normed = MLXFast.rmsNorm(
      x, weight: inputLayerNorm.asType(x.dtype), eps: eps
    ).asType(compute)

    let attended: MLXArray
    if let linearAttention {
      attended = linearAttention(normed, cache: cache as? GatedDeltaNetCache)
    } else if let selfAttention {
      attended = selfAttention(
        normed, mask: mask, cache: cache as? AttentionKVCache, positions: positions)
    } else {
      attended = normed
    }

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
  private let norm: MLXArray
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

    self.norm = try store(factory.tensorPrefix + "model.norm.weight")
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
    MLXFast.rmsNorm(h, weight: norm.asType(h.dtype), eps: eps)
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

    h = h.asType(.float32)
    for (index, layer) in layers.enumerated() {
      h = layer(
        h, mask: mask, cache: cache?.layers[index], positions: positions, compute: compute)
    }
    return h.asType(compute)
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
