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
  private let mlp: MLP
  private let eps: Float

  public init(
    config: BonsaiConfig.TextConfig, layer: Int, isFullAttention: Bool,
    factory: PackedModuleFactory, store: WeightStore, rope: RotaryEmbedding,
    path: String? = nil
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
    self.mlp = try MLP(layer: layer, factory: factory, path: path)
  }

  public func callAsFunction(
    _ x: MLXArray, mask: MLXArray?, cache: LayerCache?, positions: MLXArray?
  ) -> MLXArray {
    let normed = MLXFast.rmsNorm(
      x, weight: inputLayerNorm.asType(x.dtype), eps: eps)

    let attended: MLXArray
    if let linearAttention {
      attended = linearAttention(normed, cache: cache as? GatedDeltaNetCache)
    } else if let selfAttention {
      attended = selfAttention(
        normed, mask: mask, cache: cache as? AttentionKVCache, positions: positions)
    } else {
      attended = normed
    }

    let h = x + attended
    let postNormed = MLXFast.rmsNorm(
      h, weight: postAttentionLayerNorm.asType(h.dtype), eps: eps)
    return h + mlp(postNormed)
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
    var built: [DecoderLayer] = []
    built.reserveCapacity(text.numHiddenLayers)
    for layer in 0..<text.numHiddenLayers {
      built.append(
        try DecoderLayer(
          config: text, layer: layer, isFullAttention: fullAttention[layer],
          factory: factory, store: store, rope: rope))
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

    let offset = cache?.offset ?? 0
    let mask = causalMask(length: h.dim(1), offset: offset, dtype: h.dtype)

    for (index, layer) in layers.enumerated() {
      h = layer(h, mask: mask, cache: cache?.layers[index], positions: positions)
    }
    return h
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
