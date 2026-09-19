// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXFast

/// A multi-token-prediction head: the short stack a pack ships beside the backbone so one
/// forward pass can propose the token after the one it just produced.
///
/// It predicts t+2 by fusing the backbone's pre-norm hidden state at t with the embedding of
/// the token sampled at t+1, then running that through its own decoder layers and the
/// backbone's head. The fused pair is normalized separately before the join, which is why the
/// pack carries two norms rather than reusing the model's.
public final class MTPHead: @unchecked Sendable {
  public let layerCount: Int

  private let preNormEmbedding: MLXArray
  private let preNormHidden: MLXArray
  private let fc: any Projection
  private let layers: [DecoderLayer]
  private let norm: MLXArray
  private let eps: Float
  private let schedule: [Bool]

  public init(
    config: BonsaiConfig.TextConfig, factory: PackedModuleFactory, store: WeightStore,
    rope: RotaryEmbedding
  ) throws {
    let count = config.mtpNumHiddenLayers ?? 0
    guard count > 0 else {
      throw BonsaiError.missingComponent(
        "the config declares no MTP layers (mtp_num_hidden_layers is 0)")
    }
    let prefix = factory.tensorPrefix + "mtp"
    guard store.has(prefix + ".fc.weight") else {
      throw BonsaiError.missingComponent(
        "the config declares \(count) MTP layer(s) but the pack ships no \(prefix).* tensors")
    }

    self.layerCount = count
    self.eps = config.rmsNormEps
    self.preNormEmbedding = try store(prefix + ".pre_fc_norm_embedding.weight")
    self.preNormHidden = try store(prefix + ".pre_fc_norm_hidden.weight")
    self.fc = try factory.projection("mtp.fc")
    self.norm = try store(prefix + ".norm.weight")

    // The published heads are full-attention throughout; a recurrent draft layer would need its
    // own state plumbing through the accept/reject path, so it is refused rather than guessed.
    var built: [DecoderLayer] = []
    for layer in 0..<count {
      let path = "mtp.layers.\(layer)"
      guard store.has(factory.tensorPrefix + path + ".self_attn.q_proj.weight") else {
        throw BonsaiError.unsupportedModel(
          "MTP layer \(layer) is not a full-attention layer; this runtime drafts only with "
            + "attention heads")
      }
      built.append(
        try DecoderLayer(
          config: config, layer: -1 - layer, isFullAttention: true,
          factory: factory, store: store, rope: rope, path: path))
    }
    self.layers = built
    self.schedule = Array(repeating: true, count: count)
  }

  public func makeCache(kvConfig: KVCacheConfig = KVCacheConfig()) -> ModelCache {
    ModelCache(fullAttention: schedule, kvConfig: kvConfig)
  }

  /// `hidden` is the backbone's pre-norm activation at the positions being extended, and
  /// `embeddings` the embedding of the tokens sampled just after them. Returns the normalized
  /// draft hidden state, which the backbone's own head turns into logits.
  public func callAsFunction(
    hidden: MLXArray, embeddings: MLXArray, cache: ModelCache? = nil,
    positions: MLXArray? = nil
  ) -> MLXArray {
    let e = MLXFast.rmsNorm(
      embeddings, weight: preNormEmbedding.asType(embeddings.dtype), eps: eps)
    let h = MLXFast.rmsNorm(hidden, weight: preNormHidden.asType(hidden.dtype), eps: eps)

    var fused = fc(concatenated([e, h.asType(e.dtype)], axis: -1))

    let offset = cache?.offset ?? 0
    let mask = causalMask(length: fused.dim(1), offset: offset, dtype: fused.dtype)
    for (index, layer) in layers.enumerated() {
      fused = layer(fused, mask: mask, cache: cache?.layers[index], positions: positions)
    }
    return MLXFast.rmsNorm(fused, weight: norm.asType(fused.dtype), eps: eps)
  }
}
