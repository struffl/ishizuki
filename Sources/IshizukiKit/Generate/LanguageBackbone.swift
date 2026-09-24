// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What generation asks of a language model, whichever architecture answers it.

import Foundation
import MLX

/// The handful of calls the generator, the session cache and the benches make of a model.
///
/// The hybrid Qwen stack and DeepSeek-V4.1 share nothing below this line — not their caches'
/// insides, not their residuals, not how a position reaches a key — so this is where they meet.
/// `trunk` is the last activation before the final norm, which is what a draft head fuses.
public protocol LanguageBackbone: AnyObject, Sendable {
  func makeCache(kvConfig: KVCacheConfig) -> ModelCache
  func trunk(
    inputs: MLXArray?, inputEmbeddings: MLXArray?, cache: ModelCache?, positions: MLXArray?
  ) -> MLXArray
  func normed(_ h: MLXArray) -> MLXArray
  /// Logits at every position of a normalised hidden state.
  func logits(_ h: MLXArray) -> MLXArray
  /// Logits at the last position only, the length axis kept.
  func lastLogits(_ h: MLXArray) -> MLXArray
  func embed(_ ids: MLXArray) -> MLXArray
  /// How many of a prompt's last tokens need the whole stack when the ones before them can take
  /// a shorter path, which `encode` is; nil when every token needs every layer.
  var replayTail: Int? { get }
  func encode(inputs: MLXArray, inputEmbeddings: MLXArray?, cache: ModelCache)
}

extension LanguageBackbone {
  public var replayTail: Int? { nil }

  public func encode(inputs: MLXArray, inputEmbeddings: MLXArray?, cache: ModelCache) {
    eval(trunk(inputs: inputs, inputEmbeddings: inputEmbeddings, cache: cache, positions: nil))
  }

  public func makeCache() -> ModelCache { makeCache(kvConfig: KVCacheConfig()) }

  public func trunk(inputs: MLXArray, cache: ModelCache?) -> MLXArray {
    trunk(inputs: inputs, inputEmbeddings: nil, cache: cache, positions: nil)
  }

  public func hidden(
    inputs: MLXArray?, inputEmbeddings: MLXArray? = nil, cache: ModelCache? = nil,
    positions: MLXArray? = nil
  ) -> MLXArray {
    normed(
      trunk(inputs: inputs, inputEmbeddings: inputEmbeddings, cache: cache, positions: positions))
  }

  public func callAsFunction(
    _ inputs: MLXArray?, inputEmbeddings: MLXArray? = nil, cache: ModelCache? = nil,
    positions: MLXArray? = nil
  ) -> MLXArray {
    logits(
      hidden(inputs: inputs, inputEmbeddings: inputEmbeddings, cache: cache, positions: positions))
  }

  public func lastLogits(
    inputs: MLXArray?, inputEmbeddings: MLXArray? = nil, cache: ModelCache? = nil,
    positions: MLXArray? = nil
  ) -> MLXArray {
    lastLogits(
      hidden(inputs: inputs, inputEmbeddings: inputEmbeddings, cache: cache, positions: positions))
  }
}

extension TextModel: LanguageBackbone {
  public func logits(_ h: MLXArray) -> MLXArray { lmHead(h) }
  public func embed(_ ids: MLXArray) -> MLXArray { embedTokens(ids) }
}

extension DeepSeekModel: LanguageBackbone {
  public func trunk(
    inputs: MLXArray?, inputEmbeddings: MLXArray?, cache: ModelCache?, positions: MLXArray?
  ) -> MLXArray {
    guard let inputs else {
      fatalError("DeepSeek-V4.1 needs the token ids even beside embeddings: the engram hashes them")
    }
    return forward(inputs, cache: cache ?? makeCache(), embeddings: inputEmbeddings).trunk
  }

  public func lastLogits(_ h: MLXArray) -> MLXArray {
    logits(h[0..., (h.dim(1) - 1)..., 0...])
  }

  public func embed(_ ids: MLXArray) -> MLXArray {
    take(embedding, ids.reshaped([-1]), axis: 0).reshaped(ids.shape + [config.dim])
  }

  public var replayTail: Int? { boundedReplay && decoderStart != nil ? config.window : nil }

  public func encode(inputs: MLXArray, inputEmbeddings: MLXArray?, cache: ModelCache) {
    encode(inputs, cache: cache, embeddings: inputEmbeddings)
  }
}
