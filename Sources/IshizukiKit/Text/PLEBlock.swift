// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What a token's n-grams contribute, gated by what the streams already hold.

import Foundation
import MLX
import MLXNN

/// The block that turns fetched n-gram rows into something every residual stream can take.
///
/// The embedding is not simply added. It becomes one value and a key per stream; each stream's
/// own activations ask that stream's key how much of the value it wants, and only then does a
/// dilated depthwise convolution give the result a few tokens of local context. So a phrase the
/// table knows arrives weighted by whether this stream, here, had any use for it.
///
/// The convolution is dilated by the n-gram order, which spaces its taps one whole n-gram apart.
public struct PLEBlock: @unchecked Sendable {
  let keyProj: any Projection
  let valueProj: any Projection
  let normKey: MLXArray
  let normQuery: MLXArray
  let normConv: MLXArray
  let conv: MLXArray
  let count: Int
  let width: Int
  let kernel: Int
  let dilation: Int
  let eps: Float

  public init(
    keyProj: any Projection, valueProj: any Projection, normKey: MLXArray,
    normQuery: MLXArray, normConv: MLXArray, conv: MLXArray, count: Int, width: Int,
    kernel: Int, dilation: Int, eps: Float
  ) {
    self.keyProj = keyProj
    self.valueProj = valueProj
    self.normKey = normKey
    self.normQuery = normQuery
    self.normConv = normConv
    self.conv = conv
    self.count = count
    self.width = width
    self.kernel = kernel
    self.dilation = dilation
    self.eps = eps
  }

  /// How many earlier positions the convolution has to keep to answer for the next token.
  public var stateLength: Int { (kernel - 1) * dilation }

  /// `embeddings` is one row per token, every n-gram head laid side by side; `streams` is what
  /// the layer is carrying. The answer is the width of the streams, to be added to them.
  public func callAsFunction(
    _ embeddings: MLXArray, streams: MLXArray, state: inout MLXArray?
  ) -> MLXArray {
    let leading = Array(streams.shape.dropLast())

    let keys = groupedRMSNorm(
      keyProj(embeddings), weight: normKey, groups: count, width: width, eps: eps
    ).reshaped(leading + [count, width])
    let queries = groupedRMSNorm(
      streams, weight: normQuery, groups: count, width: width, eps: eps
    ).reshaped(leading + [count, width])

    var gate =
      (keys * queries).sum(axis: -1, keepDims: true).asType(.float32) / sqrt(Float(width))
    // A signed square root: the sign survives, the magnitude is pulled in, so one loud stream
    // cannot saturate the gate on its own.
    gate = sign(gate) * sqrt(maximum(abs(gate), 1e-6))

    let value = valueProj(embeddings).expandedDimensions(axis: -2)
    let gated = (sigmoid(gate) * value.asType(.float32)).asType(streams.dtype)
    let flat = gated.reshaped(leading + [count * width])

    let normed = groupedRMSNorm(
      flat, weight: normConv, groups: count, width: width, eps: eps)
    return flat + shortConv(normed, state: &state)
  }

  /// A depthwise convolution over the tokens, carrying the few it needs across a step boundary
  /// so that a decoded token sees the same neighbours a prefilled one would have.
  func shortConv(_ x: MLXArray, state: inout MLXArray?) -> MLXArray {
    let keep = stateLength
    let channels = x.dim(-1)
    let previous =
      state ?? MLXArray.zeros(Array(x.shape.dropLast(2)) + [keep, channels], dtype: x.dtype)
    let padded = concatenated([previous, x], axis: -2)
    state = padded[.ellipsis, (padded.dim(-2) - keep)..., 0...]
    return silu(conv1d(padded, conv, dilation: dilation, groups: channels))
  }
}

extension PLEBlock {
  /// Read from a pack. Only the one layer named by `ple_layer_ids` carries these tensors, and
  /// the convolution is dilated by the n-gram order so its taps land one whole n-gram apart.
  public init(
    config: BonsaiConfig.TextConfig, module: String, factory: PackedModuleFactory,
    store: WeightStore
  ) throws {
    let prefix = factory.tensorPrefix + module + ".ple"
    self.init(
      keyProj: try factory.projection(module + ".ple.key_proj"),
      valueProj: try factory.projection(module + ".ple.value_proj"),
      normKey: try store(prefix + ".norm_key.weight"),
      normQuery: try store(prefix + ".norm_query.weight"),
      normConv: try store(prefix + ".norm_conv.weight"),
      conv: try store(prefix + ".conv1d.weight"),
      count: config.hcCount ?? 1,
      width: config.hiddenSize,
      kernel: config.pleConvKernelSize ?? 4,
      dilation: config.ngramSize ?? 3,
      eps: config.rmsNormEps)
  }
}
