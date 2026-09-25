// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// V4.1's widened residual: copies of the stream, mixed by a doubly stochastic matrix.

import Foundation
import MLX
import MLXFast

/// How one sub-block reads and rewrites the `hc` residual copies (DeepSeek's mHC).
///
/// Each token's streams predict three sets of coefficients through one projection: `pre`, how
/// to collapse the copies into the sub-block's input; `post`, how much of its answer each copy
/// takes back; and `comb`, a mixing matrix pushed towards doubly stochastic by Sinkhorn so the
/// residual neither grows nor fades across forty layers. V4.1 shifts `pre` by one sub-block:
/// what a sub-block predicts is what the next one collapses with.
public struct SinkhornMixer: @unchecked Sendable {
  public struct Mix: @unchecked Sendable {
    public var pre: MLXArray
    public var post: MLXArray
    public var comb: MLXArray
  }

  let projection: MLXArray
  let scale: MLXArray
  let base: MLXArray
  let copies: Int
  let iterations: Int
  let eps: Float
  let normEps: Float

  public init(
    projection: MLXArray, scale: MLXArray, base: MLXArray, copies: Int, iterations: Int,
    eps: Float, normEps: Float
  ) {
    self.projection = projection.asType(.float32)
    self.scale = scale.asType(.float32)
    self.base = base.asType(.float32)
    self.copies = copies
    self.iterations = iterations
    self.eps = eps
    self.normEps = normEps
  }

  /// `streams` is `[b, s, copies, dim]`. One statistic per token over every copy at once.
  public func callAsFunction(_ streams: MLXArray) -> Mix {
    let b = streams.dim(0)
    let s = streams.dim(1)
    let flat = streams.reshaped([b, s, -1]).asType(.float32)
    let rms = rsqrt(flat.square().mean(axis: -1, keepDims: true) + normEps)
    let mixes = matmul(flat, projection.T) * rms
    let (pre, post, comb) = Self.split(
      mixes.reshaped([b * s, -1]), scale: scale, base: base, copies: copies,
      iterations: iterations, eps: eps)
    return Mix(
      pre: pre.reshaped([b, s, copies]), post: post.reshaped([b, s, copies]),
      comb: comb.reshaped([b, s, copies, copies]))
  }

  /// The copies collapsed into one input, weighted by `pre`.
  public static func collapse(_ streams: MLXArray, pre: MLXArray) -> MLXArray {
    (pre.expandedDimensions(axis: -1) * streams.asType(.float32)).sum(axis: -2)
  }

  /// A sub-block's answer spread back over the copies, the residual mixed in through `comb`:
  /// copy `k` becomes `post[k] * y + sum_j comb[j, k] * residual[j]`.
  public static func expand(
    _ y: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
  ) -> MLXArray {
    let spread = post.expandedDimensions(axis: -1) * y.asType(.float32).expandedDimensions(axis: -2)
    return spread + matmul(comb.swappedAxes(-1, -2), residual.asType(.float32))
  }

  /// The identity collapse the first block starts from: copy zero, which is the embedding.
  public static func identity(batch: Int, length: Int, copies: Int) -> MLXArray {
    let row = MLXArray((0..<copies).map { Float($0 == 0 ? 1 : 0) })
    return broadcast(row.reshaped([1, 1, copies]), to: [batch, length, copies])
  }

  static func split(
    _ mixes: MLXArray, scale: MLXArray, base: MLXArray, copies: Int, iterations: Int, eps: Float
  ) -> (MLXArray, MLXArray, MLXArray) {
    let rows = mixes.dim(0)
    #if canImport(Metal)
      let out = kernel(
        [mixes, scale, base, MLXArray([eps])],
        template: [("HC", copies), ("ITERS", iterations), ("IT", mixes.dtype)],
        grid: (rows, 1, 1), threadGroup: (min(rows, 64), 1, 1),
        outputShapes: [[rows, copies], [rows, copies], [rows, copies * copies]],
        outputDTypes: [.float32, .float32, .float32])
      return (out[0], out[1], out[2])
    #else
      return reference(
        mixes, scale: scale, base: base, copies: copies, iterations: iterations, eps: eps)
    #endif
  }

  /// The same split in ordinary operations, for a test to hold the kernel against.
  static func reference(
    _ mixes: MLXArray, scale: MLXArray, base: MLXArray, copies hc: Int, iterations: Int,
    eps: Float
  ) -> (MLXArray, MLXArray, MLXArray) {
    let pre = sigmoid(mixes[0..., ..<hc] * scale[0] + base[..<hc]) + eps
    let post = 2 * sigmoid(mixes[0..., hc..<(2 * hc)] * scale[1] + base[hc..<(2 * hc)])
    var comb = (mixes[0..., (2 * hc)...] * scale[2] + base[(2 * hc)...])
      .reshaped([-1, hc, hc])
    comb = softmax(comb, axis: -1, precise: true) + eps
    comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
    for _ in 1..<max(iterations, 1) {
      comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
      comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
    }
    return (pre, post, comb.reshaped([-1, hc * hc]))
  }

  #if canImport(Metal)
    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
      name: "ishizuki_sinkhorn_split",
      inputNames: ["mixes", "scale", "base", "eps"],
      outputNames: ["pre", "post", "comb"],
      source: """
            uint row = thread_position_in_grid.x;
            constexpr uint MIX = (2 + HC) * HC;
            const device float* m = mixes + row * MIX;
            float e = eps[0];
            for (uint j = 0; j < HC; ++j) {
                pre[row * HC + j] = 1.0f / (1.0f + metal::exp(-(m[j] * scale[0] + base[j]))) + e;
                post[row * HC + j] =
                    2.0f / (1.0f + metal::exp(-(m[HC + j] * scale[1] + base[HC + j])));
            }
            float c[HC * HC];
            for (uint i = 0; i < HC * HC; ++i) {
                c[i] = m[2 * HC + i] * scale[2] + base[2 * HC + i];
            }
            for (uint j = 0; j < HC; ++j) {
                float top = c[j * HC];
                for (uint k = 1; k < HC; ++k) top = metal::max(top, c[j * HC + k]);
                float total = 0.0f;
                for (uint k = 0; k < HC; ++k) {
                    c[j * HC + k] = metal::exp(c[j * HC + k] - top);
                    total += c[j * HC + k];
                }
                for (uint k = 0; k < HC; ++k) c[j * HC + k] = c[j * HC + k] / total + e;
            }
            for (uint k = 0; k < HC; ++k) {
                float total = 0.0f;
                for (uint j = 0; j < HC; ++j) total += c[j * HC + k];
                for (uint j = 0; j < HC; ++j) c[j * HC + k] /= (total + e);
            }
            for (uint it = 1; it < ITERS; ++it) {
                for (uint j = 0; j < HC; ++j) {
                    float total = 0.0f;
                    for (uint k = 0; k < HC; ++k) total += c[j * HC + k];
                    for (uint k = 0; k < HC; ++k) c[j * HC + k] /= (total + e);
                }
                for (uint k = 0; k < HC; ++k) {
                    float total = 0.0f;
                    for (uint j = 0; j < HC; ++j) total += c[j * HC + k];
                    for (uint j = 0; j < HC; ++j) c[j * HC + k] /= (total + e);
                }
            }
            for (uint i = 0; i < HC * HC; ++i) comb[row * HC * HC + i] = c[i];
        """)
  #endif
}
