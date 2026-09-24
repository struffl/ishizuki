// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// V4.1's feed-forward: a few of hundreds of routed experts, and one every token passes through.

import Foundation
import MLX

public final class DeepSeekMoE: @unchecked Sendable {
  let router: MLXArray
  let bias: MLXArray
  let imageBias: MLXArray?
  let experts: any DeepSeekExpertBank
  let shared: (gate: any Projection, up: any Projection, down: any Projection)
  let topK: Int
  let normalize: Bool
  let routeScale: Float
  let limit: Float
  let layer: Int

  init(layer: Int, prefix: String, config: DeepSeekConfig, weights: DeepSeekWeights) throws {
    let (routed, activated) = config.experts(layer: layer)
    self.layer = layer
    self.router = try weights.float32(prefix + ".gate.weight")
    self.bias = try weights.float32(prefix + ".gate.bias")
    self.imageBias =
      weights.has(prefix + ".gate.bias_vl") ? try weights.float32(prefix + ".gate.bias_vl") : nil
    guard router.dim(0) == routed, bias.dim(0) == routed else {
      throw BonsaiError.shapeMismatch("layer \(layer) routes to \(router.dim(0)) experts, not \(routed)")
    }
    self.experts = try weights.experts(prefix + ".experts", count: routed)
    self.shared = (
      try weights.linear(prefix + ".shared_experts.w1"),
      try weights.linear(prefix + ".shared_experts.w3"),
      try weights.linear(prefix + ".shared_experts.w2")
    )
    self.topK = activated
    self.normalize = config.normTopkProb
    self.routeScale = config.routeScale
    self.limit = config.swigluLimit
  }

  /// Which experts each row goes to and how much of each it takes. The bias only chooses: the
  /// weights come from the scores it was added to, not from the sum.
  func route(_ x: MLXArray, image: MLXArray? = nil) -> (chosen: MLXArray, weights: MLXArray) {
    let scores = sqrt(softplusTorch(matmul(x.asType(.float32), router.T)))
    var steer = bias
    if let image, let imageBias {
      steer = MLX.where(image.expandedDimensions(axis: -1), imageBias, bias)
    }
    let ranked = scores + steer
    let chosen = argPartition(-ranked, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    BonsaiRuntime.onRoute?(layer, chosen)
    var weights = takeAlong(scores, chosen, axis: -1)
    if normalize, topK > 1 {
      weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
    }
    return (chosen, weights * routeScale)
  }

  func callAsFunction(_ x: MLXArray, image: MLXArray? = nil) -> MLXArray {
    let shape = x.shape
    let rows = x.reshaped([-1, shape.last!])
    let (chosen, weights) = route(rows, image: image?.reshaped([-1]))
    let routed: MLXArray
    do {
      routed = try experts.run(rows, chosen: chosen, weights: weights, limit: limit)
    } catch {
      fatalError("layer \(layer) could not read the experts it routed to: \(error)")
    }
    var y = routed.asType(.float32).sum(axis: -2)
    let hidden = clampedSwiGLU(
      gate: shared.gate(rows).asType(.float32), up: shared.up(rows).asType(.float32),
      limit: limit)
    y = y + shared.down(hidden.asType(rows.dtype)).asType(.float32)
    return y.asType(x.dtype).reshaped(shape)
  }
}

/// torch's softplus: `log(1 + e^x)`, and `x` itself past twenty, where that is all it is.
private func softplusTorch(_ x: MLXArray) -> MLXArray {
  MLX.where(x .> 20, x, log1p(exp(minimum(x, MLXArray(Float(20))))))
}
