// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A sparse feed-forward block: a router, a bank of experts, and one shared expert.

import Foundation
import MLX
import MLXNN

/// The feed-forward half of a decoder layer, dense or sparse.
public protocol FeedForward: Sendable {
  func callAsFunction(_ x: MLXArray) -> MLXArray
}

extension MLP: FeedForward {}

/// Every expert's weights for one projection, stacked into a single quantized tensor.
///
/// The experts are never unpacked. `gatherQuantizedMM` reads only the rows the router asked
/// for, straight out of the stack, which is what keeps a 64-expert layer to the cost of the 16
/// it actually uses.
struct StackedExperts: @unchecked Sendable {
  let weight: MLXArray
  let scales: MLXArray
  let biases: MLXArray
  let groupSize: Int
  let bits: Int

  init(store: WeightStore, prefix: String, quant: BonsaiConfig.ModuleQuant) throws {
    self.weight = try store(prefix + ".weight")
    self.scales = try store(prefix + ".scales")
    self.biases = try store(prefix + ".biases")
    self.groupSize = quant.groupSize
    self.bits = quant.bits
  }

  var expertCount: Int { weight.dim(0) }

  func callAsFunction(_ x: MLXArray, _ experts: MLXArray) -> MLXArray {
    gatherQuantizedMM(
      x, weight, scales: scales, biases: biases, rhsIndices: experts,
      transpose: true, groupSize: groupSize, bits: bits)
  }
}

public final class MoEBlock: FeedForward, @unchecked Sendable {
  private let router: PackedLinear
  private let gateExperts: StackedExperts
  private let upExperts: StackedExperts
  private let downExperts: StackedExperts
  private let shared: MLP?
  private let sharedGate: PackedLinear?
  private let topK: Int
  private let normalizeWeights: Bool

  public init(
    config: BonsaiConfig.TextConfig, layer: Int, factory: PackedModuleFactory,
    store: WeightStore, path: String? = nil
  ) throws {
    let module = (path ?? "model.layers.\(layer)") + ".mlp"
    let prefix = factory.tensorPrefix + module

    guard let experts = config.numExperts, let used = config.numExpertsPerTok, experts > 0 else {
      throw BonsaiError.unsupportedModel("layer \(layer) is sparse but declares no experts")
    }
    self.topK = min(used, experts)
    self.normalizeWeights = config.normTopkProb ?? true

    self.router = try factory.linear(module + ".gate")
    guard router.outputDim == experts else {
      throw BonsaiError.shapeMismatch(
        "the router in layer \(layer) scores \(router.outputDim) experts, not \(experts)")
    }

    let quant = factory.quant(for: module + ".switch_mlp.gate_proj")
    self.gateExperts = try StackedExperts(
      store: store, prefix: prefix + ".switch_mlp.gate_proj", quant: quant)
    self.upExperts = try StackedExperts(
      store: store, prefix: prefix + ".switch_mlp.up_proj", quant: quant)
    self.downExperts = try StackedExperts(
      store: store, prefix: prefix + ".switch_mlp.down_proj",
      quant: factory.quant(for: module + ".switch_mlp.down_proj"))
    guard gateExperts.expertCount == experts else {
      throw BonsaiError.shapeMismatch(
        "layer \(layer) stacks \(gateExperts.expertCount) experts, not \(experts)")
    }

    // A checkpoint may route without one, so the shared branch is taken only when it ships.
    if store.has(prefix + ".shared_expert.gate_proj.weight") {
      self.shared = try MLP(prefix: module + ".shared_expert", layer: layer, factory: factory)
      self.sharedGate = try factory.linear(module + ".shared_expert_gate")
    } else {
      self.shared = nil
      self.sharedGate = nil
    }
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Softmax over every expert first, then the top ones — the weights a token spends are
    // shares of the whole bank, not of the slice it kept.
    let gates = softmax(router(x).asType(.float32), axis: -1, precise: true)
    let chosen = argPartition(gates, kth: gates.dim(-1) - topK, axis: -1)[
      .ellipsis, (gates.dim(-1) - topK)...]
    var scores = takeAlong(gates, chosen, axis: -1)
    if normalizeWeights {
      scores = scores / scores.sum(axis: -1, keepDims: true)
    }

    let routed = expert(x, chosen)
    var y = (routed * scores.expandedDimensions(axis: -1).asType(routed.dtype)).sum(axis: -2)

    if let shared, let sharedGate {
      y = y + sigmoid(sharedGate(x)) * shared(x)
    }
    return y
  }

  /// One SwiGLU per selected expert. The two extra axes are what `gatherQuantizedMM` reads the
  /// token against each of its experts in turn.
  private func expert(_ x: MLXArray, _ chosen: MLXArray) -> MLXArray {
    let batched = x.expandedDimensions(axes: [-2, -3])
    let up = upExperts(batched, chosen)
    let gate = gateExperts(batched, chosen)
    return downExperts(silu(gate) * up, chosen).squeezed(axis: -2)
  }
}
