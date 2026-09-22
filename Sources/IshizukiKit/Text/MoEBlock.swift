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

  init(weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int) {
    self.weight = weight
    self.scales = scales
    self.biases = biases
    self.groupSize = groupSize
    self.bits = bits
  }

  init(store: WeightStore, prefix: String, quant: BonsaiConfig.ModuleQuant) throws {
    self.init(
      weight: try store(prefix + ".weight"), scales: try store(prefix + ".scales"),
      biases: try store(prefix + ".biases"), groupSize: quant.groupSize, bits: quant.bits)
  }

  var expertCount: Int { weight.dim(0) }

  func callAsFunction(_ x: MLXArray, _ experts: MLXArray) -> MLXArray {
    gatherQuantizedMM(
      x, weight, scales: scales, biases: biases, rhsIndices: experts,
      transpose: true, groupSize: groupSize, bits: bits)
  }
}

/// Where a layer's experts come from: all of them in memory, or a few of them at a time.
protocol ExpertSource: Sendable {
  var expertCount: Int { get }
  /// Runs the chosen experts' SwiGLU over `x`, whose shape is the routed one the block builds.
  func swiglu(_ x: MLXArray, chosen: MLXArray, groupSize: Int, bits: Int) throws -> MLXArray
}

public final class MoEBlock: FeedForward, @unchecked Sendable {
  /// Read as whatever the pack holds: a router is a handful of rows against the bank it
  /// scores, small enough that a converter may well have left it at full width.
  private let router: any Projection
  private let experts: any ExpertSource
  private let groupSize: Int
  private let bits: Int
  private let shared: MLP?
  private let sharedGate: (any Projection)?
  private let topK: Int
  private let normalizeWeights: Bool
  private let layer: Int

  public init(
    config: BonsaiConfig.TextConfig, layer: Int, factory: PackedModuleFactory,
    store: WeightStore, path: String? = nil
  ) throws {
    let module = (path ?? "model.layers.\(layer)") + ".mlp"
    let prefix = factory.tensorPrefix + module
    self.layer = layer

    guard let experts = config.numExperts, let used = config.numExpertsPerTok, experts > 0 else {
      throw BonsaiError.unsupportedModel("layer \(layer) is sparse but declares no experts")
    }
    self.topK = min(used, experts)
    self.normalizeWeights = config.normTopkProb ?? true

    self.router = try factory.projection(module + ".gate")
    guard router.outputDim == experts else {
      throw BonsaiError.shapeMismatch(
        "the router in layer \(layer) scores \(router.outputDim) experts, not \(experts)")
    }

    let quant = factory.quant(for: module + ".switch_mlp.gate_proj")
    self.groupSize = quant.groupSize
    self.bits = quant.bits

    if let streamed = store.experts(layer: layer) {
      self.experts = StreamedExperts(store: streamed)
    } else if factory.dense {
      self.experts = try DenseExperts(store: store, prefix: prefix + ".switch_mlp")
    } else {
      self.experts = try ResidentExperts(
        gate: StackedExperts(
          store: store, prefix: prefix + ".switch_mlp.gate_proj", quant: quant),
        up: StackedExperts(store: store, prefix: prefix + ".switch_mlp.up_proj", quant: quant),
        down: StackedExperts(
          store: store, prefix: prefix + ".switch_mlp.down_proj",
          quant: factory.quant(for: module + ".switch_mlp.down_proj")))
    }
    guard self.experts.expertCount == experts else {
      throw BonsaiError.shapeMismatch(
        "layer \(layer) holds \(self.experts.expertCount) experts, not \(experts)")
    }

    // A checkpoint may route without one, so the shared branch is taken only when it ships.
    if store.has(prefix + ".shared_expert.gate_proj.weight") {
      self.shared = try MLP(prefix: module + ".shared_expert", layer: layer, factory: factory)
      self.sharedGate = try factory.projection(module + ".shared_expert_gate")
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

    // Sixteen experts are summed per token, so the weighting and the sum stay in float32 and
    // narrow once at the end. Folding the weights into the experts' own dtype first loses more
    // than it saves.
    // A forward pass cannot throw, and an expert that failed to load has no sane stand-in:
    // zeros would be a confident wrong answer rather than a stopped one.
    let routed: MLXArray
    do {
      routed = try experts.swiglu(x, chosen: chosen, groupSize: groupSize, bits: bits)
    } catch {
      fatalError("layer \(layer) could not read the experts it routed to: \(error)")
    }
    var y =
      (routed.asType(.float32) * scores.expandedDimensions(axis: -1))
      .sum(axis: -2)

    if let shared, let sharedGate {
      y = y + (sigmoid(sharedGate(x)) * shared(x)).asType(.float32)
    }
    return y.asType(x.dtype)
  }

}

/// Every expert in memory, which is what a machine with room for them should do.
struct ResidentExperts: ExpertSource {
  let gate: StackedExperts
  let up: StackedExperts
  let down: StackedExperts

  var expertCount: Int { gate.expertCount }

  /// The two extra axes are what `gatherQuantizedMM` reads a token against each of its experts.
  func swiglu(_ x: MLXArray, chosen: MLXArray, groupSize: Int, bits: Int) -> MLXArray {
    let batched = x.expandedDimensions(axes: [-2, -3])
    return down(silu(gate(batched, chosen)) * up(batched, chosen), chosen).squeezed(axis: -2)
  }
}

/// Every expert in memory and unquantized.
///
/// This is what a checkpoint on its way through quantization runs, and what a golden test
/// compares against: the same routing and the same SwiGLU as the packed path, over the
/// weights as they were, so a difference between the two is the quantization and nothing else.
struct DenseExperts: ExpertSource, @unchecked Sendable {
  let gate: MLXArray
  let up: MLXArray
  let down: MLXArray

  var expertCount: Int { gate.dim(0) }

  init(store: WeightStore, prefix: String) throws {
    self.gate = try store(prefix + ".gate_proj.weight")
    self.up = try store(prefix + ".up_proj.weight")
    self.down = try store(prefix + ".down_proj.weight")
  }

  func swiglu(_ x: MLXArray, chosen: MLXArray, groupSize: Int, bits: Int) -> MLXArray {
    let batched = x.expandedDimensions(axes: [-2, -3])
    func project(_ weight: MLXArray, _ input: MLXArray) -> MLXArray {
      gatherMM(input, weight.swappedAxes(-1, -2), rhsIndices: chosen)
    }
    return project(down, silu(project(gate, batched)) * project(up, batched))
      .squeezed(axis: -2)
  }
}

/// The experts a token asked for, read into slots first.
///
/// Routing has to come back to the CPU here — the file cannot be read until the router has
/// said what to read — so this is the one place a sparse layer stops being a pure graph.
///
/// A prefill chunk routes to far more experts than a decode step, and usually to every expert
/// a layer has, so the tokens are run in groups that fit the slots rather than all at once.
/// Grouping changes no arithmetic: a token is read against its own experts either way.
struct StreamedExperts: ExpertSource {
  let store: ExpertStore

  var expertCount: Int { store.layout.expertCount }

  func swiglu(_ x: MLXArray, chosen: MLXArray, groupSize: Int, bits: Int) throws -> MLXArray {
    eval(chosen)
    let topK = chosen.dim(-1)
    let width = x.dim(-1)
    let leading = Array(x.shape.dropLast())
    let rows = leading.reduce(1, *)
    let asked = chosen.asArray(Int32.self).map(Int.init)

    let flatX = x.reshaped([rows, width])
    let flatChosen = chosen.reshaped([rows, topK])

    var pieces: [MLXArray] = []
    var start = 0
    while start < rows {
      var wanted: Set<Int> = []
      var end = start
      while end < rows {
        let next = wanted.union(asked[(end * topK)..<((end + 1) * topK)])
        if next.count > store.slotCount, end > start { break }
        wanted = next
        end += 1
      }
      let piece = try run(
        flatX[start..<end], chosen: flatChosen[start..<end], groupSize: groupSize, bits: bits)
      // The slots are one buffer read over and over, and a piece is only a view onto it until
      // it is evaluated. Letting the next group land first would rewrite this group's weights
      // underneath it.
      eval(piece)
      pieces.append(piece)
      start = end
    }

    return concatenated(pieces, axis: 0).reshaped(leading + [topK, width])
  }

  /// One group of tokens, small enough that every expert they route to is in a slot at once.
  private func run(
    _ x: MLXArray, chosen: MLXArray, groupSize: Int, bits: Int
  ) throws -> MLXArray {
    let slots = try store.residency(of: chosen.asArray(Int32.self).map(Int.init))
    let placed = MLXArray(slots.map { Int32($0) }, chosen.shape)

    func project(_ name: String, _ input: MLXArray) throws -> MLXArray {
      gatherQuantizedMM(
        input, try store.array(name + ".weight"),
        scales: try store.array(name + ".scales"),
        biases: try store.array(name + ".biases"),
        rhsIndices: placed, transpose: true, groupSize: groupSize, bits: bits)
    }

    let batched = x.expandedDimensions(axes: [-2, -3])
    let hidden = silu(try project("gate_proj", batched)) * (try project("up_proj", batched))
    return try project("down_proj", hidden).squeezed(axis: -2)
  }
}
