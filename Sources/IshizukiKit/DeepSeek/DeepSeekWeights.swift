// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A V4.1 checkpoint's tensors as the modules that multiply by them.

import Foundation
import MLX
import MLXNN

/// A matrix in one of MLX's microscaling formats: mxfp8 for DeepSeek's fp8 weights, mxfp4 for
/// its experts. The words are the checkpoint's own bytes, so nothing was requantized to get here.
public final class MXLinear: Projection, @unchecked Sendable {
  public let words: MLXArray
  public let scales: MLXArray
  public let bits: Int

  public var outputDim: Int { words.dim(-2) }
  public var inputDim: Int { words.dim(-1) * 32 / bits }

  public init(words: MLXArray, scales: MLXArray, bits: Int) {
    self.words = words
    self.scales = scales
    self.bits = bits
  }

  var mode: QuantizationMode { bits == 8 ? .mxfp8 : .mxfp4 }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    quantizedMM(
      x, words, scales: scales, biases: nil, transpose: true, groupSize: 32, bits: bits,
      mode: mode)
  }

  public func dense() -> MLXArray {
    dequantized(
      words, scales: scales, biases: nil, groupSize: 32, bits: bits, mode: mode,
      dtype: .float32)
  }
}

/// The attention's first output projection: one small matrix per group of heads, each reading
/// only its own group's channels. The groups split the rows, not the inputs, so the packed
/// bytes of each row are untouched and the whole thing stays quantized.
public struct GroupedProjection: @unchecked Sendable {
  let dense: MLXArray?
  let words: MLXArray?
  let scales: MLXArray?
  public let groups: Int

  public init(dense: MLXArray, groups: Int) {
    self.dense = dense.reshaped([groups, dense.dim(0) / groups, dense.dim(1)])
    self.words = nil
    self.scales = nil
    self.groups = groups
  }

  public init(words: MLXArray, scales: MLXArray, groups: Int) {
    self.dense = nil
    self.words = words.reshaped([groups, words.dim(0) / groups, words.dim(1)])
    self.scales = scales.reshaped([groups, scales.dim(0) / groups, scales.dim(1)])
    self.groups = groups
  }

  /// `x` is `[rows, groups, width]`; the answer is `[rows, groups, rank]`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let rows = x.dim(0)
    let input = x.expandedDimensions(axis: -2)
    if let dense {
      return matmul(input, dense.asType(x.dtype).swappedAxes(-1, -2)).squeezed(axis: -2)
    }
    let order = broadcast(
      MLXArray(0..<Int32(groups)).expandedDimensions(axis: 0), to: [rows, groups])
    return gatherQuantizedMM(
      input, words!, scales: scales!, biases: nil, rhsIndices: order, transpose: true,
      groupSize: 32, bits: 8, mode: .mxfp8
    ).squeezed(axis: -2)
  }
}

/// One layer's routed experts: gate (`w1`), up (`w3`) and down (`w2`) for every expert.
public protocol DeepSeekExpertBank: Sendable {
  var count: Int { get }
  /// `x` is `[rows, dim]`, `chosen` `[rows, k]`. Returns the down projections `[rows, k, dim]`
  /// of each chosen expert's clamped SwiGLU, each already scaled by its routing weight.
  func run(_ x: MLXArray, chosen: MLXArray, weights: MLXArray, limit: Float) throws -> MLXArray
}

func clampedSwiGLU(gate: MLXArray, up: MLXArray, limit: Float) -> MLXArray {
  guard limit > 0 else { return silu(gate) * up }
  return silu(minimum(gate, MLXArray(limit))) * clip(up, min: -limit, max: limit)
}

/// Every expert in memory at full width, for a golden test.
public struct DenseExpertBank: DeepSeekExpertBank, @unchecked Sendable {
  let gate: MLXArray
  let up: MLXArray
  let down: MLXArray

  public var count: Int { gate.dim(0) }

  public init(gate: MLXArray, up: MLXArray, down: MLXArray) {
    self.gate = gate
    self.up = up
    self.down = down
  }

  public func run(_ x: MLXArray, chosen: MLXArray, weights: MLXArray, limit: Float) -> MLXArray {
    let batched = x.expandedDimensions(axes: [-2, -3])
    func project(_ w: MLXArray, _ input: MLXArray) -> MLXArray {
      gatherMM(input, w.swappedAxes(-1, -2).asType(input.dtype), rhsIndices: chosen)
    }
    let hidden = clampedSwiGLU(
      gate: project(gate, batched).asType(.float32), up: project(up, batched).asType(.float32),
      limit: limit)
    let scaled = hidden * weights.expandedDimensions(axes: [-1, -2])
    return project(down, scaled.asType(x.dtype)).squeezed(axis: -2)
  }
}

/// Every expert in memory as the release's fp4, multiplied where it lies.
public struct MXExpertBank: DeepSeekExpertBank, @unchecked Sendable {
  let gate: (MLXArray, MLXArray)
  let up: (MLXArray, MLXArray)
  let down: (MLXArray, MLXArray)

  public var count: Int { gate.0.dim(0) }

  public init(gate: (MLXArray, MLXArray), up: (MLXArray, MLXArray), down: (MLXArray, MLXArray)) {
    self.gate = gate
    self.up = up
    self.down = down
  }

  public func run(_ x: MLXArray, chosen: MLXArray, weights: MLXArray, limit: Float) -> MLXArray {
    let batched = x.expandedDimensions(axes: [-2, -3])
    func project(_ w: (MLXArray, MLXArray), _ input: MLXArray) -> MLXArray {
      gatherQuantizedMM(
        input, w.0, scales: w.1, biases: nil, rhsIndices: chosen, transpose: true,
        groupSize: 32, bits: 4, mode: .mxfp4)
    }
    let hidden = clampedSwiGLU(
      gate: project(gate, batched).asType(.float32), up: project(up, batched).asType(.float32),
      limit: limit)
    let scaled = hidden * weights.expandedDimensions(axes: [-1, -2])
    return project(down, scaled.asType(x.dtype)).squeezed(axis: -2)
  }
}

/// Where a model's modules come from: a release checkpoint read as it is.
///
/// `dense` widens every fp8 and fp4 matrix to float32, which is how a golden test reads it; the
/// default keeps them in the formats the checkpoint wrote and multiplies them there.
/// `expertSlots` streams each layer's routed experts from the shards into that many slots rather
/// than holding them, which is the only way the release fits a machine with less than 300 GB.
public final class DeepSeekWeights: @unchecked Sendable {
  public let checkpoint: DeepSeekCheckpoint
  public let dense: Bool
  public let compute: DType
  public let expertSlots: Int?

  public init(
    checkpoint: DeepSeekCheckpoint, dense: Bool = false, compute: DType = .bfloat16,
    expertSlots: Int? = nil
  ) {
    self.checkpoint = checkpoint
    self.dense = dense
    self.compute = compute
    self.expertSlots = expertSlots
  }

  public var config: DeepSeekConfig { checkpoint.config }

  public func has(_ name: String) -> Bool { checkpoint.has(name) }

  /// A tensor kept as a plain array: a norm, a bias, a hyper-connection projection. Float32
  /// ones stay float32; the rest arrive at the compute width.
  public func array(_ name: String) throws -> MLXArray {
    let tensor = try checkpoint.tensor(name)
    return settled(tensor.dtype == .float32 ? tensor : tensor.asType(dense ? .float32 : compute))
  }

  public func float32(_ name: String) throws -> MLXArray {
    settled(try checkpoint.tensor(name).asType(.float32))
  }

  /// Converted now rather than at the first forward pass, so the bytes it was read into are
  /// let go as the model loads instead of all being held until it runs.
  private func settled(_ array: MLXArray) -> MLXArray {
    eval(array)
    return array
  }

  private func fp8(_ prefix: String) throws -> (MLXArray, MLXArray)? {
    let weight = try checkpoint.entry(prefix + ".weight")
    guard weight.dtype == "F8_E4M3", checkpoint.has(prefix + ".scale") else { return nil }
    let (words, scales) = DeepSeekFormat.mxfp8(
      codes: try checkpoint.tensor(prefix + ".weight"),
      blockScales: try checkpoint.tensor(prefix + ".scale"))
    eval(words, scales)
    return (words, scales)
  }

  /// A projection, quantized if the checkpoint stored it that way. `wide` keeps a plain
  /// matrix at float32, where the reference promotes one to float32 too.
  public func linear(_ prefix: String, wide: Bool = false) throws -> any Projection {
    if let (words, scales) = try fp8(prefix) {
      let packed = MXLinear(words: words, scales: scales, bits: 8)
      return dense ? DenseLinear(weight: packed.dense(), bias: nil) : packed
    }
    let weight = try checkpoint.tensor(prefix + ".weight")
    let width: DType = dense || wide ? .float32 : compute
    return DenseLinear(weight: settled(weight.asType(width)), bias: nil)
  }

  public func grouped(_ prefix: String, groups: Int) throws -> GroupedProjection {
    if let (words, scales) = try fp8(prefix) {
      guard dense else { return GroupedProjection(words: words, scales: scales, groups: groups) }
      return GroupedProjection(
        dense: MXLinear(words: words, scales: scales, bits: 8).dense(), groups: groups)
    }
    let weight = try checkpoint.tensor(prefix + ".weight")
    return GroupedProjection(dense: weight.asType(dense ? .float32 : compute), groups: groups)
  }

  /// Every routed expert of one layer, stacked along a leading expert axis, or streamed.
  public func experts(_ prefix: String, count: Int) throws -> any DeepSeekExpertBank {
    if let expertSlots, !dense {
      return try StreamedExpertBank(
        checkpoint: checkpoint, prefix: prefix, count: count, slots: expertSlots)
    }
    func stack(_ name: String) throws -> (MLXArray, MLXArray) {
      var words: [MLXArray] = []
      var scales: [MLXArray] = []
      for expert in 0..<count {
        let base = "\(prefix).\(expert).\(name)"
        let (w, s) = DeepSeekFormat.mxfp4(
          pairs: try checkpoint.tensor(base + ".weight"),
          scales: try checkpoint.tensor(base + ".scale"))
        words.append(w)
        scales.append(s)
      }
      return (stacked(words, axis: 0), stacked(scales, axis: 0))
    }
    let gate = try stack("w1")
    let up = try stack("w3")
    let down = try stack("w2")
    guard dense else { return MXExpertBank(gate: gate, up: up, down: down) }
    func widen(_ pair: (MLXArray, MLXArray)) -> MLXArray {
      dequantized(
        pair.0, scales: pair.1, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4,
        dtype: .float32)
    }
    return DenseExpertBank(gate: widen(gate), up: widen(up), down: widen(down))
  }
}
