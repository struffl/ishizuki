// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXFast
import MLXNN

public final class Attention: @unchecked Sendable {
  private let qProj: PackedLinear
  private let kProj: PackedLinear
  private let vProj: PackedLinear
  private let oProj: PackedLinear
  private let qNorm: MLXArray
  private let kNorm: MLXArray
  private let rope: RotaryEmbedding

  private let numHeads: Int
  private let numKeyValueHeads: Int
  private let headDim: Int
  private let scale: Float
  private let normEps: Float
  private let outputGate: Bool

  public init(
    config: BonsaiConfig.TextConfig, layer: Int,
    factory: PackedModuleFactory, store: WeightStore, rope: RotaryEmbedding
  ) throws {
    let prefix = "model.layers.\(layer).self_attn"
    let tensorPrefix = factory.tensorPrefix + prefix

    self.numHeads = config.numAttentionHeads
    self.numKeyValueHeads = config.numKeyValueHeads
    self.headDim = config.headDim
    self.scale = 1.0 / Float(config.headDim).squareRoot()
    self.normEps = config.rmsNormEps
    self.outputGate = config.attnOutputGate ?? false
    self.rope = rope

    self.qProj = try factory.linear(prefix + ".q_proj")
    self.kProj = try factory.linear(prefix + ".k_proj")
    self.vProj = try factory.linear(prefix + ".v_proj")
    self.oProj = try factory.linear(prefix + ".o_proj")
    self.qNorm = try store(tensorPrefix + ".q_norm.weight")
    self.kNorm = try store(tensorPrefix + ".k_norm.weight")
  }

  public func callAsFunction(
    _ x: MLXArray, mask: MLXArray?, cache: AttentionKVCache?, positions: MLXArray? = nil
  ) -> MLXArray {
    let b = x.dim(0)
    let l = x.dim(1)

    var queries: MLXArray
    var gate: MLXArray?

    if outputGate {
      let projected = qProj(x).reshaped([b, l, numHeads, 2 * headDim])
      queries = projected[0..., 0..., 0..., ..<headDim]
      gate = projected[0..., 0..., 0..., headDim...].reshaped([b, l, numHeads * headDim])
    } else {
      queries = qProj(x).reshaped([b, l, numHeads, headDim])
    }

    var keys = kProj(x).reshaped([b, l, numKeyValueHeads, headDim])
    var values = vProj(x).reshaped([b, l, numKeyValueHeads, headDim])

    queries = MLXFast.rmsNorm(queries, weight: qNorm.asType(queries.dtype), eps: normEps)
      .transposed(0, 2, 1, 3)
    keys = MLXFast.rmsNorm(keys, weight: kNorm.asType(keys.dtype), eps: normEps)
      .transposed(0, 2, 1, 3)
    values = values.transposed(0, 2, 1, 3)

    let offset = cache?.offset ?? 0
    if let positions {
      queries = rope(queries, positions: positions)
      keys = rope(keys, positions: positions)
    } else {
      queries = rope(queries, offset: offset)
      keys = rope(keys, offset: offset)
    }

    var output: MLXArray
    if let cache {
      switch cache.appendForAttention(keys: keys, values: values) {
      case .dense(let cachedKeys, let cachedValues):
        output = MLXFast.scaledDotProductAttention(
          queries: queries, keys: cachedKeys, values: cachedValues,
          scale: scale, mask: mask)
      case .quantized(let qKeys, let qValues, let groupSize, let keyBits, let valueBits):
        output = quantizedAttention(
          queries: queries, keys: qKeys, values: qValues,
          window: (cache as? QuantizedKVCache)?.window,
          groupSize: groupSize, keyBits: keyBits, valueBits: valueBits, mask: mask)
      }
    } else {
      output = MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }
    output = output.transposed(0, 2, 1, 3).reshaped([b, l, numHeads * headDim])

    if let gate {
      output = output * sigmoid(gate)
    }
    return oProj(output)
  }

  private func quantizedAttention(
    queries: MLXArray,
    keys: (MLXArray, MLXArray, MLXArray),
    values: (MLXArray, MLXArray, MLXArray),
    window: (keys: MLXArray, values: MLXArray)?,
    groupSize: Int, keyBits: Int, valueBits: Int,
    mask: MLXArray?
  ) -> MLXArray {
    let b = queries.dim(0)
    let l = queries.dim(2)
    let d = queries.dim(3)
    let kvHeads = keys.0.dim(1)
    let repeats = numHeads / kvHeads

    var q = queries * scale
    var qKeys = keys
    var qValues = values

    if repeats > 1 {
      q = q.reshaped([b, kvHeads, repeats, l, d])
      qKeys = (
        keys.0.expandedDimensions(axis: 2), keys.1.expandedDimensions(axis: 2),
        keys.2.expandedDimensions(axis: 2)
      )
      qValues = (
        values.0.expandedDimensions(axis: 2), values.1.expandedDimensions(axis: 2),
        values.2.expandedDimensions(axis: 2)
      )
    }

    var scores = quantizedMM(
      q, qKeys.0, scales: qKeys.1, biases: qKeys.2,
      transpose: true, groupSize: groupSize, bits: keyBits, mode: .affine)

    var windowScores: MLXArray?
    if let window {
      var windowKeys = window.keys
      if repeats > 1 { windowKeys = windowKeys.expandedDimensions(axis: 2) }
      windowScores = matmul(q, windowKeys.swappedAxes(-1, -2))
    }

    if let mask {
      let quantizedLength = scores.dim(-1)
      scores = scores + mask[.ellipsis, 0..<quantizedLength]
      if var tail = windowScores {
        tail = tail + mask[.ellipsis, quantizedLength...]
        windowScores = tail
      }
    }

    let combined = windowScores.map { concatenated([scores, $0], axis: -1) } ?? scores
    let weights = softmax(combined, axis: -1, precise: true)

    let quantizedLength = scores.dim(-1)
    var output = quantizedMM(
      weights[.ellipsis, 0..<quantizedLength], qValues.0,
      scales: qValues.1, biases: qValues.2,
      transpose: false, groupSize: groupSize, bits: valueBits, mode: .affine)

    if let window {
      var windowValues = window.values
      if repeats > 1 { windowValues = windowValues.expandedDimensions(axis: 2) }
      output = output + matmul(weights[.ellipsis, quantizedLength...], windowValues)
    }

    return repeats > 1 ? output.reshaped([b, numHeads, l, d]) : output
  }
}

public final class MLP: @unchecked Sendable {
  private let gateProj: PackedLinear
  private let upProj: PackedLinear
  private let downProj: PackedLinear
  private let split: Split?

  private struct Split {
    let gate: ANESlice
    let up: ANESlice
    let gateTail: PackedLinear
    let upTail: PackedLinear
  }

  public init(layer: Int, factory: PackedModuleFactory) throws {
    let prefix = "model.layers.\(layer).mlp"
    self.gateProj = try factory.linear(prefix + ".gate_proj")
    self.upProj = try factory.linear(prefix + ".up_proj")
    self.downProj = try factory.linear(prefix + ".down_proj")
    self.split = MLP.split(layer: layer, gate: gateProj, up: upProj)
  }

  // gate and up read the same activation, so they share one rotation — but only once the pack
  // says so, rather than on the assumption that a layer's modules always agree.
  private static func split(layer: Int, gate: PackedLinear, up: PackedLinear) -> Split? {
    guard let bank = BonsaiRuntime.aneBank,
      gate.block == up.block,
      sharesRotation(gate, up),
      let gateSlice = bank.slice("\(layer).mlp.gate_proj"),
      let upSlice = bank.slice("\(layer).mlp.up_proj"),
      gateSlice.inputDim == gate.inputDim, upSlice.inputDim == up.inputDim,
      gateSlice.outputDim < gate.outputDim, upSlice.outputDim < up.outputDim,
      let gateTail = try? gate.channels(from: gateSlice.outputDim),
      let upTail = try? up.channels(from: upSlice.outputDim)
    else { return nil }
    return Split(gate: gateSlice, up: upSlice, gateTail: gateTail, upTail: upTail)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    if let split, x.ndim == 3, x.dim(0) * x.dim(1) == split.gate.rows {
      return hybrid(x, split)
    }
    return downProj(silu(gateProj(x)) * upProj(x))
  }

  private func hybrid(_ x: MLXArray, _ split: Split) -> MLXArray {
    let shape = x.shape
    let rotated = gateProj.rotate(x.reshaped([split.gate.rows, shape[2]]))
    let gatePending = split.gate.dispatch(rotated)
    let upPending = split.up.dispatch(rotated)

    let gateTail = split.gateTail.applyRotated(rotated)
    let upTail = split.upTail.applyRotated(rotated)
    eval(gateTail, upTail)

    guard let gateHead = try? gatePending.wait(), let upHead = try? upPending.wait() else {
      return downProj(silu(gateProj(x)) * upProj(x))
    }
    let gate = concatenated([gateHead, gateTail], axis: -1)
    let up = concatenated([upHead, upTail], axis: -1)
    let activated = silu(gate) * up
    return downProj(activated.reshaped([shape[0], shape[1], activated.dim(-1)]))
  }
}

func sharesRotation(_ a: PackedLinear, _ b: PackedLinear) -> Bool {
  guard a.block == b.block else { return false }
  guard let left = a.signs, let right = b.signs else { return a.signs == nil && b.signs == nil }
  guard left.shape == right.shape else { return false }
  return (left .!= right).sum().item(Int.self) == 0
}

public func causalMask(length: Int, offset: Int, dtype: DType) -> MLXArray? {
  guard length > 1 else { return nil }
  let queryPositions = MLXArray(0..<length) + offset
  let keyPositions = MLXArray(0..<(offset + length))
  let allowed =
    queryPositions.expandedDimensions(axis: 1)
    .>= keyPositions
    .expandedDimensions(axis: 0)
  return MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.greatestFiniteMagnitude))
    .asType(dtype)
}
