// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

@inlinable
public func hadamardRotate(
  _ x: MLXArray, block: Int, signs: MLXArray, inverse: Bool = false
) -> MLXArray {
  let shape = x.shape
  let dtype = x.dtype
  precondition(
    shape[shape.count - 1] % block == 0,
    "Hadamard block \(block) does not divide activation width \(shape[shape.count - 1])")

  var y = x.asType(.float32)
  if !inverse { y = y * signs }
  y = hadamardTransform(y.reshaped([-1, block]), scale: 1.0 / Float(block).squareRoot())
    .reshaped(shape)
  if inverse { y = y * signs }
  return y.asType(dtype)
}

public final class PackedLinear: @unchecked Sendable {
  public let weight: MLXArray
  public let scales: MLXArray
  public let biases: MLXArray
  public let signs: MLXArray?
  public let block: Int
  public let groupSize: Int
  public let bits: Int
  /// Set for a projection built by ``init(dense:)``: `weight` is the real fp16 checkpoint
  /// weight, not a quantized one, and `scales`/`biases` are unused placeholders.
  public let isDense: Bool
  /// Calibration's hook onto this projection's exact input, called before every forward pass
  /// when this projection is dense. Never set for a quantized projection.
  private let collect: (@Sendable (MLXArray) -> Void)?

  public let inputDim: Int
  public let outputDim: Int

  public init(
    weight: MLXArray, scales: MLXArray, biases: MLXArray,
    signs: MLXArray?, block: Int, groupSize: Int = 128, bits: Int = 2
  ) throws {
    self.weight = weight
    self.scales = scales
    self.biases = biases
    self.signs = signs
    self.block = block
    self.groupSize = groupSize
    self.bits = bits
    self.isDense = false
    self.collect = nil

    self.outputDim = weight.dim(0)
    // MLX packs quantized weights densely across 32-bit words, so a width that does not divide
    // 32 — 3, 5 and 6 bits all appear in imatrix packs — has no whole number of values per
    // word. The group structure does carry it exactly: one scale per group of inputs.
    self.inputDim = scales.dim(1) * groupSize

    if block > 0 {
      guard let signs else {
        throw BonsaiError.invalidTransform("rotated module is missing its sign vector")
      }
      guard signs.dim(0) == inputDim else {
        throw BonsaiError.shapeMismatch(
          "sign vector is \(signs.dim(0)) wide, expected \(inputDim)")
      }
      guard inputDim % block == 0 else {
        throw BonsaiError.invalidTransform(
          "Hadamard block \(block) does not divide input width \(inputDim)")
      }
    }
  }

  /// A projection over the real, unquantized checkpoint weight — what calibration runs
  /// against, since it needs the exact forward the about-to-be-quantized model produces, not
  /// an approximation of it. Every other consumer of `PackedLinear` (Attention, MLP,
  /// GatedDeltaNet) is unmodified: this is the same type, just backed by real weights instead
  /// of packed ones, so the calibration forward pass is the production forward pass.
  public init(dense weight: MLXArray, collect: (@Sendable (MLXArray) -> Void)? = nil) {
    self.weight = weight
    self.scales = MLXArray.ones([weight.dim(0), 1])
    self.biases = MLXArray.zeros([weight.dim(0), 1])
    self.signs = nil
    self.block = 0
    self.groupSize = weight.dim(1)
    self.bits = 16
    self.isDense = true
    self.collect = collect
    self.outputDim = weight.dim(0)
    self.inputDim = weight.dim(1)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = x
    if block > 0, let signs {
      if BonsaiRuntime.useFusedHadamard,
        let fused = FusedHadamard.apply(h, block: block, signs: signs)
      {
        h = fused
      } else {
        h = hadamardRotate(h, block: block, signs: signs, inverse: false)
      }
    }

    if isDense {
      collect?(h)
      return matmul(h, weight.T.asType(h.dtype))
    }

    if BonsaiRuntime.useQMVWide {
      let shape = h.shape
      let width = shape[shape.count - 1]
      let rows = h.size / width
      if QMVWide.supportedBatch.contains(rows),
        let y = QMVWide.apply(
          h.reshaped([rows, width]), weight, scales: scales, biases: biases,
          groupSize: groupSize, bits: bits)
      {
        return y.reshaped(Array(shape.dropLast()) + [outputDim])
      }
    }

    return quantizedMM(
      h, weight, scales: scales, biases: biases,
      transpose: true, groupSize: groupSize, bits: bits, mode: .affine)
  }

  // A split projection rotates its input once and hands the same activation to both halves, so
  // the rotation and the matmul are reachable on their own.
  public func rotate(_ x: MLXArray) -> MLXArray {
    guard block > 0, let signs else { return x }
    if BonsaiRuntime.useFusedHadamard,
      let fused = FusedHadamard.apply(x, block: block, signs: signs)
    {
      return fused
    }
    return hadamardRotate(x, block: block, signs: signs, inverse: false)
  }

  public func applyRotated(_ h: MLXArray) -> MLXArray {
    if isDense {
      return matmul(h, weight.T.asType(h.dtype))
    }
    return quantizedMM(
      h, weight, scales: scales, biases: biases,
      transpose: true, groupSize: groupSize, bits: bits, mode: .affine)
  }

  // Output channels are rows of the packed weight, so the half Metal keeps is a row slice.
  public func channels(from start: Int) throws -> PackedLinear {
    try PackedLinear(
      weight: weight[start...], scales: scales[start...], biases: biases[start...],
      signs: signs, block: block, groupSize: groupSize, bits: bits)
  }
}

public final class PackedEmbedding: @unchecked Sendable {
  public let weight: MLXArray
  public let scales: MLXArray
  public let biases: MLXArray
  public let signs: MLXArray?
  public let block: Int
  public let groupSize: Int
  public let bits: Int
  public let dtype: DType
  /// Set for an embedding built by ``init(dense:)``: `weight` holds the real fp16 checkpoint
  /// rows directly, nothing to dequantize.
  public let isDense: Bool

  public init(
    weight: MLXArray, scales: MLXArray, biases: MLXArray,
    signs: MLXArray?, block: Int, groupSize: Int = 128, bits: Int = 2,
    dtype: DType = .float16
  ) throws {
    self.weight = weight
    self.scales = scales
    self.biases = biases
    self.signs = signs
    self.block = block
    self.groupSize = groupSize
    self.bits = bits
    self.dtype = dtype
    self.isDense = false
    if block > 0 && signs == nil {
      throw BonsaiError.invalidTransform("rotated embedding is missing its sign vector")
    }
  }

  /// The real, unquantized embedding table — what calibration reads from, since the token
  /// embeddings feed everything downstream and have no meaningful per-channel importance of
  /// their own to weight (a row is gathered by token id, not consumed as an input channel).
  public init(dense weight: MLXArray, dtype: DType = .float16) {
    self.weight = weight
    self.scales = MLXArray.ones([weight.dim(0), 1])
    self.biases = MLXArray.zeros([weight.dim(0), 1])
    self.signs = nil
    self.block = 0
    self.groupSize = weight.dim(1)
    self.bits = 16
    self.dtype = dtype
    self.isDense = true
  }

  public func callAsFunction(_ ids: MLXArray) -> MLXArray {
    let shape = ids.shape
    let flat = ids.reshaped([-1])
    var out: MLXArray
    if isDense {
      out = weight[flat]
    } else {
      out = dequantized(
        weight[flat], scales: scales[flat], biases: biases[flat],
        groupSize: groupSize, bits: bits, mode: .affine)
    }
    out = out.reshaped(shape + [-1]).asType(dtype)
    if block > 0, let signs {
      out = hadamardRotate(out, block: block, signs: signs, inverse: true)
    }
    return out
  }
}
