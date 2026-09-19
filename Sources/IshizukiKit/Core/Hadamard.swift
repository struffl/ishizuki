// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

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

    self.outputDim = weight.dim(0)
    self.inputDim = weight.dim(1) * (32 / bits)

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
    quantizedMM(
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
    if block > 0 && signs == nil {
      throw BonsaiError.invalidTransform("rotated embedding is missing its sign vector")
    }
  }

  public func callAsFunction(_ ids: MLXArray) -> MLXArray {
    let shape = ids.shape
    let flat = ids.reshaped([-1])
    var out = dequantized(
      weight[flat], scales: scales[flat], biases: biases[flat],
      groupSize: groupSize, bits: bits, mode: .affine)
    out = out.reshaped(shape + [-1]).asType(dtype)
    if block > 0, let signs {
      out = hadamardRotate(out, block: block, signs: signs, inverse: true)
    }
    return out
  }
}
