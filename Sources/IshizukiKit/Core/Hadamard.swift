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
  /// Set for a projection built by ``init(dense:)``: `weight` is the real fp16 checkpoint
  /// weight, not a quantized one, and `scales`/`biases` are unused placeholders.
  public let isDense: Bool
  /// Set for a projection read straight out of a GGUF: the weight is still in GGML blocks and
  /// is decoded inside the kernel rather than before it.
  public let ggml: GGUFBlocks?
  /// Set for a projection read out of an EXL3 pack: the weight stays trellis-coded and is
  /// decoded inside the kernel.
  public let exl3: EXL3Tensor?
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
    self.ggml = nil
    self.exl3 = nil
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
    VerifyMatmul.warm(weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
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
    self.ggml = nil
    self.exl3 = nil
    self.collect = collect
    self.outputDim = weight.dim(0)
    self.inputDim = weight.dim(1)
  }

  /// A projection over a GGUF tensor, still blocked. `weight` holds the raw bytes so that
  /// anything walking this module's storage still sees one array rather than nothing.
  public init(ggml blocks: GGUFBlocks) {
    self.weight = blocks.bytes
    self.scales = MLXArray.ones([1])
    self.biases = MLXArray.zeros([1])
    self.signs = nil
    self.block = 0
    self.groupSize = blocks.type.blockSize
    self.bits = 0
    self.isDense = false
    self.ggml = blocks
    self.exl3 = nil
    self.collect = nil
    self.outputDim = blocks.outputDim
    self.inputDim = blocks.inputDim
  }

  public init(exl3 tensor: EXL3Tensor) {
    self.weight = tensor.trellis
    self.scales = MLXArray.ones([1])
    self.biases = MLXArray.zeros([1])
    self.signs = nil
    self.block = 0
    self.groupSize = 256
    self.bits = tensor.wholeBits
    self.isDense = false
    self.ggml = nil
    self.exl3 = tensor
    self.collect = nil
    self.outputDim = tensor.outputDim
    self.inputDim = tensor.inputDim
  }

  /// Blocks decode inside the matvec while the batch is small enough to pay for the decode
  /// once per block; past that the weight is expanded once and multiplied the ordinary way,
  /// which is still never stored.
  private func ggmlApply(_ h: MLXArray, _ blocks: GGUFBlocks) -> MLXArray {
    let shape = h.shape
    let width = shape[shape.count - 1]
    let rows = h.size / width

    if BonsaiRuntime.useVerifyMatmul, (5...16).contains(rows) {
      let x = h.reshaped([rows, width])
      let pieces = stride(from: 0, to: rows, by: 8).map { start in
        GGMLKernels.matmulFew(
          x[start..<min(start + 8, rows)], blocks: blocks.bytes, type: blocks.type,
          outputDim: blocks.outputDim, rowBlocks: 2)
      }
      if pieces.allSatisfy({ $0 != nil }) {
        let y = pieces.count == 1 ? pieces[0]! : concatenated(pieces.map { $0! }, axis: 0)
        return y.reshaped(Array(shape.dropLast()) + [outputDim])
      }
    }

    if GGMLKernels.matvecBatch.contains(rows),
      let y = GGMLKernels.matvec(
        h.reshaped([rows, width]), blocks: blocks.bytes, type: blocks.type,
        outputDim: blocks.outputDim)
    {
      return y.reshaped(Array(shape.dropLast()) + [outputDim])
    }

    guard
      let weight = GGMLKernels.dequantize(
        blocks: blocks.bytes, type: blocks.type, shape: blocks.shape, dtype: h.dtype)
    else {
      return MLXArray.zeros(Array(shape.dropLast()) + [outputDim], dtype: h.dtype)
    }
    return matmul(h, weight.T)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    if let ggml { return ggmlApply(x, ggml) }
    if let exl3 { return EXL3Kernels.apply(x, exl3) }
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

    return quantized(h)
  }

  private func quantized(_ h: MLXArray) -> MLXArray {
    if BonsaiRuntime.useVerifyMatmul {
      let shape = h.shape
      let width = shape[shape.count - 1]
      let rows = h.size / width
      if VerifyMatmul.supportedRows.contains(rows),
        let y = VerifyMatmul.apply(
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
    if let ggml { return ggmlApply(h, ggml) }
    if let exl3 { return EXL3Kernels.apply(h, exl3) }
    if isDense {
      return matmul(h, weight.T.asType(h.dtype))
    }
    return quantized(h)
  }

  // Output channels are rows of the packed weight, so the half Metal keeps is a row slice.
  public func channels(from start: Int) throws -> PackedLinear {
    guard ggml == nil, exl3 == nil else {
      throw BonsaiError.invalidTransform("a coded projection cannot be split by channel")
    }
    return try PackedLinear(
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
  /// Set for an embedding read straight out of a GGUF, gathered row by row rather than
  /// expanded: the table is far too large to hold decoded.
  public let ggml: GGUFBlocks?
  /// Set for an embedding built by ``init(dense:)``: `weight` holds the real fp16 checkpoint
  /// rows directly, nothing to dequantize.
  public let isDense: Bool
  /// Set for an embedding built by ``init(codes:rowScales:dtype:)``: `weight` holds int8 codes
  /// and `scales` one scale per row.
  public let isRowScaled: Bool

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
    self.ggml = nil
    self.isDense = false
    self.isRowScaled = false
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
    self.ggml = nil
    self.isDense = true
    self.isRowScaled = false
  }

  /// A table of int8 codes with one scale per row, a row being its codes times its scale. An
  /// OrcaSAQ2 pack keeps its embedding this way beside EXL3 layers, at half the bytes of the
  /// bfloat16 table exllamav3 would have left.
  public init(codes: MLXArray, rowScales: MLXArray, dtype: DType = .float16) throws {
    guard codes.ndim == 2, codes.dtype == .int8, rowScales.size == codes.dim(0) else {
      throw BonsaiError.shapeMismatch(
        "an int8 embedding wants [rows, width] codes and a scale per row, got "
          + "\(codes.shape) \(codes.dtype) and \(rowScales.shape)")
    }
    self.weight = codes
    self.scales = rowScales.reshaped([-1])
    self.biases = MLXArray.zeros([1])
    self.signs = nil
    self.block = 0
    self.groupSize = codes.dim(1)
    self.bits = 8
    self.dtype = dtype
    self.ggml = nil
    self.isDense = false
    self.isRowScaled = true
  }

  public init(ggml blocks: GGUFBlocks, dtype: DType = .bfloat16) {
    self.weight = blocks.bytes
    self.scales = MLXArray.ones([1])
    self.biases = MLXArray.zeros([1])
    self.signs = nil
    self.block = 0
    self.groupSize = blocks.type.blockSize
    self.bits = 0
    self.dtype = dtype
    self.ggml = blocks
    self.isDense = false
    self.isRowScaled = false
  }

  public func callAsFunction(_ ids: MLXArray) -> MLXArray {
    let shape = ids.shape
    let flat = ids.reshaped([-1])
    var out: MLXArray
    if let ggml {
      guard
        let rows = GGMLKernels.gather(
          ids: flat, blocks: ggml.bytes, type: ggml.type, inputDim: ggml.inputDim,
          dtype: dtype)
      else { return MLXArray.zeros(shape + [ggml.inputDim], dtype: dtype) }
      return rows.reshaped(shape + [-1])
    }
    if isDense {
      out = weight[flat]
    } else if isRowScaled {
      out =
        weight[flat].asType(.float32) * scales[flat].asType(.float32).expandedDimensions(axis: -1)
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
