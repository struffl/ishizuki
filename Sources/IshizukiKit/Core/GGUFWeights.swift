// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// A GGUF file's tensors as the runtime's modules want them.

import Foundation
import MLX

/// One tensor still in its GGML blocks.
///
/// The bytes are handed to the kernels as they came off disk; nothing decodes until a forward
/// pass asks for it, which is the whole point of reading this format natively.
public struct GGUFBlocks: @unchecked Sendable {
  public let bytes: MLXArray
  public let type: GGMLType
  public let shape: [Int]

  public var outputDim: Int { shape[0] }
  public var inputDim: Int { shape[shape.count - 1] }

  public init(bytes: MLXArray, type: GGMLType, shape: [Int]) {
    self.bytes = bytes
    self.type = type
    self.shape = shape
  }
}

/// Splits a GGUF into the two halves the runtime needs: small tensors it can hold as ordinary
/// arrays, and quantized ones it must keep packed.
///
/// llama.cpp stores the norms, the delta-net's `A_log`, `dt_bias` and its depthwise convolution
/// unquantized, which is what makes this split clean — everything the forward pass wants as a
/// plain array already is one, and everything else is a projection that can go through a
/// kernel.
///
/// Two of llama.cpp's conversion-time folds have to be undone here, because the runtime shares
/// one forward pass with the checkpoints it quantizes itself. The norms are the fold that does
/// *not* need undoing: llama.cpp writes `1 + w` for every `*norm.weight` but the delta-net's
/// own, which is exactly the convention `TensorNaming` produces, so they are read as they lie.
/// `A_log` is the one that does: llama.cpp stores `-exp(A_log)` ready to multiply, and
/// `GatedDeltaNet` takes the exponential itself, so the log is taken back here rather than
/// branching the gate.
public enum GGUFWeights {
  /// Tensors small enough, or awkward enough, to keep dense. A depthwise convolution is
  /// transposed here for the same reason a safetensors pack is: PyTorch stores it
  /// channels-second and MLX wants it channels-last.
  public static func load(file: GGUFFile, dtype: DType = .bfloat16) throws -> WeightStore {
    var dense: [String: MLXArray] = [:]
    var packed: [String: GGUFBlocks] = [:]

    for tensor in file.tensors {
      guard let name = GGUFTensorNaming.canonical(tensor.name) else {
        throw BonsaiError.unsupportedModel(
          "\(tensor.name) has no module in this runtime")
      }
      guard GGMLDequant.supported.contains(tensor.type) else {
        throw BonsaiError.unsupportedModel(
          "\(tensor.name) is \(tensor.type.name), which this runtime cannot read")
      }

      if tensor.type.isQuantized {
        let bytes = try file.data(for: tensor)
        packed[name] = GGUFBlocks(
          bytes: MLXArray([UInt8](bytes)), type: tensor.type, shape: tensor.shape)
        continue
      }

      let values = try GGMLDequant.dequantize(
        try file.data(for: tensor), type: tensor.type, count: tensor.elementCount)
      var array = MLXArray(values, tensor.shape).asType(
        tensor.type == .f32 ? .float32 : dtype)
      if name.hasSuffix(".conv1d.weight") {
        array = array.reshaped([tensor.shape[0], tensor.shape[1], 1])
      }
      if name.hasSuffix(".A_log") {
        array = MLX.log(-array.asType(.float32))
      }
      dense[name] = array
    }

    return WeightStore(arrays: dense, ggml: packed, valueHeadLayout: .tiled)
  }

  /// The tower out of its own file, folded into a store the language model already built.
  ///
  /// Nothing here stays in blocks. The tower is a rounding error beside the language model, it
  /// runs dense in this runtime either way, and `VisionTower` reads every one of these through
  /// the plain array table.
  public static func loadVision(
    file: GGUFFile, into store: WeightStore, dtype: DType = .bfloat16
  ) throws -> WeightStore {
    var dense = store.arrays
    var halves: [Int: MLXArray] = [:]

    for tensor in file.tensors {
      let values = try GGMLDequant.dequantize(
        try file.data(for: tensor), type: tensor.type, count: tensor.elementCount)
      let array = MLXArray(values, tensor.shape).asType(
        tensor.type == .f32 ? .float32 : dtype)

      switch tensor.name {
      case GGUFVisionNaming.patchEmbedding: halves[0] = array
      case GGUFVisionNaming.patchEmbeddingSecond: halves[1] = array
      default:
        guard let name = GGUFVisionNaming.canonical(tensor.name) else {
          throw BonsaiError.unsupportedModel(
            "\(tensor.name) has no module in this runtime's vision tower")
        }
        dense[name] = array
      }
    }

    guard let first = halves[0], let second = halves[1] else {
      throw BonsaiError.missingWeight(
        "the mmproj is missing half of its patch embedding")
    }
    // ggml has no Conv3D, so the converter sliced the temporal axis into two Conv2Ds. Stacking
    // them back gives [out, in, t, h, w] — the layout a checkpoint ships — which then takes the
    // same channels-last relayout every other path applies to it.
    let name = GGUFVisionNaming.prefix + "patch_embed.proj.weight"
    dense[name] = TensorNaming.relayout(name, stacked([first, second], axis: 2))

    return WeightStore(
      arrays: dense, ggml: store.ggmlArrays, valueHeadLayout: store.valueHeadLayout)
  }
}
