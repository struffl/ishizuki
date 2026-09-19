// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// llama.cpp's clip container: the vision tower as its own file, in its own layout.

import Foundation

/// The tower travels separately in this format. `convert_hf_to_gguf.py` writes the language
/// model and the vision encoder as two files, so a picture needs the `mmproj-*.gguf` that was
/// split from the same checkpoint as the model beside it.
public struct GGUFVision: Sendable {
  public let config: BonsaiConfig.VisionConfig

  /// The projector this runtime can run. llava, gemma and the rest put different modules
  /// between the encoder and the language model; only Qwen3-VL's merger is built here.
  public static let supportedProjector = "qwen3vl"

  public init(file: GGUFFile) throws {
    guard file["clip.has_vision_encoder"]?.boolValue == true else {
      throw BonsaiError.unsupportedModel("this GGUF carries no vision encoder")
    }
    let projector =
      file["clip.vision.projector_type"]?.stringValue
      ?? file["clip.projector_type"]?.stringValue
    guard projector == Self.supportedProjector else {
      throw BonsaiError.unsupportedModel(
        "projector \(projector ?? "unnamed") is not \(Self.supportedProjector)")
    }

    // An mmproj that carries deepstack layers is a different tower: it feeds intermediate
    // blocks back into the language model, which this one does not do. Loading it would drop
    // those tensors and quietly change what the model sees.
    if let deepstack = file["clip.vision.is_deepstack_layers"]?.intArray,
      deepstack.contains(where: { $0 != 0 })
    {
      throw BonsaiError.unsupportedModel(
        "this tower has deepstack layers, which this runtime does not run")
    }

    func required(_ key: String) throws -> Int {
      guard let value = file["clip.vision." + key]?.intValue else {
        throw BonsaiError.unsupportedModel("the mmproj has no clip.vision.\(key)")
      }
      return value
    }

    let patchSize = try required("patch_size")
    let imageSize = try required("image_size")
    guard patchSize > 0, imageSize % patchSize == 0 else {
      throw BonsaiError.unsupportedModel(
        "image_size \(imageSize) is not a whole number of \(patchSize)-pixel patches")
    }
    let side = imageSize / patchSize

    self.config = BonsaiConfig.VisionConfig(
      depth: try required("block_count"),
      hiddenSize: try required("embedding_length"),
      intermediateSize: try required("feed_forward_length"),
      numHeads: try required("attention.head_count"),
      // Neither is written to the file: llama.cpp's clip reader assumes RGB, and the converter
      // refuses any temporal patch size but two when it splits the Conv3D.
      inChannels: 3,
      patchSize: patchSize,
      temporalPatchSize: 2,
      spatialMergeSize: file["clip.vision.spatial_merge_size"]?.intValue ?? 1,
      outHiddenSize: try required("projection_dim"),
      // The position grid is square by construction, which is how the converter recovers
      // image_size from it in the first place.
      numPositionEmbeddings: side * side,
      hiddenAct: "gelu_pytorch_tanh",
      deepstackVisualIndexes: [])
  }
}

/// Clip's tensor names against this runtime's.
public enum GGUFVisionNaming {
  public static let prefix = "vision_tower."

  /// The halves of the patch embedding, which llama.cpp stores as two Conv2Ds because ggml has
  /// no Conv3D. They are fused back before the tower sees them.
  public static let patchEmbedding = "v.patch_embd.weight"
  public static let patchEmbeddingSecond = "v.patch_embd.weight.1"

  public static func canonical(_ name: String) -> String? {
    guard let (stem, suffix) = split(name) else { return nil }

    if let mapped = fixed[stem] { return prefix + mapped + suffix }

    guard stem.hasPrefix("v.blk.") else { return nil }
    let parts = stem.split(separator: ".", maxSplits: 3, omittingEmptySubsequences: false)
    guard parts.count == 4, let layer = Int(parts[2]), let module = block[String(parts[3])] else {
      return nil
    }
    return "\(prefix)blocks.\(layer).\(module)\(suffix)"
  }

  private static func split(_ name: String) -> (stem: String, suffix: String)? {
    for suffix in [".weight", ".bias"] where name.hasSuffix(suffix) {
      return (String(name.dropLast(suffix.count)), suffix)
    }
    return nil
  }

  private static let fixed: [String: String] = [
    "v.patch_embd": "patch_embed.proj",
    "v.position_embd": "pos_embed",
    // The merger's own layer norm is clip's post-encoder norm, and its two projections are the
    // numbered slots every clip projector writes into.
    "v.post_ln": "merger.norm",
    "mm.0": "merger.linear_fc1",
    "mm.2": "merger.linear_fc2",
  ]

  private static let block: [String: String] = [
    "ln1": "norm1",
    "ln2": "norm2",
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
  ]
}
