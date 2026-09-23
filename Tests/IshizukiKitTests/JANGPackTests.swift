// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

// Shaped after dealignai/Bonsai-2-27B-CRACK-Ternary-JANG: the Prism ternary pack repacked as
// MLX affine, its rotation declared in config.json and its norm layout in jang_config.json.
@Suite("JANG packs")
struct JANGPackTests {
  private let config = """
    {
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120, "intermediate_size": 17408,
        "num_hidden_layers": 2, "num_attention_heads": 24, "num_key_value_heads": 4,
        "head_dim": 256, "rms_norm_eps": 1e-06, "vocab_size": 248320,
        "max_position_embeddings": 262144, "tie_word_embeddings": false,
        "attn_output_gate": true,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_value_heads": 48, "linear_num_key_heads": 16,
        "linear_value_head_dim": 128, "linear_key_head_dim": 128, "linear_conv_kernel_dim": 4,
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
      },
      "vision_config": {
        "depth": 1, "hidden_size": 1152, "intermediate_size": 4304, "num_heads": 16,
        "in_channels": 3, "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
        "out_hidden_size": 5120, "num_position_embeddings": 2304
      },
      "quantization": {
        "group_size": 128, "bits": 2, "mode": "affine",
        "language_model.lm_head": {"bits": 2, "group_size": 128, "mode": "affine"},
        "vision_tower.blocks.0.attn.qkv": {"bits": 6, "group_size": 128, "mode": "affine"}
      },
      "hadamard": {
        "contract": "prism.hadamard.v1",
        "block_size": 1024,
        "transform": "normalized-sylvester-walsh-hadamard",
        "axis": "input-last-dimension",
        "sign_mode": "explicit",
        "gdn_v_grouped": true,
        "forward_modules": [
          "language_model.lm_head", "language_model.model.layers.0.mlp.down_proj"
        ],
        "inverse_modules": ["language_model.model.embed_tokens"]
      }
    }
    """

  private let manifest = """
    {
      "format": "jang", "format_version": "2.0", "weight_format": "affine",
      "layout": {
        "language_norms": "zero-centered-runtime-plus-one", "shifted_norm_count": 161,
        "gdn_activation_layout": "grouped"
      },
      "runtime": {
        "requires_hadamard_activation_transform": true, "requires_jang_affine1_expansion": false
      }
    }
    """

  private var unrotated: String {
    config.replacingOccurrences(of: "\"hadamard\"", with: "\"unused\"")
  }

  private func load(config: String? = nil, manifest: String? = nil) throws -> BonsaiConfig {
    let dir = URL(filePath: NSTemporaryDirectory())
      .appending(path: "jang-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    try Data((config ?? self.config).utf8).write(to: dir.appending(path: "config.json"))
    try Data((manifest ?? self.manifest).utf8).write(
      to: dir.appending(path: JANGPack.manifestFile))
    return try BonsaiConfig.load(directory: dir)
  }

  @Test("the rotation block becomes the packed-module records the rotated path reads")
  func rotation() throws {
    let config = try load()
    try config.validate()
    #expect(config.profile == .rotated)
    #expect(config.baseModelType == "qwen3_5")
    #expect(config.centredNorms == 161)
    #expect(config.components?.vision == true)
    #expect(config.quantization.module("vision_tower.blocks.0.attn.qkv").bits == 6)

    let records = Dictionary(uniqueKeysWithValues: config.modules.map { ($0.path, $0) })
    #expect(records.count == 3)
    #expect(records["lm_head"]?.block == 1024)
    #expect(records["lm_head"]?.embedding == false)
    #expect(records["model.layers.0.mlp.down_proj"] != nil)
    #expect(records["model.embed_tokens"]?.embedding == true)
  }

  @Test("a manifest this cannot honour refuses rather than loading as plain affine")
  func refusals() {
    #expect(throws: BonsaiError.self) { try load(config: unrotated) }
    #expect(throws: BonsaiError.self) {
      try load(
        manifest: manifest.replacingOccurrences(
          of: "\"weight_format\": \"affine\"", with: "\"weight_format\": \"mxtq\""))
    }
    #expect(throws: BonsaiError.self) {
      try load(
        manifest: manifest.replacingOccurrences(
          of: "\"requires_jang_affine1_expansion\": false",
          with: "\"requires_jang_affine1_expansion\": true"))
    }
    #expect(throws: BonsaiError.self) {
      try load(
        config: config.replacingOccurrences(
          of: "\"sign_mode\": \"explicit\"", with: "\"sign_mode\": \"seeded\""))
    }
    #expect(throws: BonsaiError.self) {
      try load(
        config: config.replacingOccurrences(
          of: "\"language_model.model.layers.0.mlp.down_proj\"",
          with: "\"vision_tower.blocks.0.attn.qkv\""))
    }
  }

  @Test("a JANG pack without a rotation still takes its norm layout")
  func unrotatedManifest() throws {
    let config = try load(
      config: unrotated,
      manifest: manifest.replacingOccurrences(
        of: "\"requires_hadamard_activation_transform\": true",
        with: "\"requires_hadamard_activation_transform\": false"))
    try config.validate()
    #expect(config.profile == .affine)
    #expect(config.modules.isEmpty)
    #expect(config.centredNorms == 161)
  }

  @Test("only the centred norms gain their one, and a miscount is refused")
  func normFold() throws {
    let zero = MLXArray.zeros([4], dtype: .float32)
    let store = WeightStore(arrays: [
      "language_model.model.norm.weight": zero,
      "language_model.model.layers.0.input_layernorm.weight": zero,
      "language_model.model.layers.3.self_attn.q_norm.weight": zero,
      "language_model.model.layers.0.linear_attn.norm.weight": zero,
      "vision_tower.merger.norm.weight": zero,
    ])
    let folded = try store.foldingCentredNorms(expected: 3)
    func total(_ name: String) throws -> Float { try folded(name).sum().item(Float.self) }
    #expect(try total("language_model.model.norm.weight") == 4)
    #expect(try total("language_model.model.layers.0.input_layernorm.weight") == 4)
    #expect(try total("language_model.model.layers.3.self_attn.q_norm.weight") == 4)
    #expect(try total("language_model.model.layers.0.linear_attn.norm.weight") == 0)
    #expect(try total("vision_tower.merger.norm.weight") == 0)
    #expect(throws: BonsaiError.self) { try store.foldingCentredNorms(expected: 161) }
  }

  @Test("a quantized tower linear keeps its own bias apart from the affine biases")
  func quantizedVisionLinear() throws {
    let prefix = "vision_tower.blocks.0.attn.qkv"
    let w = MLXRandom.normal([96, 256]) * 0.05
    let bias = MLXRandom.normal([96])
    let (wq, scales, affine) = quantized(w, groupSize: 128, bits: 6, mode: .affine)
    let biases = try #require(affine)
    let store = WeightStore(arrays: [
      prefix + ".weight": wq, prefix + ".scales": scales, prefix + ".biases": biases,
      prefix + ".bias": bias,
    ])
    let quantization = BonsaiConfig.QuantizationConfig(
      bits: 2, groupSize: 128, overrides: [prefix: .init(bits: 6, groupSize: 128)])
    let linear = try visionLinear(store: store, prefix: prefix, quantization: quantization)
    #expect(linear.inputDim == 256)
    #expect(linear.outputDim == 96)

    let x = MLXRandom.normal([3, 256])
    let dense = dequantized(wq, scales: scales, biases: biases, groupSize: 128, bits: 6)
    let reference = matmul(x, dense.T) + bias
    #expect(abs(linear(x) - reference).max().item(Float.self) < 1e-3)
    #expect(throws: BonsaiError.self) {
      try visionLinear(store: store, prefix: prefix, quantization: nil)
    }
  }
}
