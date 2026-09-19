// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Testing

@testable import IshizukiKit

@Suite("Bonsai config")
struct BonsaiConfigTests {
  private func load(_ json: String) throws -> BonsaiConfig {
    let dir = URL(filePath: NSTemporaryDirectory())
      .appending(path: "bonsai-config-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    try Data(json.utf8).write(to: dir.appending(path: "config.json"))
    return try BonsaiConfig.load(directory: dir)
  }

  private let ternary1_7B = """
    {
      "model_type": "qwen3",
      "hidden_size": 2048,
      "intermediate_size": 6144,
      "num_hidden_layers": 28,
      "num_attention_heads": 16,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "rms_norm_eps": 1e-06,
      "vocab_size": 151669,
      "max_position_embeddings": 32768,
      "rope_theta": 1000000.0,
      "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192},
      "tie_word_embeddings": true,
      "quantization": {"group_size": 128, "bits": 2}
    }
    """

  private let ternary8B = """
    {
      "model_type": "qwen3",
      "hidden_size": 4096,
      "intermediate_size": 12288,
      "num_hidden_layers": 36,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "rms_norm_eps": 1e-06,
      "vocab_size": 151669,
      "max_position_embeddings": 65536,
      "rope_theta": 1000000.0,
      "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16384},
      "tie_word_embeddings": false,
      "quantization": {"group_size": 128, "bits": 2}
    }
    """

  private let flagshipPack = """
    {
      "schema_version": 2,
      "model_type": "prism_hadamard_qwen35",
      "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120, "intermediate_size": 17408,
        "num_hidden_layers": 2, "num_attention_heads": 24, "num_key_value_heads": 4,
        "head_dim": 256, "rms_norm_eps": 1e-06, "vocab_size": 248320,
        "max_position_embeddings": 262144, "tie_word_embeddings": false,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_value_heads": 48, "linear_num_key_heads": 16,
        "linear_value_head_dim": 128, "linear_key_head_dim": 128, "linear_conv_kernel_dim": 4,
        "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
      },
      "modules": [{"path": "lm_head", "block": 1024, "embedding": false, "dtype": "float16"}],
      "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
      "components": {"text": true, "vision": true, "mtp": false}
    }
    """

  @Test("a standard MLX ternary pack loads, translates, and validates")
  func legacyLoads() throws {
    let config = try load(ternary1_7B)
    try config.validate()
    #expect(config.modelType == "qwen3")
    #expect(config.quantization.bits == 2)
    #expect(config.quantization.groupSize == 128)
    #expect(config.quantization.mode == "affine")
    #expect(config.modules.isEmpty)
    #expect(config.components?.vision == false)

    let text = config.textConfig
    #expect(text.numHiddenLayers == 28)
    #expect(text.numKeyValueHeads == 8)
    #expect(text.headDim == 128)
    #expect(text.tieWordEmbeddings)
    #expect(text.isFullAttention.count == 28)
    #expect(text.isFullAttention.allSatisfy { $0 })
    #expect(text.ropeParameters.ropeType == "yarn")
    #expect(text.ropeParameters.factor == Float(4))
    #expect(text.ropeParameters.originalMaxPositionEmbeddings == 8192)
  }

  @Test("the 8B pack keeps its own untied head and extended context")
  func legacyUntied() throws {
    let config = try load(ternary8B)
    try config.validate()
    #expect(config.textConfig.tieWordEmbeddings == false)
    #expect(config.textConfig.numHiddenLayers == 36)
    #expect(config.textConfig.maxPositionEmbeddings == 65536)
    #expect(config.textConfig.ropeParameters.originalMaxPositionEmbeddings == 16384)
  }

  @Test("the flagship Hadamard pack still loads through the schema path")
  func flagshipLoads() throws {
    let config = try load(flagshipPack)
    try config.validate()
    #expect(config.modelType == "prism_hadamard_qwen35")
    #expect(config.quantization.mode == "affine")
    #expect(config.modules.first?.block == 1024)
    #expect(config.textConfig.isFullAttention == [false, true])
  }

  @Test("a width MLX can run is accepted whatever the pack was built at")
  func widthsWiden() throws {
    // An affine pack is no longer pinned to the ternary width; 4-bit is a pack, not an error.
    #expect(throws: Never.self) {
      try load(ternary1_7B.replacingOccurrences(of: "\"bits\": 2", with: "\"bits\": 4"))
        .validate()
    }
  }

  @Test("unsupported bit widths and model types are refused")
  func rejections() {
    #expect(throws: BonsaiError.self) {
      try load(ternary1_7B.replacingOccurrences(of: "\"bits\": 2", with: "\"bits\": 7"))
        .validate()
    }
    #expect(throws: BonsaiError.self) {
      try load(
        ternary1_7B.replacingOccurrences(of: "\"group_size\": 128", with: "\"group_size\": 96")
      ).validate()
    }
    #expect(throws: BonsaiError.self) {
      try load(ternary1_7B.replacingOccurrences(of: "\"qwen3\"", with: "\"llama\""))
        .validate()
    }
  }

  @Test("a rotated pack still has to be uniform, because its kernels assume it")
  func rotatedStaysNarrow() {
    #expect(throws: BonsaiError.self) {
      try load(flagshipPack.replacingOccurrences(of: "\"bits\": 2", with: "\"bits\": 4"))
        .validate()
    }
  }
}
