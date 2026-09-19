// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import IshizukiKit

@Suite("Model catalog")
struct ModelCatalogTests {
  private func pack(
    at directory: URL, config: String, weights: Bool = true
  ) throws {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try Data(config.utf8).write(to: directory.appending(path: "config.json"))
    if weights {
      try Data().write(to: directory.appending(path: "model.safetensors"))
    }
  }

  private let affine = """
    {
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "rms_norm_eps": 1e-6, "vocab_size": 100,
        "max_position_embeddings": 262144, "tie_word_embeddings": false,
        "mtp_num_hidden_layers": 1,
        "rope_parameters": {"rope_theta": 10000, "type": "default"}
      },
      "vision_config": {
        "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
        "in_channels": 3, "patch_size": 16, "temporal_patch_size": 2,
        "spatial_merge_size": 2, "out_hidden_size": 64, "num_position_embeddings": 64
      },
      "quantization": {
        "bits": 4, "group_size": 64, "mode": "affine",
        "language_model.model.layers.0.mlp.down_proj": {
          "bits": 5, "group_size": 64, "mode": "affine"
        }
      }
    }
    """

  @Test("a GGUF is offered on its own, and an mmproj beside it is not a model")
  func discoversGGUF() throws {
    let root = URL(filePath: NSTemporaryDirectory()).appending(path: "cat-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let directory = root.appending(path: "ggufs")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    let model = directory.appending(path: "Qwen3.8-27B-IQ3_S.gguf")
    try GGUFFixture.qwen35(at: model)
    // Half a model: a tower with no language model, which nothing can load by itself.
    try GGUFFixture.qwen35(at: directory.appending(path: "mmproj-Qwen3.8-27B-BF16.gguf"))
    // Not this container at all.
    try Data("not a gguf".utf8).write(to: directory.appending(path: "notes.gguf"))

    let catalog = ModelCatalog.discover(in: [root])
    #expect(catalog.entries.count == 1)
    let entry = try #require(catalog["Qwen3.8-27B-IQ3_S"])
    #expect(entry.format == .gguf)
    // FileManager hands back /private/var where NSTemporaryDirectory says /var.
    #expect(entry.url.resolvingSymlinksInPath().path == model.resolvingSymlinksInPath().path)
    #expect(
      entry.directory.resolvingSymlinksInPath().path
        == directory.resolvingSymlinksInPath().path)
    #expect(entry.contextTokens == 262_144)
    #expect(entry.hasVision)
    #expect(entry.byteCount > 0)
    // Counting tensors would name it after the norms; counting bytes names it after the
    // projections, which is what fills the file.
    #expect(entry.quantization == "IQ3_S")
  }

  @Test("a pack is found, named and described")
  func discovers() throws {
    let root = URL(filePath: NSTemporaryDirectory()).appending(path: "cat-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    try pack(at: root.appending(path: "my-pack"), config: affine)

    let catalog = ModelCatalog.discover(in: [root])
    let entry = try #require(catalog["my-pack"])
    #expect(entry.quantization == "4/5-bit g64")
    #expect(entry.hasVision)
    #expect(entry.hasMTP)
    #expect(entry.contextTokens == 262_144)
  }

  @Test("a HuggingFace checkout is named by its repo, not its revision")
  func huggingFaceNaming() throws {
    let root = URL(filePath: NSTemporaryDirectory()).appending(path: "cat-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    try pack(
      at: root.appending(path: "models--org--Some-Pack/snapshots/abc123"), config: affine)

    let catalog = ModelCatalog.discover(in: [root])
    #expect(catalog.entries.map(\.id) == ["org/Some-Pack"])
  }

  @Test("a directory with no weights is not offered as a model")
  func needsWeights() throws {
    let root = URL(filePath: NSTemporaryDirectory()).appending(path: "cat-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    try pack(at: root.appending(path: "no-weights"), config: affine, weights: false)

    #expect(ModelCatalog.discover(in: [root]).entries.isEmpty)
  }

  @Test("a config this runtime cannot read is not offered either")
  func needsValidConfig() throws {
    let root = URL(filePath: NSTemporaryDirectory()).appending(path: "cat-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    try pack(
      at: root.appending(path: "foreign"),
      config: affine.replacingOccurrences(of: "\"qwen3_5\"", with: "\"llama\""))

    #expect(ModelCatalog.discover(in: [root]).entries.isEmpty)
  }
}
