// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

@Suite("Calibration")
struct CalibrationTests {
  private let hidden = 256
  private let intermediate = 512
  private let vocab = 1024

  /// A small checkpoint shaped like the real thing, already in this runtime's canonical
  /// layout (no `model.language_model.` nesting to translate) — calibration only needs an
  /// architecture and real weights, not the upstream-naming path `Quantizer` also handles.
  private func writeSource(at directory: URL) throws {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    var arrays: [String: MLXArray] = [:]
    func dense(_ name: String, _ shape: [Int]) {
      arrays[name] = MLXRandom.normal(shape).asType(.bfloat16)
    }

    dense("language_model.model.embed_tokens.weight", [vocab, hidden])
    dense("language_model.lm_head.weight", [vocab, hidden])
    dense("language_model.model.norm.weight", [hidden])
    for layer in 0..<2 {
      let p = "language_model.model.layers.\(layer)"
      dense("\(p).input_layernorm.weight", [hidden])
      dense("\(p).post_attention_layernorm.weight", [hidden])
      dense("\(p).self_attn.q_proj.weight", [hidden, hidden])
      dense("\(p).self_attn.k_proj.weight", [hidden, hidden])
      dense("\(p).self_attn.v_proj.weight", [hidden, hidden])
      dense("\(p).self_attn.o_proj.weight", [hidden, hidden])
      dense("\(p).self_attn.q_norm.weight", [64])
      dense("\(p).self_attn.k_norm.weight", [64])
      dense("\(p).mlp.gate_proj.weight", [intermediate, hidden])
      dense("\(p).mlp.up_proj.weight", [intermediate, hidden])
      dense("\(p).mlp.down_proj.weight", [hidden, intermediate])
    }

    let config: [String: Any] = [
      "model_type": "qwen3_5",
      "text_config": [
        "model_type": "qwen3_5_text", "hidden_size": hidden,
        "intermediate_size": intermediate, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 64,
        "rms_norm_eps": 1e-6, "vocab_size": vocab, "max_position_embeddings": 4096,
        "tie_word_embeddings": false,
        "layer_types": ["full_attention", "full_attention"],
        "rope_parameters": ["rope_theta": 10000, "type": "default"],
      ],
    ]
    try JSONSerialization.data(withJSONObject: config)
      .write(to: directory.appending(path: "config.json"))
    try MLX.save(arrays: arrays, url: directory.appending(path: "model.safetensors"))
  }

  private func temp() -> URL {
    URL(filePath: NSTemporaryDirectory()).appending(path: "calib-\(UUID().uuidString)")
  }

  @Test("calibration collects a non-degenerate importance vector for every linear module")
  func collectsImportancePerModule() throws {
    let root = temp()
    defer { try? FileManager.default.removeItem(at: root) }
    try writeSource(at: root)

    let model = try CalibrationModel(source: try SourceCheckpoint(directory: root))
    for _ in 0..<4 {
      let tokens = (0..<16).map { _ in Int32.random(in: 0..<Int32(vocab)) }
      model.calibrate(tokens)
    }

    let modulesPerLayer = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    for layer in 0..<2 {
      for module in modulesPerLayer {
        let path = "model.layers.\(layer).\(module)"
        let importance = try #require(
          model.collector.importance(for: path), "no importance collected for \(path)")
        eval(importance)
        #expect(importance.min().item(Float.self) >= 0, "\(path): importance must be non-negative")
        #expect(importance.max().item(Float.self) > 0, "\(path): importance is degenerately all-zero")
        // Different input channels should not all carry identical energy — that would mean
        // the collector is summing over the wrong axis rather than per input channel.
        let spread = importance.max().item(Float.self) - importance.min().item(Float.self)
        #expect(spread > 0, "\(path): every channel has identical importance, which means the wrong axis was reduced")
      }
    }

    // Nothing collects for the embedding: a row is gathered by token id, not consumed as an
    // input channel, matching oMLX's own carve-out for token embeddings.
    #expect(model.collector.importance(for: "model.embed_tokens") == nil)
  }

  @Test("a dense PackedLinear computes exactly x @ w.T, with no quantization involved")
  func denseLinearIsPlainMatmul() {
    let weight = MLXRandom.normal([32, 64]).asType(.float16)
    let x = MLXRandom.normal([4, 8, 64]).asType(.float16)

    let dense = PackedLinear(dense: weight)
    let got = dense(x)
    eval(got)

    let expected = matmul(x, weight.T)
    eval(expected)

    #expect(dense.isDense)
    #expect(got.shape == [4, 8, 32])
    let diff = (got.asType(.float32) - expected.asType(.float32))
    #expect(abs(diff).max().item(Float.self) == 0, "a dense projection must not alter the weight at all")
  }

  @Test("the collector's importance is the mean square per input channel, not a proxy for it")
  func importanceIsExactMeanSquare() throws {
    let collector = ActivationCollector()
    // Channel 0 is always large, channel 1 always small, across two batches of two rows.
    let batch1 = MLXArray([2.0, 1.0, -2.0, 1.0] as [Float], [2, 2])
    let batch2 = MLXArray([4.0, -1.0, -4.0, -1.0] as [Float], [2, 2])
    collector.record(path: "m", x: batch1)
    collector.record(path: "m", x: batch2)

    let importance = try #require(collector.importance(for: "m"))
    eval(importance)
    // channel 0: (4+4+16+16)/4 = 10; channel 1: (1+1+1+1)/4 = 1
    #expect(importance.asArray(Float.self) == [10.0, 1.0])
  }
}
