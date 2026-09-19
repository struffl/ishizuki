// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The sparse feed-forward block, against a reference that shares none of its machinery.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import IshizukiKit

/// `MoEBlock` routes with a softmax over the whole bank, keeps the top few, renormalizes, and
/// runs the chosen experts straight out of a stacked quantized tensor. The reference below
/// dequantizes every expert and loops, so the two agree only if the routing and the gather do.
@Suite("MoE block")
struct MoEBlockTests {
  private let hidden = 128
  private let expertWidth = 64
  private let sharedWidth = 128
  private let experts = 8
  private let topK = 3
  private let groupSize = 64
  private let bits = 4

  /// Random weights, quantized once, kept as both the packed arrays the block reads and the
  /// dense arrays the reference multiplies.
  private struct Bank {
    var store: [String: MLXArray] = [:]
    var dense: [String: MLXArray] = [:]

    mutating func add(_ name: String, _ w: MLXArray, groupSize: Int, bits: Int) {
      let (q, scales, biases) = quantized(w, groupSize: groupSize, bits: bits)
      store[name + ".weight"] = q
      store[name + ".scales"] = scales
      store[name + ".biases"] = biases!
      dense[name] = dequantized(
        q, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
    }
  }

  private func build(shared: Bool) -> (MoEBlock, Bank, BonsaiConfig.TextConfig) {
    MLXRandom.seed(4242)
    let prefix = "model.layers.0.mlp"
    var bank = Bank()
    bank.add(
      prefix + ".gate", MLXRandom.normal([experts, hidden]).asType(.float32) * 0.2,
      groupSize: groupSize, bits: 8)
    for name in ["gate_proj", "up_proj"] {
      bank.add(
        "\(prefix).switch_mlp.\(name)",
        MLXRandom.normal([experts, expertWidth, hidden]).asType(.float32) * 0.1,
        groupSize: groupSize, bits: bits)
    }
    bank.add(
      prefix + ".switch_mlp.down_proj",
      MLXRandom.normal([experts, hidden, expertWidth]).asType(.float32) * 0.1,
      groupSize: groupSize, bits: bits)

    if shared {
      for (name, shape) in [
        ("gate_proj", [sharedWidth, hidden]), ("up_proj", [sharedWidth, hidden]),
        ("down_proj", [hidden, sharedWidth]),
      ] {
        bank.add(
          "\(prefix).shared_expert.\(name)",
          MLXRandom.normal(shape).asType(.float32) * 0.1, groupSize: groupSize, bits: bits)
      }
      bank.add(
        prefix + ".shared_expert_gate", MLXRandom.normal([1, hidden]).asType(.float32) * 0.2,
        groupSize: groupSize, bits: 8)
    }

    var overrides: [String: BonsaiConfig.ModuleQuant] = [:]
    for path in [prefix + ".gate", prefix + ".shared_expert_gate"] {
      overrides[path] = BonsaiConfig.ModuleQuant(bits: 8, groupSize: groupSize)
    }
    let config = BonsaiConfig.TextConfig.forMoETest(
      hidden: hidden, experts: experts, topK: topK, expertWidth: expertWidth)
    let store = WeightStore(arrays: bank.store)
    let factory = PackedModuleFactory(
      store: store,
      config: BonsaiConfig.moeTestPack(text: config, overrides: overrides),
      tensorPrefix: "")

    let block = try! MoEBlock(
      config: config, layer: 0, factory: factory, store: store)
    return (block, bank, config)
  }

  /// Softmax, top-k, renormalize, one SwiGLU per chosen expert, weighted sum — written out
  /// rather than gathered.
  private func reference(_ x: MLXArray, _ bank: Bank, shared: Bool) -> MLXArray {
    let p = "model.layers.0.mlp"
    let logits = matmul(x, bank.dense[p + ".gate"]!.T)
    let gates = softmax(logits, axis: -1, precise: true)

    var rows: [MLXArray] = []
    for token in 0..<x.dim(0) {
      let row = gates[token]
      let order = argSort(row).asArray(Int32.self).reversed().prefix(topK)
      var sum = MLXArray.zeros([hidden], dtype: .float32)
      var total: Float = 0
      for e in order { total += row[Int(e)].item(Float.self) }
      for e in order {
        let index = Int(e)
        let weight = row[index].item(Float.self) / total
        let xi = x[token]
        let gate = matmul(bank.dense[p + ".switch_mlp.gate_proj"]![index], xi)
        let up = matmul(bank.dense[p + ".switch_mlp.up_proj"]![index], xi)
        let out = matmul(bank.dense[p + ".switch_mlp.down_proj"]![index], silu(gate) * up)
        sum = sum + weight * out
      }
      if shared {
        let xi = x[token]
        let gate = matmul(bank.dense[p + ".shared_expert.gate_proj"]!, xi)
        let up = matmul(bank.dense[p + ".shared_expert.up_proj"]!, xi)
        let out = matmul(bank.dense[p + ".shared_expert.down_proj"]!, silu(gate) * up)
        let g = sigmoid(matmul(bank.dense[p + ".shared_expert_gate"]!, xi))
        sum = sum + g * out
      }
      rows.append(sum)
    }
    return stacked(rows, axis: 0)
  }

  private func check(shared: Bool) {
    let (block, bank, _) = build(shared: shared)
    let x = MLXRandom.normal([5, hidden]).asType(.float32)
    let got = block(x).asType(.float32)
    let want = reference(x, bank, shared: shared)
    eval(got, want)

    let scale = maximum(abs(want).max(), MLXArray(Float(1e-3))).item(Float.self)
    let error = abs(got - want).max().item(Float.self) / scale
    #expect(got.shape == [5, hidden])
    #expect(error < 2e-2, "routed output differs by \(error) of full scale")
  }

  @Test("routes to the top experts and matches an expert-by-expert reference")
  func matchesReference() { check(shared: false) }

  @Test("adds the shared expert through its own sigmoid gate")
  func matchesReferenceWithSharedExpert() { check(shared: true) }

  @Test("a checkpoint says which layers are sparse")
  func sparseLayers() {
    var config = BonsaiConfig.TextConfig.forMoETest(
      hidden: hidden, experts: experts, topK: topK, expertWidth: expertWidth)
    config.numHiddenLayers = 6
    #expect(config.isSparse == Array(repeating: true, count: 6))

    config.decoderSparseStep = 2
    #expect(config.isSparse == [true, false, true, false, true, false])

    config.decoderSparseStep = 1
    config.mlpOnlyLayers = [0, 3]
    #expect(config.isSparse == [false, true, true, false, true, true])

    config.numExperts = 0
    #expect(config.isSparse == Array(repeating: false, count: 6))
  }
}

extension BonsaiConfig.TextConfig {
  /// The smallest text config `MoEBlock` reads: the sparse fields, plus whatever the decoder
  /// would need if one were built around it.
  static func forMoETest(hidden: Int, experts: Int, topK: Int, expertWidth: Int)
    -> BonsaiConfig.TextConfig
  {
    let json: [String: Any] = [
      "model_type": "qwen3_5_moe", "hidden_size": hidden, "intermediate_size": hidden,
      "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
      "head_dim": hidden, "rms_norm_eps": 1e-6, "vocab_size": 32,
      "max_position_embeddings": 64, "tie_word_embeddings": false,
      "num_experts": experts, "num_experts_per_tok": topK,
      "moe_intermediate_size": expertWidth,
      "layer_types": ["full_attention"],
      "linear_num_value_heads": 0, "linear_num_key_heads": 0,
      "linear_value_head_dim": 0, "linear_key_head_dim": 0, "linear_conv_kernel_dim": 0,
      "rope_parameters": ["rope_theta": 10000.0],
    ]
    return try! JSONDecoder().decode(
      BonsaiConfig.TextConfig.self, from: try! JSONSerialization.data(withJSONObject: json))
  }
}

extension BonsaiConfig {
  static func moeTestPack(
    text: BonsaiConfig.TextConfig, overrides: [String: BonsaiConfig.ModuleQuant]
  ) -> BonsaiConfig {
    BonsaiConfig(
      schemaVersion: 0, modelType: "qwen3_5_moe", baseModelType: nil, textConfig: text,
      visionConfig: nil, modules: [],
      quantization: BonsaiConfig.QuantizationConfig(
        bits: 4, groupSize: 64, mode: "affine", overrides: overrides),
      components: nil, tensorNamespace: nil, gdnActivationLayout: nil, requiresRuntime: nil)
  }
}
