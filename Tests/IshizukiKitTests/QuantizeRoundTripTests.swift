// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

/// The progress callback is handed out to a Sendable closure, so the test collects through a
/// lock rather than a captured var.
private final class PhaseLog: @unchecked Sendable {
  private let lock = NSLock()
  private var seen: Set<Quantizer.Progress.Phase> = []

  func record(_ phase: Quantizer.Progress.Phase) {
    lock.lock()
    seen.insert(phase)
    lock.unlock()
  }

  var phases: Set<Quantizer.Progress.Phase> {
    lock.lock()
    defer { lock.unlock() }
    return seen
  }
}

@Suite("Quantize round trip")
struct QuantizeRoundTripTests {
  private let hidden = 256
  private let intermediate = 512
  private let vocab = 1024

  /// A small checkpoint shaped like the real thing: nested under language_model, two layers,
  /// a vision tensor and an MTP head, so the parts that must survive untouched are present.
  private func writeSource(at directory: URL, shards: Int = 1) throws {
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
    dense("language_model.mtp.fc.weight", [hidden, 2 * hidden])
    dense("language_model.mtp.norm.weight", [hidden])
    dense("vision_tower.patch_embed.proj.bias", [hidden])

    let config: [String: Any] = [
      "model_type": "qwen3_5",
      "text_config": [
        "model_type": "qwen3_5_text", "hidden_size": hidden,
        "intermediate_size": intermediate, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 64,
        "rms_norm_eps": 1e-6, "vocab_size": vocab, "max_position_embeddings": 4096,
        "tie_word_embeddings": false, "mtp_num_hidden_layers": 1,
        "layer_types": ["full_attention", "full_attention"],
        "rope_parameters": ["rope_theta": 10000, "type": "default"],
      ],
    ]
    try JSONSerialization.data(withJSONObject: config)
      .write(to: directory.appending(path: "config.json"))

    if shards == 1 {
      try MLX.save(arrays: arrays, url: directory.appending(path: "model.safetensors"))
    } else {
      let names = arrays.keys.sorted()
      var weightMap: [String: String] = [:]
      for (index, chunk) in stride(from: 0, to: names.count, by: names.count / shards + 1)
        .map({ Array(names[$0..<min($0 + names.count / shards + 1, names.count)]) }).enumerated()
      {
        let file = String(format: "model-%05d-of-%05d.safetensors", index + 1, shards)
        var part: [String: MLXArray] = [:]
        for name in chunk {
          part[name] = arrays[name]
          weightMap[name] = file
        }
        try MLX.save(arrays: part, url: directory.appending(path: file))
      }
      try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
        .write(to: directory.appending(path: "model.safetensors.index.json"))
    }
  }

  private func temp() -> URL {
    URL(filePath: NSTemporaryDirectory()).appending(path: "quant-\(UUID().uuidString)")
  }

  @Test("a checkpoint becomes a pack this runtime can read back")
  func roundTrip() throws {
    let root = temp()
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appending(path: "src")
    let output = root.appending(path: "out")
    try writeSource(at: source)

    let checkpoint = try SourceCheckpoint(directory: source)
    let seen = PhaseLog()
    let outcome = try Quantizer(
      source: checkpoint, profile: .balanced, destination: output
    ) { seen.record($0.phase) }.run()
    let phases = seen.phases

    #expect(outcome.shards >= 1)
    #expect(outcome.byteCount > 0)
    #expect(phases.contains(.surveying))
    #expect(phases.contains(.writing))

    // The pack has to satisfy the loader, not merely exist.
    let config = try BonsaiConfig.load(directory: output)
    try config.validate()
    #expect(config.profile == .affine)
    #expect(config.quantization.bits == 3)
    #expect(config.quantization.groupSize == 64)
    #expect(config.components?.mtp == true)

    let store = try WeightStore(directory: output)
    #expect(store.has("language_model.model.layers.0.mlp.down_proj.scales"))
    // Norms are not quantized, and must come through unchanged in shape.
    #expect(store.has("language_model.model.layers.0.input_layernorm.weight"))
    #expect(!store.has("language_model.model.layers.0.input_layernorm.scales"))
    // Things that are not the language model still travel with the pack.
    #expect(store.has("vision_tower.patch_embed.proj.bias"))
    #expect(store.has("language_model.mtp.fc.scales"))

    // Every override the config names must resolve to the width it claims.
    let factory = PackedModuleFactory(
      store: store, config: config, tensorPrefix: "language_model.")
    for (path, entry) in config.quantization.overrides {
      #expect(entry.bits > 3, "\(path) is listed as an override but sits at the base width")
    }
    // The embedding and the head are the two tensors most likely to be starved by the auction,
    // so they are floored at 4 bits rather than left to compete for it.
    #expect(config.quantization.overrides["language_model.model.embed_tokens"]?.bits == 4)
    #expect(config.quantization.overrides["language_model.lm_head"]?.bits == 4)
    let down = try factory.linear("model.layers.0.mlp.down_proj")
    #expect(down.inputDim == intermediate)
    #expect(down.outputDim == hidden)
  }

  @Test("calibration changes what gets written, and the pack still loads and validates")
  func calibrationAffectsOutput() throws {
    let root = temp()
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appending(path: "src")
    try writeSource(at: source)

    // Bypasses the tokenizer this fixture has no tokenizer.json for — Quantizer accepts
    // pre-tokenized calibration sequences for exactly this reason.
    let calibrationTokens: [[Int32]] = (0..<4).map { _ in
      (0..<32).map { _ in Int32.random(in: 0..<Int32(vocab)) }
    }

    let plainOutput = root.appending(path: "out-plain")
    _ = try Quantizer(
      source: try SourceCheckpoint(directory: source), profile: .balanced,
      destination: plainOutput
    ).run()

    let calibratedOutput = root.appending(path: "out-calibrated")
    let seen = PhaseLog()
    let outcome = try Quantizer(
      source: try SourceCheckpoint(directory: source), profile: .balanced,
      destination: calibratedOutput, calibrate: true, calibrationTokens: calibrationTokens
    ) { seen.record($0.phase) }.run()

    #expect(seen.phases.contains(.calibrating))
    #expect(outcome.achievedBpw > 0)

    // Calibration genuinely changes which scale/bias each group's clip search settles on, so
    // an otherwise-identical run should not write byte-identical weights.
    let plainStore = try WeightStore(directory: plainOutput)
    let calibratedStore = try WeightStore(directory: calibratedOutput)
    let path = "language_model.model.layers.0.mlp.down_proj.weight"
    let plainW = try plainStore(path)
    let calibratedW = try calibratedStore(path)
    eval(plainW, calibratedW)
    #expect(plainW.shape == calibratedW.shape)
    let mismatches = (plainW .!= calibratedW).sum().item(Int32.self)
    #expect(
      mismatches > 0,
      "calibrated quantization produced byte-identical output to the uncalibrated run")

    // The calibrated pack has to satisfy the loader like any other.
    let config = try BonsaiConfig.load(directory: calibratedOutput)
    try config.validate()
  }

  @Test("the measured bpw lands under the profile's target, except for the embedding and head floor")
  func respectsBudget() throws {
    let root = temp()
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appending(path: "src")
    try writeSource(at: source)

    for profile in QuantProfile.all {
      let output = root.appending(path: "out-\(profile.name)")
      let outcome = try Quantizer(
        source: try SourceCheckpoint(directory: source), profile: profile,
        destination: output
      ).run()
      // The embedding and lm_head are floored at 4 bits regardless of the budget (see
      // BitAllocator.pinnedMinimumBits), so a profile whose base sits below that floor can land
      // over its target here — this synthetic checkpoint is small enough that the two pinned
      // tensors are a much larger share of it than of a real model, so the overshoot is more
      // visible than it would be in practice. Half a bit of headroom covers it.
      #expect(
        outcome.achievedBpw <= profile.targetBpw + 0.5,
        "\(profile.name) reached \(outcome.achievedBpw), over \(profile.targetBpw)")
      #expect(outcome.achievedBpw >= Double(profile.baseBits))
    }
  }

  /// The pack is the only place the implied one can be folded in — the runtime scales by the
  /// weight it reads — so a source in upstream layout has to come out of the quantizer already
  /// centred on one, and a delta-net norm has to come out untouched.
  @Test("an upstream checkpoint's zero-centred norms are folded into the pack")
  func foldsZeroCentredNorms() throws {
    let root = temp()
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appending(path: "src")
    let output = root.appending(path: "out")
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)

    let inputNorm = MLXArray([-0.25, 0.5] as [Float]).asType(.bfloat16)
    let gatedNorm = MLXArray([0.75, 1.25] as [Float]).asType(.bfloat16)
    var arrays: [String: MLXArray] = [
      "model.language_model.embed_tokens.weight": MLXRandom.normal([vocab, hidden])
        .asType(.bfloat16),
      "lm_head.weight": MLXRandom.normal([vocab, hidden]).asType(.bfloat16),
      "model.language_model.norm.weight": MLXRandom.normal([hidden]).asType(.bfloat16),
      "model.language_model.layers.0.input_layernorm.weight": inputNorm,
      "model.language_model.layers.0.linear_attn.norm.weight": gatedNorm,
    ]
    arrays["model.language_model.layers.0.post_attention_layernorm.weight"] =
      MLXRandom.normal([hidden]).asType(.bfloat16)
    for name in ["gate_proj", "up_proj"] {
      arrays["model.language_model.layers.0.mlp.\(name).weight"] =
        MLXRandom.normal([intermediate, hidden]).asType(.bfloat16)
    }
    arrays["model.language_model.layers.0.mlp.down_proj.weight"] =
      MLXRandom.normal([hidden, intermediate]).asType(.bfloat16)
    try MLX.save(arrays: arrays, url: source.appending(path: "model.safetensors"))

    let config: [String: Any] = [
      "model_type": "qwen3_5",
      "text_config": [
        "model_type": "qwen3_5_text", "hidden_size": hidden,
        "intermediate_size": intermediate, "num_hidden_layers": 1,
        "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": 64,
        "rms_norm_eps": 1e-6, "vocab_size": vocab, "max_position_embeddings": 4096,
        "tie_word_embeddings": false,
        "layer_types": ["linear_attention"],
        "rope_parameters": ["rope_theta": 10000, "type": "default"],
      ],
    ]
    try JSONSerialization.data(withJSONObject: config)
      .write(to: source.appending(path: "config.json"))

    _ = try Quantizer(
      source: try SourceCheckpoint(directory: source), profile: .balanced, destination: output
    ).run()

    let store = try WeightStore(directory: output)
    let written = try store("language_model.model.layers.0.input_layernorm.weight")
    #expect(written.asType(.float32).asArray(Float.self) == [0.75, 1.5])

    let gated = try store("language_model.model.layers.0.linear_attn.norm.weight")
    #expect(gated.asType(.float32).asArray(Float.self) == [0.75, 1.25])
  }

  @Test("a sharded source is read the same as a single-file one")
  func shardedSource() throws {
    let root = temp()
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appending(path: "src")
    try writeSource(at: source, shards: 3)

    let checkpoint = try SourceCheckpoint(directory: source)
    #expect(checkpoint.layerCount == 2)
    let outcome = try Quantizer(
      source: checkpoint, profile: .balanced, destination: root.appending(path: "out")
    ).run()
    #expect(outcome.byteCount > 0)
    try BonsaiConfig.load(directory: root.appending(path: "out")).validate()
  }
}
