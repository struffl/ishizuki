// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

// Probe: how far a written pack's weights sit from the source they came from.
@Suite(
  "WeightProbe",
  .enabled(
    if: ProcessInfo.processInfo.environment["ISHIZUKI_SRC"] != nil
      && ProcessInfo.processInfo.environment["ISHIZUKI_PACK"] != nil))
struct WeightProbe {
  @Test("probe")
  func probe() throws {
    let env = ProcessInfo.processInfo.environment
    let source = try SourceCheckpoint(directory: URL(filePath: env["ISHIZUKI_SRC"]!))
    let pack = try WeightStore(directory: URL(filePath: env["ISHIZUKI_PACK"]!))
    let config = try BonsaiConfig.load(directory: URL(filePath: env["ISHIZUKI_PACK"]!))

    let pairs = [
      ("lm_head", "lm_head.weight", "language_model.lm_head"),
      (
        "L0.down_proj", "model.language_model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.mlp.down_proj"
      ),
      (
        "L0.q_proj", "model.language_model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj"
      ),
      (
        "embed", "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens"
      ),
    ]

    for (label, sourceName, packName) in pairs {
      guard let original = source.optional(sourceName),
        let wq = pack.optional(packName + ".weight"),
        let scales = pack.optional(packName + ".scales"),
        let biases = pack.optional(packName + ".biases")
      else {
        print("\(label): absent")
        continue
      }
      let entry = config.quantization.module(packName)
      let restored = dequantized(
        wq, scales: scales, biases: biases, groupSize: entry.groupSize, bits: entry.bits,
        mode: .affine
      ).asType(.float32)
      let reference = original.asType(.float32)
      let difference = restored - reference
      let error =
        sqrt((difference * difference).sum()).item(Float.self)
        / sqrt((reference * reference).sum()).item(Float.self)
      print(
        "\(label): \(entry.bits)-bit g\(entry.groupSize) relative error "
          + String(format: "%.4f", error)
          + "  src \(original.shape) \(original.dtype)  packed \(wq.shape)")
    }
  }
}
