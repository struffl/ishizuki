// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// EXL3 decode held to the exllamav3 reconstruction, and to the checkpoints it was cut from.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

@Suite("EXL3")
struct EXL3Tests {
  private static func fixture() throws -> [String: MLXArray] {
    let url = try #require(
      Bundle.module.url(
        forResource: "exl3-reference", withExtension: "safetensors", subdirectory: "Fixtures")
        ?? Bundle.module.url(forResource: "exl3-reference", withExtension: "safetensors"))
    return try loadArrays(url: url)
  }

  private static let rates: [(String, EXL3Tensor.Codebook)] = [
    ("r1c0", .threeInst), ("r2c1", .mcg), ("r3c2", .mul1), ("r4c0", .threeInst),
    ("r5c2", .mul1), ("r6c1", .mcg), ("r7c2", .mul1), ("r8c0", .threeInst),
    ("r1h5c2", .mul1), ("r2h5c2", .mul1), ("r3h5c2", .mul1),
  ]

  private static func tensor(
    _ arrays: [String: MLXArray], _ name: String, _ codebook: EXL3Tensor.Codebook
  ) throws -> EXL3Tensor {
    let trellis = try #require(arrays[name + ".trellis"])
    let k = trellis.dim(0) * 16
    let n = trellis.dim(1) * 16
    let paddedK = (k + 127) / 128 * 128
    let paddedN = (n + 127) / 128 * 128
    let padded = padded(
      trellis, widths: [[0, (paddedK - k) / 16], [0, (paddedN - n) / 16], 0])
    return try EXL3Tensor(
      trellis: padded, inputScales: MLXArray.ones([paddedK], dtype: .float16),
      outputScales: MLXArray.ones([paddedN], dtype: .float16), codebook: codebook)
  }

  @Test("the Metal decode is bit-exact against the reference at every rate and codebook")
  func decodeMatchesReference() throws {
    let arrays = try Self.fixture()
    for (name, codebook) in Self.rates {
      let t = try Self.tensor(arrays, name, codebook)
      let expected = try #require(arrays[name + ".expected"])
      let w = try #require(EXL3Kernels.dequantize(t, dtype: .float32))
      let actual = w[0..<expected.dim(0), 0..<expected.dim(1)]
      let worst = abs(actual - expected).max().item(Float.self)
      #expect(worst == 0, "\(name) is off by \(worst)")
    }
  }

  @Test("the fused matvec agrees with the expanded weight for every row count")
  func matvecMatchesExpanded() throws {
    let arrays = try Self.fixture()
    for (name, codebook) in Self.rates {
      let t = try Self.tensor(arrays, name, codebook)
      let w = try #require(EXL3Kernels.dequantize(t, dtype: .float32))
      for m in EXL3Kernels.matvecRows {
        let x = MLXRandom.normal([m, t.inputDim])
        let y = try #require(EXL3Kernels.matvec(x, t))
        let expected = matmul(x, w)
        let bound = matmul(abs(x), abs(w)).max().item(Float.self)
        let worst = abs(y - expected).max().item(Float.self) / bound
        #expect(worst < 1e-5, "\(name) at \(m) rows is off by \(worst)")
      }
    }
  }

  @Test("a decoded projection reproduces the checkpoint it was quantized from")
  func projectionMatchesCheckpoint() throws {
    let arrays = try Self.fixture()
    for name in ["qwen3", "qwen38"] {
      let store = WeightStore(arrays: arrays)
      let t = try #require(try EXL3Tensor(store: store, key: name))
      #expect(t.codebook == (name == "qwen38" ? .mul1 : .threeInst))
      let reference = try #require(arrays[name + ".weight"]).asType(.float32)
      let linear = PackedLinear(exl3: t)
      let batch = MLXRandom.normal([40, t.inputDim], key: MLXRandom.key(3)).asType(.float16)
      var previous: MLXArray?
      for rows in [1, 5, 12, 40] {
        let x = batch[0..<rows]
        let y = linear(x).asType(.float32)
        let expected = matmul(x.asType(.float32), reference)
        let error = sqrt(sum(square(y - expected)) / sum(square(expected))).item(Float.self)
        #expect(error < 0.12, "\(name) at \(rows) rows drifts \(error) from the checkpoint")
        if let previous {
          let agree = abs(y[0..<1] - previous[0..<1]).max().item(Float.self)
          #expect(agree < 2e-2, "\(name) at \(rows) rows disagrees with fewer rows by \(agree)")
        }
        previous = y
      }
    }
  }

  @Test("the fused rotation agrees with MLX's Hadamard on both sides of the matvec")
  func rotationMatchesHadamard() {
    for (rows, width) in [(1, 5120), (7, 1024), (40, 256)] {
      let x = MLXRandom.normal([rows, width], key: MLXRandom.key(5))
      let s = MLXRandom.normal([width], key: MLXRandom.key(6)).asType(.float16)
      let scale = 1 / Float(128).squareRoot()
      let before = hadamardTransform((x * s.asType(.float32)).reshaped([-1, 128]), scale: scale)
        .reshaped([rows, width])
      let after = hadamardTransform(x.reshaped([-1, 128]), scale: scale)
        .reshaped([rows, width]) * s.asType(.float32)
      let pre = EXL3Kernels.rotate(x, s, before: true, to: .float32)
      let post = EXL3Kernels.rotate(x, s, before: false, to: .float32)
      #expect(abs(pre - before).max().item(Float.self) < 1e-4)
      #expect(abs(post - after).max().item(Float.self) < 1e-4)
    }
  }

  @Test("an exllamav3 config reads as an EXL3 pack")
  func configReadsAsEXL3() throws {
    let object: [String: Any] = [
      "model_type": "qwen3",
      "hidden_size": 128, "num_hidden_layers": 1, "num_attention_heads": 2,
      "num_key_value_heads": 1, "head_dim": 64, "intermediate_size": 256,
      "vocab_size": 64, "rms_norm_eps": 1e-6, "max_position_embeddings": 4096,
      "quantization_config": [
        "quant_method": "exl3", "version": "1.5.0", "bits": 4.0, "head_bits": 6,
        "calibration": ["rows": 250, "cols": 2048], "codebook": "mul1",
      ],
    ]
    let config = try BonsaiConfig.standard(object)
    #expect(config.profile == .exl3)
    #expect(config.quantization.bits == 4)
    try config.validate()
  }

  @Test("an EXL3 projection refuses to be split by channel")
  func refusesChannelSplit() throws {
    let arrays = try Self.fixture()
    let t = try #require(try EXL3Tensor(store: WeightStore(arrays: arrays), key: "qwen3"))
    #expect(throws: BonsaiError.self) { try PackedLinear(exl3: t).channels(from: 64) }
  }
}
