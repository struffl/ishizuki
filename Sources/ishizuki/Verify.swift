// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import IshizukiKit
import MLX

struct Verify: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "verify",
    abstract: "Check the Hadamard + 2-bit path against Python golden values.")

  @Option(name: .long, help: "Path to the MLX pack directory.")
  var model: String = defaultModelPath

  @Option(name: .long, help: "Path to hadamard_golden.safetensors.")
  var golden: String = "Tests/Golden/hadamard_golden.safetensors"

  func run() throws {
    let packURL = URL(filePath: model)
    let config = try BonsaiConfig.load(directory: packURL)
    try config.validate()

    let store = try WeightStore(directory: packURL)
    let prefix = config.components?.vision == true ? "language_model." : ""
    let factory = PackedModuleFactory(store: store, config: config, tensorPrefix: prefix)

    let goldenArrays = try loadArrays(url: URL(filePath: golden))
    var failures = 0

    func compare(_ label: String, _ got: MLXArray, _ want: MLXArray, tolerance: Float) {
      let a = got.asType(.float32)
      let b = want.asType(.float32)
      let maxAbs = abs(a - b).max().item(Float.self)
      let scale = max(abs(b).max().item(Float.self), 1e-6)
      let relative = maxAbs / scale
      let ok = relative <= tolerance
      if !ok { failures += 1 }
      print(
        "  " + Style.muted(label.padding(toLength: 34, withPad: " ", startingAt: 0))
          + (ok ? Style.good("PASS") : Style.bad("FAIL"))
          + Style.faint(String(format: "  max|Δ|=%.5f  rel=%.6f", maxAbs, relative)))
    }

    print(Style.banner("verify"))
    print("")
    print(Style.bright("Hadamard rotation + 2-bit matmul"))
    let cases: [(String, String, String)] = [
      ("layer0 mlp.gate_proj (5120)", "model.layers.0.mlp.gate_proj", "case1"),
      ("layer3 self_attn.o_proj (6144)", "model.layers.3.self_attn.o_proj", "case2"),
      ("layer0 mlp.down_proj (17408)", "model.layers.0.mlp.down_proj", "case3"),
    ]
    for (label, path, key) in cases {
      let linear = try factory.linear(path)
      let x = goldenArrays[key + "_x"]!
      compare(label, linear(x), goldenArrays[key + "_y"]!, tolerance: 2e-3)
    }

    let gate = try factory.linear("model.layers.0.mlp.gate_proj")
    let rotated = hadamardRotate(
      goldenArrays["case1_x"]!, block: gate.block, signs: gate.signs!, inverse: false)
    compare("rotation only (5120)", rotated, goldenArrays["case1_hadamard"]!, tolerance: 1e-3)

    print(Style.bright("Inverse-rotated embedding"))
    let embedding = try factory.embedding("model.embed_tokens")
    let ids = goldenArrays["case4_ids"]!
    compare("embed_tokens lookup", embedding(ids), goldenArrays["case4_y"]!, tolerance: 2e-3)

    print("")
    if failures == 0 {
      print(Style.good("All checks passed."))
    } else {
      print(Style.bad("\(failures) check(s) failed."))
      throw ExitCode.failure
    }
  }
}

let defaultModelPath =
  applicationSupportDirectory
  .appending(path: "Ishizuki/models/Ternary-Bonsai-2-27B-mlx-2bit").path

let defaultRepo = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"

private let applicationSupportDirectory =
  FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
  ?? URL(filePath: NSHomeDirectory()).appending(path: "Library/Application Support")
