// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit
import MLX
import MLXRandom

struct KernelCheck: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "kernel-check",
    abstract: "Verify the qmv_wide kernel against MLX's quantized matmul, and time it.")

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .long) var iterations: Int = 40

  func run() throws {
    let packURL = URL(filePath: model)
    let config = try BonsaiConfig.load(directory: packURL)
    try config.validate()
    let store = try WeightStore(directory: packURL)
    let prefix = store.has("language_model.model.norm.weight") ? "language_model." : ""
    let factory = PackedModuleFactory(store: store, config: config, tensorPrefix: prefix)

    let cases: [(String, String)] = [
      ("mlp.gate_proj  5120 -> 17408", "model.layers.0.mlp.gate_proj"),
      ("self_attn.o_proj 6144 -> 5120", "model.layers.3.self_attn.o_proj"),
      ("mlp.down_proj 17408 -> 5120", "model.layers.0.mlp.down_proj"),
    ]

    var failures = 0
    MLXRandom.seed(7)

    for (label, path) in cases {
      let linear = try factory.linear(path)
      print("\n\(label)")

      for m in QMVWide.supportedBatch {
        let x = MLXRandom.normal([m, linear.inputDim]).asType(.float16)

        let reference = quantizedMM(
          x, linear.weight, scales: linear.scales, biases: linear.biases,
          transpose: true, groupSize: linear.groupSize, bits: linear.bits,
          mode: .affine)
        eval(reference)

        guard
          let got = QMVWide.apply(
            x, linear.weight, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits)
        else {
          print("  M=\(m)  kernel declined this shape")
          continue
        }
        eval(got)

        let a = got.asType(.float32)
        let b = reference.asType(.float32)
        let maxAbs = abs(a - b).max().item(Float.self)
        let scale = max(abs(b).max().item(Float.self), 1e-6)
        let relative = maxAbs / scale
        let ok = relative < 5e-3
        if !ok { failures += 1 }

        let referenceSeconds = time(iterations) {
          let y = quantizedMM(
            x, linear.weight, scales: linear.scales, biases: linear.biases,
            transpose: true, groupSize: linear.groupSize, bits: linear.bits,
            mode: .affine)
          eval(y)
        }
        let kernelSeconds = time(iterations) {
          if let y = QMVWide.apply(
            x, linear.weight, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits)
          {
            eval(y)
          }
        }

        print(
          String(
            format: "  M=%d  %@  rel %.5f   mlx %6.2f ms   qmv_wide %6.2f ms  (%.2fx)",
            m, ok ? "PASS" : "FAIL", relative,
            referenceSeconds * 1000, kernelSeconds * 1000,
            referenceSeconds / kernelSeconds))
      }
    }

    failures += checkFusedHadamard(factory: factory)
    breakdown(factory: factory)

    print("")
    if failures == 0 {
      print("qmv_wide matches MLX's quantized matmul at every supported batch size.")
    } else {
      print("\(failures) case(s) diverged.")
      throw ExitCode.failure
    }
  }

  private func checkFusedHadamard(factory: PackedModuleFactory) -> Int {
    print("\n\nFused Hadamard rotation vs the op-based path")
    var failures = 0
    for (label, path) in [
      ("width  5120", "model.layers.0.mlp.gate_proj"),
      ("width  6144", "model.layers.3.self_attn.o_proj"),
      ("width 17408", "model.layers.0.mlp.down_proj"),
    ] {
      guard let linear = try? factory.linear(path), let signs = linear.signs else { continue }
      for m in [1, 4] {
        let x = MLXRandom.normal([m, linear.inputDim]).asType(.float16)

        let reference = hadamardRotate(
          x, block: linear.block, signs: signs, inverse: false)
        eval(reference)
        guard let fused = FusedHadamard.apply(x, block: linear.block, signs: signs) else {
          print("  \(label) M=\(m)  kernel declined")
          continue
        }
        eval(fused)

        let delta = abs(fused.asType(.float32) - reference.asType(.float32))
          .max().item(Float.self)
        let scale = max(abs(reference.asType(.float32)).max().item(Float.self), 1e-6)
        let ok = delta / scale < 5e-3
        if !ok { failures += 1 }

        let referenceSeconds = time(iterations) {
          eval(hadamardRotate(x, block: linear.block, signs: signs, inverse: false))
        }
        let fusedSeconds = time(iterations) {
          if let y = FusedHadamard.apply(x, block: linear.block, signs: signs) {
            eval(y)
          }
        }
        print(
          String(
            format: "  %@ M=%d  %@  rel %.5f   ops %6.3f ms   fused %6.3f ms  (%.2fx)",
            label, m, ok ? "PASS" : "FAIL", delta / scale,
            referenceSeconds * 1000, fusedSeconds * 1000,
            referenceSeconds / fusedSeconds))
      }
    }
    return failures
  }

  private func breakdown(factory: PackedModuleFactory) {
    guard let linear = try? factory.linear("model.layers.0.mlp.gate_proj"),
      let signs = linear.signs
    else { return }

    print("\n\nPer-projection cost breakdown (gate_proj 5120 -> 17408)")
    print("  M   rotation    matmul     total   rotation share")

    for m in [1, 2, 4, 8] {
      let x = MLXRandom.normal([m, linear.inputDim]).asType(.float16)

      let rotationSeconds = time(iterations) {
        let r = hadamardRotate(
          x, block: linear.block, signs: signs, inverse: false)
        eval(r)
      }
      let rotated = hadamardRotate(
        x, block: linear.block, signs: signs, inverse: false)
      eval(rotated)

      let matmulSeconds = time(iterations) {
        let y = quantizedMM(
          rotated, linear.weight, scales: linear.scales, biases: linear.biases,
          transpose: true, groupSize: linear.groupSize, bits: linear.bits,
          mode: .affine)
        eval(y)
      }
      let totalSeconds = time(iterations) {
        let y = linear(x)
        eval(y)
      }

      print(
        String(
          format: "  %d  %7.3f ms %7.3f ms %7.3f ms   %4.0f%%",
          m, rotationSeconds * 1000, matmulSeconds * 1000, totalSeconds * 1000,
          rotationSeconds / totalSeconds * 100))
    }
  }

  private func time(_ iterations: Int, _ body: () -> Void) -> Double {
    body()
    let start = Date()
    for _ in 0..<iterations { body() }
    return -start.timeIntervalSinceNow / Double(iterations)
  }
}
