// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom

/// Verify the qmv_wide kernel against MLX's quantized matmul, and time it.
public struct KernelCheck: Sendable {
  public struct Options: Sendable {
    public var model: URL
    public var iterations: Int = 40

    public init(model: URL) { self.model = model }
  }

  public static func run(
    _ options: Options,
    log: @escaping @Sendable (String) -> Void
  ) throws {
    let packURL = options.model
    let config = try BonsaiConfig.load(directory: packURL)
    try config.validate()
    let store = try WeightStore(directory: packURL)
    let prefix = store.languageModelPrefix
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
      log("\n\(label)")

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
          log("  M=\(m)  kernel declined this shape")
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

        let referenceSeconds = time(options.iterations) {
          let y = quantizedMM(
            x, linear.weight, scales: linear.scales, biases: linear.biases,
            transpose: true, groupSize: linear.groupSize, bits: linear.bits,
            mode: .affine)
          eval(y)
        }
        let kernelSeconds = time(options.iterations) {
          if let y = QMVWide.apply(
            x, linear.weight, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits)
          {
            eval(y)
          }
        }

        log(
          String(
            format: "  M=%d  %@  rel %.5f   mlx %6.2f ms   qmv_wide %6.2f ms  (%.2fx)",
            m, ok ? "PASS" : "FAIL", relative,
            referenceSeconds * 1000, kernelSeconds * 1000,
            referenceSeconds / kernelSeconds))
      }
    }

    failures += checkFusedHadamard(factory: factory, options: options, log: log)
    breakdown(factory: factory, options: options, log: log)

    log("")
    if failures == 0 {
      log("qmv_wide matches MLX's quantized matmul at every supported batch size.")
    } else {
      log("\(failures) case(s) diverged.")
      throw BenchFailure(reason: "\(failures) kernel case(s) diverged from the reference path")
    }
  }

  private static func checkFusedHadamard(
    factory: PackedModuleFactory, options: Options, log: (String) -> Void
  ) -> Int {
    log("\n\nFused Hadamard rotation vs the op-based path")
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
          log("  \(label) M=\(m)  kernel declined")
          continue
        }
        eval(fused)

        let delta = abs(fused.asType(.float32) - reference.asType(.float32))
          .max().item(Float.self)
        let scale = max(abs(reference.asType(.float32)).max().item(Float.self), 1e-6)
        let ok = delta / scale < 5e-3
        if !ok { failures += 1 }

        let referenceSeconds = time(options.iterations) {
          eval(hadamardRotate(x, block: linear.block, signs: signs, inverse: false))
        }
        let fusedSeconds = time(options.iterations) {
          if let y = FusedHadamard.apply(x, block: linear.block, signs: signs) {
            eval(y)
          }
        }
        log(
          String(
            format: "  %@ M=%d  %@  rel %.5f   ops %6.3f ms   fused %6.3f ms  (%.2fx)",
            label, m, ok ? "PASS" : "FAIL", delta / scale,
            referenceSeconds * 1000, fusedSeconds * 1000,
            referenceSeconds / fusedSeconds))
      }
    }
    return failures
  }

  private static func breakdown(
    factory: PackedModuleFactory, options: Options, log: (String) -> Void
  ) {
    guard let linear = try? factory.linear("model.layers.0.mlp.gate_proj"),
      let signs = linear.signs
    else { return }

    log("\n\nPer-projection cost breakdown (gate_proj 5120 -> 17408)")
    log("  M   rotation    matmul     total   rotation share")

    for m in [1, 2, 4, 8] {
      let x = MLXRandom.normal([m, linear.inputDim]).asType(.float16)

      let rotationSeconds = time(options.iterations) {
        let r = hadamardRotate(
          x, block: linear.block, signs: signs, inverse: false)
        eval(r)
      }
      let rotated = hadamardRotate(
        x, block: linear.block, signs: signs, inverse: false)
      eval(rotated)

      let matmulSeconds = time(options.iterations) {
        let y = quantizedMM(
          rotated, linear.weight, scales: linear.scales, biases: linear.biases,
          transpose: true, groupSize: linear.groupSize, bits: linear.bits,
          mode: .affine)
        eval(y)
      }
      let totalSeconds = time(options.iterations) {
        let y = linear(x)
        eval(y)
      }

      log(
        String(
          format: "  %d  %7.3f ms %7.3f ms %7.3f ms   %4.0f%%",
          m, rotationSeconds * 1000, matmulSeconds * 1000, totalSeconds * 1000,
          rotationSeconds / totalSeconds * 100))
    }
  }

  private static func time(_ iterations: Int, _ body: () -> Void) -> Double {
    body()
    let start = Date()
    for _ in 0..<iterations { body() }
    return -start.timeIntervalSinceNow / Double(iterations)
  }
}
