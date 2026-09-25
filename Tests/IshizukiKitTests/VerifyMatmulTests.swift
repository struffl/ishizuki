// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The few-row verify matmul, against MLX's own quantized product.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

@Suite("Verify matmul")
struct VerifyMatmulTests {
  @Test(
    "multiplies a few rows as MLX's quantized matmul does",
    arguments: [(2, 128), (4, 128), (8, 128), (4, 64), (3, 128)])
  func matchesQuantizedMM(bits: Int, groupSize: Int) throws {
    let n = 384
    let k = 1024
    let dense = MLXRandom.normal([n, k]) * 0.05
    let (weight, scales, biases) = quantized(dense, groupSize: groupSize, bits: bits)
    for (rows, dtype, wide) in [
      (8, DType.float16, DType.float32), (5, .float16, .float32), (7, .float32, .float32),
      (6, .float32, .float32), (8, .float16, .bfloat16), (6, .float32, .bfloat16),
      (7, .bfloat16, .bfloat16), (8, .bfloat16, .float16),
    ] {
      let x = (MLXRandom.normal([rows, k]) * 2).asType(dtype)
      let scales = scales.asType(wide)
      let biases = biases?.asType(wide)
      let want = quantizedMM(
        x.asType(.float32), weight, scales: scales.asType(.float32),
        biases: biases!.asType(.float32), transpose: true, groupSize: groupSize, bits: bits,
        mode: .affine)
      let got = try #require(
        VerifyMatmul.applyAny(
          x, weight, scales: scales, biases: biases!, groupSize: groupSize, bits: bits))
      let worst = (got.asType(.float32) - want).abs().max().item(Float.self)
      let scale = want.abs().max().item(Float.self)
      let tolerance: Float =
        dtype == .bfloat16 ? 8e-3 : dtype == .float32 && wide == .float32 ? 1e-5 : 2e-3
      #expect(
        worst / scale < tolerance,
        "\(bits)-bit, group \(groupSize), \(rows) rows of \(dtype), scales \(wide): off by \(worst / scale)"
      )
    }
  }

  /// A target in fp16 and a drafter in fp32 can share a projection shape, and one graph then
  /// holds both. Each must run the kernel built for its own dtype.
  @Test("keeps one shape's dtypes apart in a single graph")
  func mixedDTypes() throws {
    let n = 384
    let k = 1024
    let dense = MLXRandom.normal([n, k]) * 0.05
    let (weight, scales, biases) = quantized(dense, groupSize: 128, bits: 2)
    let x = MLXRandom.normal([8, k])
    let want = quantizedMM(
      x, weight, scales: scales, biases: biases!, transpose: true, groupSize: 128, bits: 2,
      mode: .affine)
    let dtypes = [DType.float16, .float32, .bfloat16]
    let outputs = try dtypes.map {
      try #require(
        VerifyMatmul.applyAny(
          x.asType($0), weight, scales: scales, biases: biases!, groupSize: 128, bits: 2))
    }
    eval(outputs)
    let scale = want.abs().max().item(Float.self)
    for (dtype, got) in zip(dtypes, outputs) {
      let worst = (got.asType(.float32) - want).abs().max().item(Float.self)
      #expect(worst / scale < 8e-3, "\(dtype) beside the others: off by \(worst / scale)")
    }
  }

  @Test("takes a shape only from the row count where it beats MLX")
  func rowFloor() {
    func rows(_ n: Int, _ k: Int, bits: Int = 2) -> Int {
      VerifyMatmul.minimumRows(bytes: n * k * bits / 8)
    }
    #expect(rows(248_320, 5120) == 2)
    #expect(rows(17_408, 5120) == 3)
    #expect(rows(12_288, 5120) == 3)
    #expect(rows(6144, 5120) == 4)
    #expect(rows(1024, 5120) == 6)
    #expect(rows(4096, 5120, bits: 8) == 3)
    #expect(rows(1024, 5120, bits: 8) == 4)
  }

  /// Throughput on the Qwen3.8-27B shapes, eight rows of float16, as a probe.
  @Test(
    "keeps the arithmetic units busy",
    .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_VERIFY_BENCH"] != nil))
  func throughput() throws {
    let shapes = (ProcessInfo.processInfo.environment["ISHIZUKI_VERIFY_SHAPES"] ?? "4x2")
      .split(separator: ",").map { $0.split(separator: "x").compactMap { Int($0) } }
    defer {
      VerifyMatmul.rowBlocks = 4
      VerifyMatmul.simdgroups = 2
    }
    for shape in shapes {
      VerifyMatmul.rowBlocks = shape[0]
      VerifyMatmul.simdgroups = shape[1]
      print("row blocks \(shape[0]), simdgroups \(shape[1])")
      for (n, k) in [
        (17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144), (1024, 5120),
      ] {
        let (weight, scales, biases) = quantized(
          MLXRandom.normal([n, k]) * 0.05, groupSize: 128, bits: 2)
        let s = scales.asType(.float16)
        let b = biases!.asType(.float16)
        eval(weight, s, b)
        for rows in [8] {
          let x = MLXRandom.normal([rows, k]).asType(.float16)
          let reps = n > 100_000 ? 8 : 48
          func run() -> MLXArray {
            var total = MLXArray.zeros([rows, n], dtype: .float16)
            for _ in 0..<reps {
              let y =
                rows == 1
                ? quantizedMM(
                  x, weight, scales: s, biases: b, transpose: true, groupSize: 128, bits: 2,
                  mode: .affine)
                : VerifyMatmul.applyAny(x, weight, scales: s, biases: b, groupSize: 128, bits: 2)!
              total = total + y
            }
            return total
          }
          for _ in 0..<2 { eval(run()) }
          let start = Date()
          for _ in 0..<5 { eval(run()) }
          let each = -start.timeIntervalSinceNow / Double(5 * reps)
          let tflops = Double(2 * rows * n * k) / each / 1e12
          let gbs = Double(n * k / 4 + n * (k / 128) * 4) / each / 1e9
          print(
            String(
              format: "%6d x %5d, %d rows: %8.1f us  %5.2f TFLOPS  %6.1f GB/s", n, k, rows,
              each * 1e6, tflops, gbs))
        }
      }
    }
  }

  /// The real model's shapes, which the small cases above do not reach.
  @Test(
    "multiplies the model's own shapes",
    .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_VERIFY_BENCH"] != nil),
    arguments: [(4, 64), (8, 128), (2, 128), (4, 128)])
  func matchesAtScale(bits: Int, groupSize: Int) throws {
    for (n, k) in [
      (1024, 5120), (6144, 5120), (5120, 6144), (10240, 5120), (17408, 5120), (5120, 17408),
      (248320, 5120), (4096, 5120), (5120, 4096), (1280, 5120), (256, 5120), (5120, 25600),
    ] {
      let (weight, wideScales, wideBiases) = quantized(
        MLXRandom.normal([n, k]) * 0.05, groupSize: groupSize, bits: bits)
      let scales = wideScales.asType(.bfloat16)
      let biases = wideBiases?.asType(.bfloat16)
      for rows in [5, 7, 8] {
        let x = MLXRandom.normal([rows, k]).asType(.float16)
        let want = quantizedMM(
          x.asType(.float32), weight, scales: scales.asType(.float32),
          biases: biases!.asType(.float32), transpose: true, groupSize: groupSize, bits: bits,
          mode: .affine)
        guard
          let got = VerifyMatmul.applyAny(
            x, weight, scales: scales, biases: biases!, groupSize: groupSize, bits: bits)
        else {
          print("\(bits)-bit g\(groupSize) \(n)x\(k) \(rows) rows: not taken")
          continue
        }
        eval(got)
        let off =
          (got.asType(.float32) - want).abs().max().item(Float.self)
          / want.abs().max().item(Float.self)
        print("\(bits)-bit g\(groupSize) \(n)x\(k) \(rows) rows: off by \(off)")
        #expect(off < 2e-3)
      }
    }
  }
}
