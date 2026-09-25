// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// MLX's product, QMVWide and the verify kernel at two to eight rows, on chosen shapes.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

@Suite(
  "Few-row race", .serialized,
  .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_RACE"] != nil))
struct FewRowRaceProbe {
  @Test("times QMVWide, VerifyMatmul and MLX at 2-8 rows")
  func race() throws {
    let pairs = ProcessInfo.processInfo.environment["ISHIZUKI_RACE_SHAPES"].map { $0.split(separator: ",").map { p -> (Int, Int) in let v = p.split(separator: "x").map { Int($0)! }; return (v[1], v[0]) } } ?? [(17408, 5120), (5120, 17408), (248320, 5120), (6144, 5120)]
    for (n, k) in pairs {
      let dense = MLXRandom.normal([n, k]) * 0.02
      let bits = Int(ProcessInfo.processInfo.environment["ISHIZUKI_RACE_BITS"] ?? "2")!
      let xt: DType = ProcessInfo.processInfo.environment["ISHIZUKI_RACE_F32"] != nil ? .float32 : .float16
      let (w, s, b) = quantized(dense, groupSize: 128, bits: bits)
      let scales = s.asType(.float16)
      let biases = b!.asType(.float16)
      eval(w, scales, biases)
      if ProcessInfo.processInfo.environment["ISHIZUKI_RACE_WARM"] != nil { VerifyMatmul.warm(w, scales: scales, biases: biases, groupSize: 128, bits: bits) }
      var line = "\(k)x\(n)"
      let only = ProcessInfo.processInfo.environment["ISHIZUKI_RACE"] ?? "all"
      for m in 2...8 {
        let x = (MLXRandom.normal([m, k])).asType(xt)
        eval(x)
        func time(_ f: () -> MLXArray?) -> Double {
          guard f() != nil else { return .nan }
          for _ in 0..<5 { eval(f()!) }
          var best = Double.infinity
          for _ in 0..<5 {
            let start = Date()
            eval((0..<30).map { _ in f()! })
            best = min(best, Date().timeIntervalSince(start) / 30 * 1000)
          }
          return best
        }
        print("shape \(k)x\(n) m \(m) kernels \(only)", terminator: "")
        fflush(stdout)
        let mlx = only != "all" && only != "mlx" ? .nan : time {
          quantizedMM(
            x, w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: bits,
            mode: .affine)
        }
        let wide = only != "all" && only != "wide" ? .nan : time {
          QMVWide.supportedBatch.contains(m)
            ? QMVWide.apply(x, w, scales: scales, biases: biases, groupSize: 128, bits: bits) : nil
        }
        let verify = only != "all" && only != "verify" ? .nan : time {
          VerifyMatmul.applyAny(x, w, scales: scales, biases: biases, groupSize: 128, bits: bits)
        }
        print(" ok")
        line += String(format: " | M%d mlx %.2f wide %.2f verify %.2f", m, mlx, wide, verify)
      }
      print(line)
    }
  }
}

