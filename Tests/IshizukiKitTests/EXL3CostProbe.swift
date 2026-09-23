// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Probe, not a test: what one decode token's EXL3 projections cost, each on its own.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

@Suite("EXL3CostProbe", .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_EXL3"] != nil))
struct EXL3CostProbe {
  @Test("probe")
  func probe() throws {
    let url = URL(filePath: ProcessInfo.processInfo.environment["ISHIZUKI_EXL3"]!)
    let store = try WeightStore(directory: url).canonical(zeroCentredNorms: true)
    let keys = store.names(prefix: "language_model.").filter {
      $0.hasSuffix(".trellis") && !$0.contains(".mtp.")
    }.map { String($0.dropLast(".trellis".count)) }
    let tensors = try keys.map { try #require(try EXL3Tensor(store: store, key: $0)) }
    eval(tensors.flatMap { [$0.trellis, $0.inputScales, $0.outputScales] })

    func time(_ label: String, _ body: (EXL3Tensor, MLXArray) -> MLXArray) {
      let inputs = tensors.map { MLXRandom.normal([1, $0.inputDim]).asType(.float16) }
      eval(inputs)
      for _ in 0..<2 { eval(zip(tensors, inputs).map { body($0, $1) }) }
      let reps = 5
      let start = Date()
      for _ in 0..<reps { eval(zip(tensors, inputs).map { body($0, $1) }) }
      let ms = Date().timeIntervalSince(start) / Double(reps) * 1000
      print(String(format: "%-10@ %7.1f ms per token over %d projections", label, ms, tensors.count))
    }
    let weights = tensors.reduce(0) { $0 + $1.inputDim * $1.outputDim }
    print("weights \(weights / 1_000_000)M")
    for _ in 0..<3 {
      time("matvec") { EXL3Kernels.matvec($1, $0)! }
      time("rot fused") { t, x in
        EXL3Kernels.rotate(x, t.inputScales, before: true, to: .float16)
      }
      time("rot mlx") { t, x in
        EXL3Kernels.rotate(x, t.inputScales, before: true, to: .float16, fused: false)
      }
      time("apply") { EXL3Kernels.apply($1, $0) }
    }
    var byShape: [String: (Int, Double)] = [:]
    for t in tensors {
      let x = MLXRandom.normal([1, t.inputDim]).asType(.float16)
      eval(EXL3Kernels.matvec(x, t)!)
      let start = Date()
      eval((0..<16).map { _ in EXL3Kernels.matvec(x, t)! })
      let ms = Date().timeIntervalSince(start) / 16 * 1000
      let key = "\(t.inputDim)x\(t.outputDim)@\(t.bits)"
      byShape[key] = ((byShape[key]?.0 ?? 0) + 1, (byShape[key]?.1 ?? 0) + ms)
    }
    for (key, (count, ms)) in byShape.sorted(by: { $0.value.1 > $1.value.1 }) {
      print(String(format: "%-22@ x%3d  %7.2f ms total  %6.3f ms each", key, count, ms, ms / Double(count)))
    }
  }
}
