// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

// Probe, not a test: the few-row GGUF multiply against the matvec, per format in a real file.
@Suite("GGMLFewRowProbe", .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_GGUF"] != nil))
struct GGMLFewRowProbe {
  @Test("probe")
  func probe() throws {
    let model = try BonsaiModel(path: URL(filePath: ProcessInfo.processInfo.environment["ISHIZUKI_GGUF"]!))
    var byType: [GGMLType: GGUFBlocks] = [:]
    var seen = Set<ObjectIdentifier>()
    func walk(_ value: Any, depth: Int) {
      guard depth < 12 else { return }
      if let linear = value as? PackedLinear {
        if let g = linear.ggml, g.type.isQuantized,
          (byType[g.type].map { $0.outputDim * $0.shape.last! } ?? 0) < g.outputDim * g.shape.last!
        {
          byType[g.type] = g
        }
        return
      }
      if type(of: value) is AnyClass {
        let id = ObjectIdentifier(value as AnyObject)
        if seen.contains(id) { return }
        seen.insert(id)
      }
      for child in Mirror(reflecting: value).children { walk(child.value, depth: depth + 1) }
    }
    walk(model.text, depth: 0)

    func time(_ body: () -> MLXArray) -> Double {
      eval(body())
      var best = Double.infinity
      for _ in 0..<4 {
        let start = Date()
        eval((0..<8).map { _ in body() })
        best = min(best, -start.timeIntervalSinceNow / 8)
      }
      return best * 1e6
    }

    let rowBlocks = Int(ProcessInfo.processInfo.environment["ISHIZUKI_RB"] ?? "") ?? 1
    for (type, blocks) in byType.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
      let k = blocks.shape.last!
      let n = blocks.outputDim
      let single = time {
        GGMLKernels.matvec(
          MLXRandom.normal([1, k]).asType(.float16), blocks: blocks.bytes, type: type,
          outputDim: n)!
      }
      var line = String(format: "%-8@ %5dx%-5d m1 %6.0fus |", "\(type)", n, k, single)
      for m in [2, 4, 8] {
        let x = MLXRandom.normal([m, k]).asType(.float16)
        eval(x)
        let old = GGMLKernels.matvec(x, blocks: blocks.bytes, type: type, outputDim: n)!
        guard
          let new = GGMLKernels.matmulFew(
            x, blocks: blocks.bytes, type: type, outputDim: n, rowBlocks: rowBlocks)
        else {
          line += " m\(m) unsupported |"
          continue
        }
        let err = (abs(new.asType(.float32) - old.asType(.float32)).max()
          / abs(old.asType(.float32)).max()).item(Float.self)
        let tOld = time { GGMLKernels.matvec(x, blocks: blocks.bytes, type: type, outputDim: n)! }
        let tNew = time {
          GGMLKernels.matmulFew(x, blocks: blocks.bytes, type: type, outputDim: n, rowBlocks: rowBlocks)!
        }
        line += String(format: " m%d old %4.2fx new %4.2fx e%.0e |", m, tOld / single, tNew / single, err)
      }
      print(line)
    }
  }
}
