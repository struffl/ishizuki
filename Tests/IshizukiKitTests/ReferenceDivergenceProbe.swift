// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

// Probe: runs each layer on the *reference's* own input rather than on this runtime's, so a
// layer that computes something slightly different is told apart from a layer that merely
// inherited the previous one's rounding. Both numbers are printed, isolated and accumulated.
//
//   Scripts/gen-mlx-hidden.py <model> "<prompt>" /tmp/io.bin
//   ISHIZUKI_MODEL=<model> ISHIZUKI_LAYERIO=/tmp/io.bin swift test --filter Divergence
@Suite(
  "ReferenceDivergenceProbe",
  .enabled(
    if: ProcessInfo.processInfo.environment["ISHIZUKI_MODEL"] != nil
      && ProcessInfo.processInfo.environment["ISHIZUKI_LAYERIO"] != nil))
struct ReferenceDivergenceProbe {
  @Test("probe")
  func probe() throws {
    let env = ProcessInfo.processInfo.environment
    let data = try Data(contentsOf: URL(filePath: env["ISHIZUKI_LAYERIO"]!))
    guard data.prefix(8) == Data("LAYERIO1".utf8) else {
      Issue.record("not a layer-io dump")
      return
    }

    var off = 8
    func u32() -> Int {
      defer { off += 4 }
      return Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: UInt32.self) })
    }
    let count = u32()
    let tokens = u32()
    off += 4 * tokens

    var reference: [String: MLXArray] = [:]
    for _ in 0..<count {
      let n = u32()
      let name = String(
        data: data[data.startIndex + off..<data.startIndex + off + n], encoding: .utf8)!
      off += n
      let rows = u32()
      let cols = u32()
      var values = [Float]()
      values.reserveCapacity(rows * cols)
      for _ in 0..<(rows * cols) {
        values.append(
          data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: off, as: Float.self) })
        off += 4
      }
      reference[name] = MLXArray(values, [1, rows, cols])
    }

    let model = try BonsaiModel(path: URL(filePath: env["ISHIZUKI_MODEL"]!))
    let depth = count - 1
    let compute = reference["in0"]!.dtype
    let mask = causalMask(length: tokens, offset: 0, dtype: .bfloat16)

    func agreement(_ got: MLXArray, _ want: MLXArray) -> Float {
      let a = got.asType(.float32)
      let b = want.asType(.float32)
      eval(a, b)
      let covariance = ((a - a.mean()) * (b - b.mean())).mean().item(Float.self)
      let deviation = (a.variance().sqrt() * b.variance().sqrt()).item(Float.self)
      return covariance / max(deviation, 1e-9)
    }

    for i in 0..<depth {
      let given = reference[i == 0 ? "in0" : "out\(i - 1)"]!
      let isolated = model.text.layers[i](
        given.asType(.float32), mask: mask, cache: nil, positions: nil, compute: compute)
      print(
        String(
          format: "layer %2d isolated %.8f", i, agreement(isolated, reference["out\(i)"]!)))
    }

    var h = reference["in0"]!.asType(.float32)
    for i in 0..<depth {
      h = model.text.layers[i](h, mask: mask, cache: nil, positions: nil, compute: compute)
      print(String(format: "layer %2d running  %.8f", i, agreement(h, reference["out\(i)"]!)))
    }
  }
}
