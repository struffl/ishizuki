// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

// Probe: walks the dense source model and a known-good pack side by side, a layer at a time,
// so the first place they part company is visible rather than inferred from the logits.
@Suite(
  "LayerDivergenceProbe",
  .enabled(
    if: ProcessInfo.processInfo.environment["ISHIZUKI_DENSE"] != nil
      && ProcessInfo.processInfo.environment["ISHIZUKI_REF"] != nil))
struct LayerDivergenceProbe {
  @Test("probe")
  func probe() throws {
    let env = ProcessInfo.processInfo.environment
    let sourceURL = URL(filePath: env["ISHIZUKI_DENSE"]!)
    let reference = try BonsaiModel(directory: URL(filePath: env["ISHIZUKI_REF"]!), loadVision: false)
    let dense = try CalibrationModel(source: try SourceCheckpoint(directory: sourceURL))

    let prompt = reference.tokenizer.encode("The capital of France is")
    let ids = MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])

    var lines: [String] = []
    func note(_ label: String, _ a: MLXArray, _ b: MLXArray) {
      let x = a.asType(.float32)
      let y = b.asType(.float32)
      let d = x - y
      let error =
        sqrt((d * d).sum()).item(Float.self) / max(sqrt((y * y).sum()).item(Float.self), 1e-9)
      lines.append(
        String(
          format: "%-12@ rel %.4f   dense absmax %8.3f   ref absmax %8.3f", label as NSString,
          error, abs(x).max().item(Float.self), abs(y).max().item(Float.self)))
    }

    let ed = dense.text.embedTokens(ids)
    var hr = reference.text.embedTokens(ids)
    eval(ed, hr)
    note("embed", ed, hr)

    // Each layer is fed the reference's own input, so what it reports is that one layer's
    // disagreement rather than everything upstream of it compounded.
    let mask = causalMask(length: hr.dim(1), offset: 0, dtype: hr.dtype)
    for index in 0..<reference.text.layers.count {
      let hd = dense.text.layers[index](hr, mask: mask, cache: nil, positions: nil)
      let next = reference.text.layers[index](hr, mask: mask, cache: nil, positions: nil)
      eval(hd, next)
      let kind = reference.text.layers[index].isLinear ? "linear" : "attn"
      note("L\(index) \(kind)", hd, next)
      hr = next
    }

    FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
  }
}
