// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

// Probe: every tensor a layer reads, as the source hands it over versus as a known-good pack
// stores it, so a naming or layout mistake shows up as one bad row rather than bad logits.
@Suite(
  "TensorDiffProbe",
  .enabled(
    if: ProcessInfo.processInfo.environment["ISHIZUKI_DENSE"] != nil
      && ProcessInfo.processInfo.environment["ISHIZUKI_REF"] != nil))
struct TensorDiffProbe {
  @Test("probe")
  func probe() throws {
    let env = ProcessInfo.processInfo.environment
    let refURL = URL(filePath: env["ISHIZUKI_REF"]!)
    let source = try SourceCheckpoint(directory: URL(filePath: env["ISHIZUKI_DENSE"]!))
    let pack = try WeightStore(directory: refURL)
    let config = try BonsaiConfig.load(directory: refURL)

    let names = source.tensorNames.sorted()
    let canonical = TensorNaming.map(names)
    let upstream = TensorNaming.isHuggingFaceLayout(names)

    var lines: [String] = []
    for name in names where name.contains(".layers.0.") || name.contains("embed_tokens") {
      guard !name.hasPrefix("mtp.") else { continue }
      let target = canonical[name] ?? name
      var mine = try source.tensor(name)
      if upstream {
        mine = TensorNaming.relayout(
          name, mine, zeroCentredNorms: TensorNaming.usesZeroCentredNorms(source.config))
      }
      mine = mine.asType(.float32)

      let base = target.hasSuffix(".weight") ? String(target.dropLast(7)) : target
      let theirs: MLXArray
      if let scales = pack.optional(base + ".scales"),
        let biases = pack.optional(base + ".biases"),
        let wq = pack.optional(base + ".weight")
      {
        let entry = config.quantization.module(base)
        theirs = dequantized(
          wq, scales: scales, biases: biases, groupSize: entry.groupSize, bits: entry.bits,
          mode: .affine
        ).asType(.float32)
      } else if let plain = pack.optional(target) {
        theirs = plain.asType(.float32)
      } else {
        lines.append("\(target)  MISSING FROM PACK")
        continue
      }

      guard mine.shape == theirs.shape else {
        lines.append("\(target)  SHAPE \(mine.shape) vs \(theirs.shape)")
        continue
      }
      let d = mine - theirs
      let error =
        sqrt((d * d).sum()).item(Float.self) / max(sqrt((theirs * theirs).sum()).item(Float.self), 1e-9)
      lines.append(String(format: "%-72@ rel %.4f", target as NSString, error))
    }
    FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
  }
}
