// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import IshizukiKit
import MLX

struct VerifyLogits: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "verify-logits",
    abstract: "Compare a full forward pass against Python golden logits.")

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .long) var golden: String = "Tests/Golden/logits_golden.safetensors"

  func run() throws {
    let goldenArrays = try loadArrays(url: URL(filePath: golden))
    let inputIds = goldenArrays["input_ids"]!

    let start = Date()
    let bonsai = try BonsaiModel(directory: URL(filePath: model), loadVision: false)
    print(String(format: "loaded in %.1fs", -start.timeIntervalSinceNow))

    let cache = bonsai.text.makeCache()
    let logits = bonsai.text(inputIds, cache: cache)
    eval(logits)

    report("prefill", logits, goldenArrays["prefill_logits"]!)

    let decodeToken = goldenArrays["decode_token"]!
    let decodeLogits = bonsai.text(decodeToken, cache: cache)
    eval(decodeLogits)
    report("decode", decodeLogits, goldenArrays["decode_logits"]!)
  }

  private func report(_ label: String, _ got: MLXArray, _ want: MLXArray) {
    let a = got.asType(.float32)
    let b = want.asType(.float32)
    let maxAbs = abs(a - b).max().item(Float.self)

    let lastGot = a[0, -1]
    let lastWant = b[0, -1]
    let gotArgmax = lastGot.argMax().item(Int.self)
    let wantArgmax = lastWant.argMax().item(Int.self)

    let gm = lastGot.mean()
    let wm = lastWant.mean()
    let cov = ((lastGot - gm) * (lastWant - wm)).mean().item(Float.self)
    let sd = (lastGot.variance().sqrt() * lastWant.variance().sqrt()).item(Float.self)

    print("\(label):")
    print(String(format: "  max|Δ| over all logits : %.4f", maxAbs))
    print(String(format: "  correlation (last pos) : %.6f", cov / max(sd, 1e-9)))
    print(
      "  argmax got=\(gotArgmax) want=\(wantArgmax) \(gotArgmax == wantArgmax ? "MATCH" : "MISMATCH")"
    )
  }
}
