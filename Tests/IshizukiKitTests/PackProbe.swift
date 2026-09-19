// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import IshizukiKit

// Probe, not a test: prints the top predictions a pack makes for a fixed prompt.
@Suite("PackProbe", .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_PACK"] != nil))
struct PackProbe {
  @Test("probe")
  func probe() throws {
    let path = ProcessInfo.processInfo.environment["ISHIZUKI_PACK"]!
    let model = try BonsaiModel(directory: URL(filePath: path), loadVision: false)
    let prompt = model.tokenizer.encode("The capital of France is")
    print("prompt tokens: \(prompt)")

    let ids = MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])
    let cache = model.text.makeCache(kvConfig: .full)
    let hidden = model.text.trunk(inputs: ids, cache: cache)
    eval(hidden)
    let h = hidden[0, -1]
    print(
      "hidden last: mean \(h.mean().item(Float.self)) absmax \(abs(h).max().item(Float.self))")

    let logits = model.text.lastLogits(model.text.normed(hidden))[0, -1]
    eval(logits)
    print(
      "logits: mean \(logits.mean().item(Float.self)) max \(logits.max().item(Float.self))")

    let order = argSort(logits, axis: -1)
    let topIds = order[(logits.dim(0) - 5)...].asArray(Int32.self).reversed()
    for id in topIds {
      let text = model.tokenizer.decode([Int(id)])
      print("  \(id)  \(logits[Int(id)].item(Float.self))  \(text.debugDescription)")
    }
  }
}
