// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

// Probe: runs the unquantized source checkpoint through this runtime's own TextModel, so a
// wrong answer here indicts the architecture rather than the quantizer.
@Suite("DenseProbe", .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_DENSE"] != nil))
struct DenseProbe {
  @Test("probe")
  func probe() throws {
    let path = ProcessInfo.processInfo.environment["ISHIZUKI_DENSE"]!
    let directory = URL(filePath: path)
    let source = try SourceCheckpoint(directory: directory)
    let model = try CalibrationModel(source: source)
    let tokenizer = try BonsaiTokenizer(directory: directory)

    let prompt = tokenizer.encode("The capital of France is")
    let ids = MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])
    let hidden = model.text.trunk(inputs: ids, cache: nil, positions: nil)
    eval(hidden)

    let logits = model.text.lastLogits(model.text.normed(hidden))[0, -1]
    eval(logits)

    var lines = ["prompt tokens: \(prompt)"]
    lines.append("logits: max \(logits.max().item(Float.self))")
    let order = argSort(logits, axis: -1)
    for id in order[(logits.dim(0) - 5)...].asArray(Int32.self).reversed() {
      lines.append(
        "  \(id)  \(logits[Int(id)].item(Float.self))  "
          + tokenizer.decode([Int(id)]).debugDescription)
    }
    FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
  }
}
