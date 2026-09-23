// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

// Probe, not a test: how long a coding-sized prompt takes to prefill when the experts stream.
@Suite(
  "StreamedPrefillProbe",
  .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_PREFILL_PACK"] != nil))
struct StreamedPrefillProbe {
  @Test("prefill")
  func prefill() throws {
    let env = ProcessInfo.processInfo.environment
    if let slots = env["ISHIZUKI_EXPERT_SLOTS"].flatMap(Int.init) {
      BonsaiRuntime.expertSlots = slots
    }
    let residency = ResidencyManager(
      options: .init(wiredBytes: (Int(env["ISHIZUKI_WIRED_GB"] ?? "") ?? 0) << 30))
    residency.wire()
    Memory.cacheLimit = (Int(env["ISHIZUKI_CACHE_GB"] ?? "") ?? 2) << 30
    let model = try BonsaiModel(path: URL(filePath: env["ISHIZUKI_PREFILL_PACK"]!))
    let lengths = (env["ISHIZUKI_PREFILL_LENGTHS"] ?? "1024,2048,4096")
      .split(separator: ",").compactMap { Int($0) }
    let chunk = Int(env["ISHIZUKI_PREFILL_CHUNK"] ?? "") ?? 512

    let sources = URL(filePath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appending(path: "Sources/IshizukiKit")
    var code = ""
    for file in ["Core/ExpertStore.swift", "Quantize/Quantizer.swift", "Text/MoEBlock.swift"] {
      code += "// \(file)\n" + (try String(contentsOf: sources.appending(path: file), encoding: .utf8))
    }
    let all = model.tokenizer.encode(
      code + "\n\nWhy does a streamed expert store read one projection per miss? Suggest a fix.")
    let out = URL(filePath: env["ISHIZUKI_PREFILL_OUT"] ?? NSTemporaryDirectory() + "prefill.txt")
    var report =
      "prompt source \(all.count) tokens, chunk \(chunk), slots \(BonsaiRuntime.expertSlots), "
      + "wired \(residency.options.wiredBytes >> 30) GB\n"

    for length in lengths where length <= all.count {
      let ids = Array(all.suffix(length))
      let cache = model.text.makeCache()
      let before = model.store.expertTraffic
      let start = Date()
      var logits = MLXArray(0)
      var at = 0
      while at < ids.count {
        let piece = Array(ids[at..<min(at + chunk, ids.count)])
        logits = model.text(MLXArray(piece.map(Int32.init)).reshaped([1, piece.count]), cache: cache)
        eval(logits)
        at += piece.count
      }
      let seconds = -start.timeIntervalSinceNow
      let after = model.store.expertTraffic
      let misses = (after?.misses ?? 0) - (before?.misses ?? 0)
      let hits = (after?.hits ?? 0) - (before?.hits ?? 0)
      let first = logits[0..., -1, 0...].argMax(axis: -1).item(Int32.self)
      let decodeStart = Date()
      _ = model.text(MLXArray([first]).reshaped([1, 1]), cache: cache)
      eval(model.text(MLXArray([first]).reshaped([1, 1]), cache: cache))
      let step = -decodeStart.timeIntervalSinceNow / 2
      let line = String(
        format: "%5d tokens: prefill %7.1f s (%6.1f tok/s), expert reads %d misses / %d hits, next step %.2f s\n",
        length, seconds, Double(length) / seconds, misses, hits, step)
      report += line
      try report.write(to: out, atomically: true, encoding: .utf8)
      print(line, terminator: "")
    }
  }
}
