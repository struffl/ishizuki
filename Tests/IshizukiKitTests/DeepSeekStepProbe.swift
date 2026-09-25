// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Where a DeepSeek-V4.1 decode step's time goes, read against the release itself.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// `ISHIZUKI_DEEPSEEK_STEPS` turns it on, and `ISHIZUKI_EXPERT_SLOTS` sets the slots. A step of
/// fresh tokens reads its experts and n-gram rows; the same token fed again finds both where the
/// last step left them, so the difference is what the reads cost and the rest is compute.
/// Everything MLX holds is wired first, so no step waits on pages the system compressed.
@Suite(
  "DeepSeek-V4.1 step", .serialized,
  .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_DEEPSEEK_STEPS"] != nil))
struct DeepSeekStepProbe {
  @Test("splits a decode step into reads and compute")
  func splitsAStep() async throws {
    let directory = try #require(DeepSeekRelease.directory)
    if let slots = Int(ProcessInfo.processInfo.environment["ISHIZUKI_EXPERT_SLOTS"] ?? "") {
      BonsaiRuntime.expertSlots = slots
    }
    defer { BonsaiRuntime.expertSlots = 0 }
    let model = try #require(try BonsaiModel(directory: directory).deepseek)
    let prompt = MLXArray((0..<22).map { Int32(1000 + 37 * $0) }).reshaped([1, 22])
    let cache = model.makeCache()
    eval(model(prompt, cache: cache))
    let held = Memory.activeMemory + Memory.cacheMemory
    let wired = WiredMemoryTicket(size: held + (4 << 30), policy: WiredSumPolicy())
    _ = await wired.start()

    func step(_ token: Int32) -> Double {
      let start = Date()
      eval(model(MLXArray([token]).reshaped([1, 1]), cache: cache))
      return -start.timeIntervalSinceNow * 1000
    }
    let fresh = (0..<4).map { step(Int32(3000 + 911 * $0)) }
    _ = step(4242)
    let same = (0..<8).map { _ in step(4242) }
    let after = (0..<4).map { step(Int32(9000 + 577 * $0)) }
    for (label, times) in [("fresh tokens", fresh), ("same token", same), ("fresh again", after)] {
      print(label + ": " + times.map { String(format: "%.0f", $0) }.joined(separator: " ") + " ms")
    }
    if let traffic = model.expertTraffic {
      print(String(format: "experts: %d reads, %.1f%% hits", traffic.reads, 100 * traffic.hitRate))
    }
    _ = await wired.end()
  }
}
