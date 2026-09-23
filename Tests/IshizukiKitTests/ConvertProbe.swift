// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

// Probe, not a test: converts a checkpoint headlessly, for sources too large to hand the app.
@Suite(
  "ConvertProbe",
  .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_CONVERT_SOURCE"] != nil))
struct ConvertProbe {
  @Test("convert")
  func convert() throws {
    let env = ProcessInfo.processInfo.environment
    let source = try SourceCheckpoint(directory: URL(filePath: env["ISHIZUKI_CONVERT_SOURCE"]!))
    let destination = URL(filePath: env["ISHIZUKI_CONVERT_DESTINATION"]!)
    let base = Int(env["ISHIZUKI_CONVERT_BITS"] ?? "") ?? 4
    let profile = QuantProfile(
      name: "stream\(base)", baseBits: base, boostBits: [base + 1, base + 2],
      targetBpw: Double(env["ISHIZUKI_CONVERT_BPW"] ?? "") ?? Double(base) + 0.75,
      groupSize: Int(env["ISHIZUKI_CONVERT_GROUP"] ?? "") ?? 64, summary: "headless conversion")
    let engramBits = env["ISHIZUKI_CONVERT_ENGRAM_BITS"].flatMap(Int.init)
    let stream = env["ISHIZUKI_CONVERT_STREAM"] != "0"
    Memory.cacheLimit = (Int(env["ISHIZUKI_CONVERT_CACHE_GB"] ?? "") ?? 4) << 30

    let started = Date()
    let lastPrint = LockedBox(Date.distantPast)
    let quantizer = Quantizer(
      source: source, profile: profile, destination: destination, streamExperts: stream,
      engramBits: engramBits
    ) { progress in
      let now = Date()
      guard lastPrint.swap(ifOlderThan: 20, now: now) || progress.done == progress.total else {
        return
      }
      print(
        String(
          format: "[%6.0fs] %@ %d/%d %@ %@", now.timeIntervalSince(started),
          progress.phase.rawValue, progress.done, progress.total, progress.detail, progress.note))
    }
    let outcome = try quantizer.run()
    print(
      String(
        format: "done: %.1f GB, %d shards, %.2f bpw, %.0f s", Double(outcome.byteCount) / 1e9,
        outcome.shards, outcome.achievedBpw, outcome.seconds))
    print("histogram: \(outcome.histogram.sorted { $0.key < $1.key })")
  }
}

final class LockedBox: @unchecked Sendable {
  private let lock = NSLock()
  private var value: Date
  init(_ value: Date) { self.value = value }
  func swap(ifOlderThan seconds: Double, now: Date) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    guard now.timeIntervalSince(value) >= seconds else { return false }
    value = now
    return true
  }
}
