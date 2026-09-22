// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Building a pack from a full-precision checkpoint: what is there to quantize, and how small.

import Foundation
import IshizukiKit
import Observation

@MainActor
@Observable
final class QuantizeController {
  private(set) var candidates: [FullPrecisionScan.Candidate] = []
  var sourceName: String = ""
  var profileName: String = QuantProfile.balanced.name
  var calibrate = false
  var replace = false
  var streamExperts = false

  private(set) var outcome: Quantizer.Outcome?

  var source: FullPrecisionScan.Candidate? {
    candidates.first { $0.name == sourceName } ?? candidates.first
  }

  var profile: QuantProfile {
    QuantProfile.named(profileName) ?? .balanced
  }

  func rescan(roots: [URL]) {
    candidates = FullPrecisionScan.run(in: roots)
    if candidates.first(where: { $0.name == sourceName }) == nil {
      sourceName = candidates.first?.name ?? ""
    }
  }

  func plan() -> QuantizePlan? {
    guard let source else { return nil }
    return try? QuantizePlan(source: source, profile: profile, within: IshizukiPaths.models)
  }

  func start(on runner: JobRunner) {
    guard let plan = plan() else { return }
    outcome = nil
    let calibrate = calibrate
    let replace = replace
    let streamExperts = streamExperts && plan.expertCount > 0

    runner.run("quantize \(plan.destination.lastPathComponent)") { [weak self] log in
      log.line("source    \(plan.source.name)")
      log.line("layers    \(plan.layerCount)")
      log.line(
        String(
          format: "profile   %@ — %d-bit base, lifts to %@, target ~%.1f bpw",
          plan.profile.name, plan.profile.baseBits,
          plan.profile.boostBits.map(String.init).joined(separator: "/"),
          plan.profile.targetBpw))
      log.line(
        "estimate  \(ReadoutFormat.bytes(plan.estimateBytes))"
          + "  from \(ReadoutFormat.bytes(plan.sourceBytes))")
      if streamExperts {
        log.line(
          "experts   \(plan.expertCount) per sparse layer"
            + "  written beside the pack, read a few at a time")
      }
      if plan.engramBytes > 0 {
        log.line(
          "n-grams   \(ReadoutFormat.bytes(plan.engramBytes))"
            + "  carried whole, streamed from disk rather than held")
      }
      log.line("output    \(plan.destination.path)")
      log.line("")

      let lastPhase = Mutex("")
      let outcome = try plan.run(
        calibrate: calibrate, streamExperts: streamExperts, replacing: replace
      ) { step in
        log.progress(step.fraction)
        if lastPhase.swap(step.phase.rawValue) != step.phase.rawValue {
          log.line("\(step.phase.rawValue)  \(step.detail)")
        }
      }
      try log.checkCancellation()

      log.line("")
      log.line(
        "size      \(ReadoutFormat.bytes(outcome.byteCount))"
          + "  \(outcome.shards) shard\(outcome.shards == 1 ? "" : "s")")
      log.line(String(format: "bpw       %.2f measured", outcome.achievedBpw))
      log.line(
        "widths    "
          + outcome.histogram.keys.sorted()
          .map { "\(outcome.histogram[$0] ?? 0)×\($0)-bit" }
          .joined(separator: " · "))
      log.line("took      \(ReadoutFormat.duration(outcome.seconds))")

      Task { @MainActor in self?.outcome = outcome }
    }
  }
}
