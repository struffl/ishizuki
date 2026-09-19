// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import IshizukiKit

struct Quantize: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "quantize",
    abstract: "Build a mixed-width pack from a full-precision checkpoint.")

  @Option(name: .long, help: "Full-precision checkpoint to quantize. Omit to pick from a list.")
  var source: String?

  @Option(
    name: .long,
    help:
      "Where to write the pack. Omit to name it after the source and the profile, beside the models ishizuki pulls."
  )
  var output: String?

  @Option(name: .long, help: "tiny, small, balanced or quality. Omit to choose from a list.")
  var profile: String?

  @Option(name: .long, help: "Quantization group size.")
  var groupSize: Int = 64

  @Option(name: .long, help: "Shard size in GB.")
  var shardGB: Double = 4

  @Flag(name: .long, help: "Replace an existing output directory.")
  var force = false

  @Flag(
    name: .long,
    help:
      "Measure real activations first and quantize against them, instead of blind to them. Slower — a real forward pass over the whole model — but closes most of the gap to a calibrated pack like oQ4e."
  )
  var calibrate = false

  @Flag(name: .long, help: "Print what would be built and stop.")
  var dryRun = false

  func run() throws {
    let picked = try resolveSource()
    let sourceURL = picked.directory
    // A HuggingFace checkout is a revision hash on disk; the repo name is what to call it, and
    // without its owner, so a pack sits in the models directory under the same kind of name as
    // the ones `pull` puts there.
    let sourceName = (picked.name as NSString).lastPathComponent
    let chosen = try resolveProfile()
    let profile =
      groupSize == chosen.groupSize
      ? chosen
      : QuantProfile(
        name: chosen.name, baseBits: chosen.baseBits, boostBits: chosen.boostBits,
        targetBpw: chosen.targetBpw, groupSize: groupSize, summary: chosen.summary)

    let destination =
      output.map { URL(filePath: ($0 as NSString).expandingTildeInPath) }
      ?? modelsDirectory.appending(path: "\(sourceName)-ishizuki-\(profile.name)")

    let fm = FileManager.default
    if fm.fileExists(atPath: destination.path) {
      guard force else {
        throw ValidationError("\(destination.path) already exists; pass --force to replace it")
      }
      if !dryRun { try fm.removeItem(at: destination) }
    }

    let checkpoint = try SourceCheckpoint(directory: sourceURL)
    let sourceBytes = MemoryBudget.weightBytes(in: sourceURL) ?? 0
    // Scales and biases are already inside the target, so this is a fair estimate rather than
    // the nominal width.
    let estimate = Double(sourceBytes) * profile.targetBpw / 16.0

    if dryRun {
      print(Style.banner("quantize"))
      print("")
      print("  " + Style.field("source", Style.accent(picked.name)))
      print("  " + Style.field("layers", Style.accent("\(checkpoint.layerCount)")))
      print(
        "  "
          + Style.field(
            "profile",
            Style.accent(profile.name)
              + Style.faint(
                String(
                  format: "  %d-bit base, lifts to %@, target ~%.1f bpw", profile.baseBits,
                  profile.boostBits.map(String.init).joined(separator: "/"),
                  profile.targetBpw))))
      print(
        "  "
          + Style.field(
            "estimate",
            Style.accent(Format.bytes(Int(estimate)))
              + Style.faint("  from \(Format.bytes(sourceBytes))")))
      print("  " + Style.field("output", Style.accent(destination.path)))
      return
    }

    let screen = QuantizeScreen(
      source: picked.name,
      destination: destination.lastPathComponent,
      profile: profile,
      sourceBytes: sourceBytes,
      estimateBytes: Int(estimate))

    screen.begin()
    let quantizer = Quantizer(
      source: checkpoint, profile: profile, destination: destination,
      shardLimit: Int(shardGB * 1_073_741_824), calibrate: calibrate
    ) { progress in
      screen.apply(progress)
    }

    do {
      let outcome = try quantizer.run()
      screen.end()
      report(outcome, destination: destination)
    } catch {
      screen.fail("\(error)")
      screen.end()
      throw error
    }
  }

  private func report(_ outcome: Quantizer.Outcome, destination: URL) {
    print("")
    print(Style.banner("built \(destination.lastPathComponent)"))
    print("")
    print(
      "  "
        + Style.field(
          "size",
          Style.accent(Format.bytes(outcome.byteCount))
            + Style.faint("  \(outcome.shards) shard\(outcome.shards == 1 ? "" : "s")")))
    print(
      "  "
        + Style.field(
          "bpw",
          Style.accent(String(format: "%.2f", outcome.achievedBpw)) + Style.faint(" measured")
        ))
    let widths = outcome.histogram.keys.sorted()
      .map { "\(outcome.histogram[$0] ?? 0)×\($0)-bit" }
      .joined(separator: " · ")
    print("  " + Style.field("widths", Style.faint(widths)))
    print("  " + Style.field("took", Style.accent(QuantizeScreen.duration(outcome.seconds))))
    print("")
    print(Style.faint("  ishizuki generate --model \(destination.path) --offline"))
  }

  // MARK: - Choosing

  private func resolveSource() throws -> FullPrecisionScan.Candidate {
    if let source {
      let url = URL(filePath: (source as NSString).expandingTildeInPath)
      return FullPrecisionScan.describe(url)
    }
    let candidates = FullPrecisionScan.run(in: modelSearchRoots)
    guard !candidates.isEmpty else {
      throw ValidationError("no full-precision checkpoints found; pass --source with a path")
    }
    if candidates.count == 1 { return candidates[0] }
    guard Picker.isInteractive else { return candidates[0] }

    let rows = candidates.map {
      Picker.Row(title: $0.name, detail: "\(Format.bytes($0.byteCount)) · \($0.dtype)")
    }
    switch Picker.run(title: "which checkpoint should be quantized?", rows: rows) {
    case .chose(let index): return candidates[index]
    case .delete, .cancelled: throw ExitCode.failure
    }
  }

  private func resolveProfile() throws -> QuantProfile {
    if let profile {
      guard let found = QuantProfile.named(profile) else {
        throw ValidationError(
          "unknown profile '\(profile)'; try "
            + QuantProfile.all.map(\.name).joined(separator: ", "))
      }
      return found
    }
    guard Picker.isInteractive else { return .balanced }
    let rows = QuantProfile.all.map {
      Picker.Row(
        title: $0.name,
        detail: String(format: "~%.1f bpw  ·  %@", $0.targetBpw, $0.summary))
    }
    let initial = QuantProfile.all.firstIndex { $0.name == "balanced" } ?? 0
    switch Picker.run(title: "how small?", rows: rows, initial: initial) {
    case .chose(let index): return QuantProfile.all[index]
    case .delete, .cancelled: throw ExitCode.failure
    }
  }
}
