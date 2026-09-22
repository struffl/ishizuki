// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Moving a sparse pack's routed experts onto disk, so a machine can run a model it cannot hold.

import Foundation
import IshizukiKit
import Observation

@MainActor
@Observable
final class ExpertSplitController {
  struct Candidate: Identifiable, Equatable {
    var id: String
    var url: URL
    var byteCount: Int
    var layers: Int
    var expertCount: Int
    var residentBytes: Int
    var expertBytes: Int

    var destination: URL {
      IshizukiPaths.models.appending(path: id + "-streamed")
    }

    var destinationExists: Bool {
      FileManager.default.fileExists(atPath: destination.path)
    }
  }

  private(set) var candidates: [Candidate] = []
  var sourceID = ""
  var replace = false

  private(set) var outcome: ExpertRepack.Plan?

  var source: Candidate? {
    candidates.first { $0.id == sourceID } ?? candidates.first
  }

  /// A pack qualifies by routing through experts it still holds in its shards. Reading that
  /// costs a config and a safetensors header, so the whole catalog can be asked.
  func rescan(catalog: ModelCatalog) {
    candidates = catalog.entries.compactMap { entry in
      guard entry.format == .pack, !ExpertRepack.isSplit(entry.url) else { return nil }
      guard let preview = try? ExpertRepack.preview(source: entry.url) else { return nil }
      return Candidate(
        id: entry.id, url: entry.url, byteCount: entry.byteCount,
        layers: preview.layers.count, expertCount: preview.expertCount,
        residentBytes: preview.residentBytes, expertBytes: preview.expertBytes)
    }
    if candidates.first(where: { $0.id == sourceID }) == nil {
      sourceID = candidates.first?.id ?? ""
    }
  }

  func start(on runner: JobRunner, then rescan: @escaping @MainActor @Sendable () -> Void) {
    guard let source else { return }
    outcome = nil
    let destination = source.destination
    let replace = self.replace
    let url = source.url

    runner.run("split \(source.id)") { [weak self] log in
      let fm = FileManager.default
      if fm.fileExists(atPath: destination.path) {
        guard replace else {
          throw BonsaiError.unsupportedModel(
            "\(destination.lastPathComponent) already exists — tick replace to overwrite it")
        }
        try fm.removeItem(at: destination)
      }
      log.line("source    \(url.lastPathComponent)")
      log.line("output    \(destination.path)")
      log.line("")

      let plan: ExpertRepack.Plan
      do {
        plan = try ExpertRepack.run(
          source: url, destination: destination,
          log: { log.line($0) },
          progress: { step in
            try log.checkCancellation()
            log.progress(step.fraction)
          })
      } catch {
        // Half a pack loads as a pack, and the catalog would offer it. Nothing survives a
        // split that did not finish.
        try? fm.removeItem(at: destination)
        throw error
      }

      log.line("")
      log.line("layers    \(plan.layers.count) sparse, \(plan.expertCount) experts each")
      log.line("resident  \(ReadoutFormat.bytes(plan.residentBytes))")
      log.line("streamed  \(ReadoutFormat.bytes(plan.expertBytes))  read from disk, not held")

      Task { @MainActor in
        self?.outcome = plan
        rescan()
      }
    }
  }
}
