// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Whether a checkpoint is all here, which a download in progress looks exactly like.

import Foundation
import Testing

@testable import IshizukiKit

@Suite("Checkpoint readiness")
struct FullPrecisionScanTests {
  private func scratch(_ files: [String], index: [String]?) throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory())
      .appending(path: "scan-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    for name in files {
      try Data().write(to: url.appending(path: name))
    }
    if let index {
      let map = Dictionary(uniqueKeysWithValues: index.enumerated().map { ("t\($0.0)", $0.1) })
      try JSONSerialization.data(withJSONObject: ["weight_map": map])
        .write(to: url.appending(path: "model.safetensors.index.json"))
    }
    return url
  }

  @Test("one file needs no index")
  func single() throws {
    let url = try scratch(["model.safetensors"], index: nil)
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(FullPrecisionScan.readiness(url, present: ["model.safetensors"]) == .ready)
  }

  /// The state a fresh `hf download` sits in for most of its run: shards arriving, and the
  /// index that names them not here yet. Nothing can be opened, so it is not a lesser problem
  /// than a gap in the shards — it is the one that reads as "fine" if you only count files.
  @Test("shards with no index are not ready")
  func indexMissing() throws {
    let files = ["model-00001.safetensors", "model-00002.safetensors"]
    let url = try scratch(files, index: nil)
    defer { try? FileManager.default.removeItem(at: url) }
    #expect(FullPrecisionScan.readiness(url, present: files) == .indexMissing(have: 2))
  }

  @Test("counts the shards the index asks for against the ones here")
  func shardsMissing() throws {
    let wanted = ["model-00001.safetensors", "model-00002.safetensors", "model-00003.safetensors"]
    let here = Array(wanted.prefix(2))
    let url = try scratch(here, index: wanted)
    defer { try? FileManager.default.removeItem(at: url) }
    let names = here + ["model.safetensors.index.json"]
    #expect(
      FullPrecisionScan.readiness(url, present: names) == .shardsMissing(have: 2, want: 3))

    let whole = try scratch(wanted, index: wanted)
    defer { try? FileManager.default.removeItem(at: whole) }
    #expect(
      FullPrecisionScan.readiness(
        whole, present: wanted + ["model.safetensors.index.json"]) == .ready)
  }
}
