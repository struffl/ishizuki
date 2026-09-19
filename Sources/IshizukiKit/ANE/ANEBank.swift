// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

// The INT8 slices a pack carries, keyed the way the exporter names them: "<layer>.mlp.gate_proj".
// Slices load on first use and are held afterwards, and a layer with no slice simply stays on
// Metal, so a partial export is usable.
public final class ANEBank: @unchecked Sendable {
  public let directory: URL
  public let rows: Int

  // One Neural Engine, so one queue: work handed over from different layers queues rather than
  // contending, and the Metal side keeps running either way.
  private let queue = DispatchQueue(label: "studio.ishizuki.ane", qos: .userInitiated)
  private let lock = NSLock()
  private var loaded: [String: ANESlice] = [:]
  private var missing: Set<String> = []

  public let fraction: Double?

  public init?(pack: URL) {
    let contents =
      (try? FileManager.default.contentsOfDirectory(atPath: pack.path)) ?? []
    let exports = contents.compactMap { name -> (Int, String)? in
      guard name.hasPrefix("ane-"), let rows = Int(name.dropFirst(4)) else { return nil }
      return (rows, name)
    }
    guard let chosen = exports.max(by: { $0.0 < $1.0 }) else { return nil }

    self.directory = pack.appending(path: chosen.1)
    self.rows = chosen.0

    let manifest = directory.appending(path: "manifest.json")
    if let data = try? Data(contentsOf: manifest),
      let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    {
      self.fraction = parsed["fraction"] as? Double
    } else {
      self.fraction = nil
    }
  }

  public var count: Int {
    (try? FileManager.default.contentsOfDirectory(atPath: directory.path))?
      .filter { $0.hasSuffix(".mlmodelc") }.count ?? 0
  }

  public func slice(_ name: String) -> ANESlice? {
    lock.lock()
    defer { lock.unlock() }
    if let cached = loaded[name] { return cached }
    if missing.contains(name) { return nil }

    let url = directory.appending(path: name + ".mlmodelc")
    guard FileManager.default.fileExists(atPath: url.path),
      let slice = try? ANESlice(url: url, queue: queue)
    else {
      missing.insert(name)
      return nil
    }
    loaded[name] = slice
    return slice
  }
}
