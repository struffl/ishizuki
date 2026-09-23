// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

// The INT8 slices a pack carries, keyed the way the exporter names them: "<layer>.mlp.gate_proj".
// Slices load on first use and are held afterwards, and a layer with no slice simply stays on
// Metal, so a partial export is usable. Past about 4.5 GB resident the Neural Engine stops
// holding them: each further slice takes seconds to load and runs ten times slower, so the bank
// stops at its budget and leaves the later layers on Metal.
public final class ANEBank: @unchecked Sendable {
  public let directory: URL
  public let rows: Int

  // One Neural Engine, so one queue: work handed over from different layers queues rather than
  // contending, and the Metal side keeps running either way.
  private let queue = DispatchQueue(label: "studio.ishizuki.ane", qos: .userInitiated)
  private let lock = NSLock()
  private var loaded: [String: ANESlice] = [:]
  private var missing: Set<String> = []
  private var loadedBytes = 0

  public let fraction: Double?
  public let budgetBytes: Int

  public init?(pack: URL, budgetBytes: Int = BonsaiRuntime.aneBudgetBytes) {
    let contents =
      (try? FileManager.default.contentsOfDirectory(atPath: pack.path)) ?? []
    let exports = contents.compactMap { name -> (Int, String)? in
      guard name.hasPrefix("ane-"), let rows = Int(name.dropFirst(4)) else { return nil }
      return (rows, name)
    }
    guard let chosen = exports.max(by: { $0.0 < $1.0 }) else { return nil }

    self.directory = pack.appending(path: chosen.1)
    self.rows = chosen.0
    self.budgetBytes = budgetBytes

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

  public var residentBytes: Int {
    lock.lock()
    defer { lock.unlock() }
    return loadedBytes
  }

  public func slice(_ name: String) -> ANESlice? {
    slices([name])?.first
  }

  public func slices(_ names: [String]) -> [ANESlice]? {
    lock.lock()
    defer { lock.unlock() }
    if names.contains(where: missing.contains) { return nil }

    let pending = names.filter { loaded[$0] == nil }
    var sizes: [String: Int] = [:]
    for name in pending {
      guard let size = Self.size(of: url(name)) else {
        missing.insert(name)
        return nil
      }
      sizes[name] = size
    }
    guard loadedBytes + sizes.values.reduce(0, +) <= budgetBytes else { return nil }

    for name in pending {
      guard let slice = try? ANESlice(url: url(name), queue: queue) else {
        missing.insert(name)
        return nil
      }
      loaded[name] = slice
      loadedBytes += sizes[name]!
    }
    return names.map { loaded[$0]! }
  }

  private func url(_ name: String) -> URL {
    directory.appending(path: name + ".mlmodelc")
  }

  private static func size(of url: URL) -> Int? {
    guard
      let walker = FileManager.default.enumerator(
        at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey])
    else { return nil }
    var total = 0
    for case let file as URL in walker {
      let values = try? file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
      if values?.isRegularFile == true { total += values?.fileSize ?? 0 }
    }
    return total > 0 ? total : nil
  }
}
