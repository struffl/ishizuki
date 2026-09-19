// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Accumulates tensors and writes them out as shards, flushing whenever the one in hand grows
/// past the limit.
///
/// Flushing is what keeps the quantizer's memory flat: a tensor is evaluated, written, and
/// dropped, so the process holds a shard's worth at most rather than the whole output.
public struct PackWriter {
  public struct Summary: Sendable {
    public let shards: Int
    public let byteCount: Int
  }

  public let directory: URL
  public let shardLimit: Int

  private var pending: [String: MLXArray] = [:]
  private var pendingBytes = 0
  private var shards: [[String]] = []
  private var written: [(name: String, shard: Int)] = []

  public init(directory: URL, shardLimit: Int = 4 << 30) {
    self.directory = directory
    self.shardLimit = shardLimit
  }

  public mutating func add(_ name: String, _ array: MLXArray) throws {
    eval(array)
    pending[name] = array
    pendingBytes += array.size * array.itemSize
    if pendingBytes >= shardLimit { try flush() }
  }

  private mutating func flush() throws {
    guard !pending.isEmpty else { return }
    let index = shards.count
    let url = directory.appending(path: Self.shardName(index: index, of: nil))
    try MLX.save(arrays: pending, url: url)
    shards.append(pending.keys.sorted())
    for name in pending.keys { written.append((name, index)) }
    pending.removeAll()
    pendingBytes = 0
  }

  /// safetensors shard names carry the total, which is only known once everything is written,
  /// so the files are renamed at the end.
  public mutating func finish() throws -> Summary {
    try flush()
    let fm = FileManager.default
    let total = shards.count

    var byteCount = 0
    var finalName: [Int: String] = [:]
    for index in 0..<total {
      let from = directory.appending(path: Self.shardName(index: index, of: nil))
      let name = total == 1 ? "model.safetensors" : Self.shardName(index: index, of: total)
      let to = directory.appending(path: name)
      if from != to {
        if fm.fileExists(atPath: to.path) { try fm.removeItem(at: to) }
        try fm.moveItem(at: from, to: to)
      }
      finalName[index] = name
      byteCount +=
        ((try? fm.attributesOfItem(atPath: to.path))?[.size] as? NSNumber)?.intValue ?? 0
    }

    if total > 1 {
      var weightMap: [String: String] = [:]
      for entry in written { weightMap[entry.name] = finalName[entry.shard] }
      let index: [String: Any] = [
        "metadata": ["total_size": byteCount],
        "weight_map": weightMap,
      ]
      try JSONSerialization.data(withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
        .write(to: directory.appending(path: "model.safetensors.index.json"))
    }

    return Summary(shards: total, byteCount: byteCount)
  }

  private static func shardName(index: Int, of total: Int?) -> String {
    guard let total else { return String(format: "shard-%05d.safetensors", index) }
    return String(format: "model-%05d-of-%05d.safetensors", index + 1, total)
  }
}
