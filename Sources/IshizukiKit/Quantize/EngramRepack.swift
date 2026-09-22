// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Lifting an n-gram table out of a checkpoint and into files a step can read a row at a time.

import Foundation
import MLX

/// Writes the n-gram table of a per-layer-embedding model beside the pack rather than into it.
///
/// The table is most of such a model's parameters and the least of its work: a token touches
/// one row per head and nothing else, and the rows it will touch are known from the tokens
/// alone. Keeping it in the safetensors would make every load pay for all of it; keeping it in
/// flat files lets a step fetch its four kilobytes and leave the rest on disk.
public enum EngramRepack {
  /// Rows per file. Large enough that a table is a handful of files, small enough that none of
  /// them is unwieldy to copy.
  public static let rowsPerPart = 1 << 21

  public struct Plan: Sendable {
    public var layout: EngramLayout
    public var byteCount: Int
  }

  /// Whether a checkpoint carries a table at all.
  public static func shards(in source: SourceCheckpoint) -> [String] {
    source.tensorNames
      .filter { $0.hasPrefix("model.ngram_embedding.shard_") && $0.hasSuffix(".weight") }
      .sorted { order($0) < order($1) }
  }

  private static func order(_ name: String) -> Int {
    let digits = name.dropFirst("model.ngram_embedding.shard_".count).prefix { $0.isNumber }
    return Int(digits) ?? 0
  }

  public static func run(
    source: SourceCheckpoint, destination: URL, log: @escaping (String) -> Void = { _ in }
  ) throws -> Plan? {
    let shardNames = shards(in: source)
    guard !shardNames.isEmpty else { return nil }

    func buffer(_ name: String) throws -> [Int] {
      try source.tensor("model.ple_embedding." + name).asArray(Int64.self).map(Int.init)
    }
    let vocabSizes = try buffer("ngram_heads_vocab_sizes")
    let offsets = try buffer("ngram_heads_offsets")
    let multipliers = try buffer("layer_multipliers")

    let text = source.textConfig
    func int(_ key: String) -> Int? { (text[key] as? NSNumber)?.intValue }
    let ngramSize = int("ngram_size") ?? 3
    let eos =
      (text["eos_token_id"] as? NSNumber)?.intValue
      ?? ((text["eos_token_id"] as? [NSNumber])?.first?.intValue ?? 0)

    let headDim = try source.tensor(shardNames[0]).dim(1)
    var total = 0
    for name in shardNames { total += try source.tensor(name).dim(0) }
    // The table is addressed by head, so its last rows may be padding the checkpoint added to
    // round the vocabulary up. Nothing ever addresses them.
    let addressable = (offsets.last ?? 0) + (vocabSizes.last ?? 0)
    guard addressable <= total else {
      throw BonsaiError.shapeMismatch(
        "the heads address \(addressable) rows but the table holds \(total)")
    }
    guard vocabSizes.count == offsets.count, multipliers.count == ngramSize else {
      throw BonsaiError.shapeMismatch("the n-gram buffers disagree with the config")
    }

    let layout = EngramLayout(
      ngramSize: ngramSize, heads: vocabSizes.count, headDim: headDim,
      vocabSizes: vocabSizes, offsets: offsets,
      parts: (addressable + rowsPerPart - 1) / rowsPerPart, rowsPerPart: rowsPerPart,
      dtype: "float16", multipliers: multipliers, eosTokenId: eos)

    let fm = FileManager.default
    let folder = destination.appending(path: "engrams")
    try fm.createDirectory(at: folder, withIntermediateDirectories: true)

    // A row block at a time, so a table far larger than memory converts on a machine that
    // could not hold it. Each block lands in exactly one part: the part boundary is a
    // multiple of the block.
    let block = 1 << 16
    var written = 0
    var handle: FileHandle?
    var openPart = -1
    defer { try? handle?.close() }

    for name in shardNames {
      let shard = try source.tensor(name)
      guard shard.dim(1) == headDim else {
        throw BonsaiError.shapeMismatch("\(name) is \(shard.dim(1)) wide, not \(headDim)")
      }
      var row = 0
      while row < shard.dim(0), written < addressable {
        let part = written / rowsPerPart
        let take = min(
          block, shard.dim(0) - row, addressable - written,
          (part + 1) * rowsPerPart - written)
        if part != openPart {
          try handle?.close()
          let url = destination.appending(path: EngramLayout.fileName(part: part))
          fm.createFile(atPath: url.path, contents: nil)
          handle = try FileHandle(forWritingTo: url)
          openPart = part
          log("engrams: part \(part) of \(layout.parts)")
        }
        let slice = shard[row..<(row + take), 0...].asType(.float16)
        eval(slice)
        try handle?.write(contentsOf: slice.asData().data)
        row += take
        written += take
      }
    }

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(layout).write(
      to: destination.appending(path: EngramLayout.layoutFile))

    return Plan(layout: layout, byteCount: written * headDim * 2)
  }
}
