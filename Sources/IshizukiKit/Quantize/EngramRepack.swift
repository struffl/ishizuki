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

  /// Whether a checkpoint carries a table at all. A text-only graft keeps it at the top of the
  /// model and upstream nests it in the one layer that reads it, so it is found by its tail.
  public static func shards(in source: SourceCheckpoint) -> [String] {
    source.tensorNames
      .filter { $0.contains("ngram_embedding.shard_") && $0.hasSuffix(".weight") }
      .sorted { order($0) < order($1) }
  }

  /// The table and the buffers that address it, which never belong in a pack's shards.
  public static func isTable(_ name: String) -> Bool {
    name.contains("ngram_embedding.") || name.contains("ple_embedding.")
  }

  private static func order(_ name: String) -> Int {
    guard let range = name.range(of: "ngram_embedding.shard_") else { return 0 }
    return Int(name[range.upperBound...].prefix { $0.isNumber }) ?? 0
  }

  public static func run(
    source: SourceCheckpoint, destination: URL, rowsPerPart: Int = rowsPerPart,
    bits: Int? = nil, groupSize: Int = 32, log: @escaping (String) -> Void = { _ in }
  ) throws -> Plan? {
    let shardNames = shards(in: source)
    guard !shardNames.isEmpty else { return nil }

    let names = source.tensorNames
    func buffer(_ name: String) throws -> [Int] {
      guard let full = names.first(where: { $0.hasSuffix("ple_embedding." + name) }) else {
        throw BonsaiError.missingWeight("ple_embedding.\(name)")
      }
      return try source.tensor(full).asArray(Int64.self).map(Int.init)
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
    // A row narrower than a group has nothing to quantize over, so it is written as it is.
    var bits = bits
    if bits != nil, headDim % groupSize != 0 {
      log("engrams: rows of \(headDim) do not divide groups of \(groupSize); writing fp16")
      bits = nil
    }
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
      dtype: "float16", multipliers: multipliers, eosTokenId: eos,
      bits: bits, groupSize: bits == nil ? nil : groupSize)
    let rowBytes = try layout.rowBytes

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
        let slice = shard[row..<(row + take), 0...]
        try handle?.write(contentsOf: try encode(slice, bits: bits, groupSize: groupSize))
        row += take
        written += take
      }
    }

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(layout).write(
      to: destination.appending(path: EngramLayout.layoutFile))

    return Plan(layout: layout, byteCount: written * rowBytes)
  }

  /// A block of rows as the store reads them: fp16 as they are, or each row's 8-bit codes
  /// followed by its groups' fp16 scales and then their biases.
  static func encode(_ rows: MLXArray, bits: Int?, groupSize: Int) throws -> Data {
    guard let bits else {
      let wide = rows.asType(.float16)
      eval(wide)
      return wide.asData().data
    }
    guard bits == 8 else {
      throw BonsaiError.unsupportedModel("an n-gram table is written at 8 bits or at 16")
    }
    let (wq, scales, biases) = quantized(
      rows.asType(.float32), groupSize: groupSize, bits: 8, mode: .affine)
    let s16 = scales.asType(.float16)
    let b16 = (biases ?? MLXArray.zeros(like: scales)).asType(.float16)
    eval(wq, s16, b16)
    let count = rows.dim(0)
    let width = rows.dim(1)
    let affine = width / groupSize * 2
    let codes = wq.asData().data
    let scaleBytes = s16.asData().data
    let biasBytes = b16.asData().data
    var out = Data(count: count * (width + 2 * affine))
    out.withUnsafeMutableBytes { target in
      codes.withUnsafeBytes { c in
        scaleBytes.withUnsafeBytes { sc in
          biasBytes.withUnsafeBytes { bi in
            for row in 0..<count {
              let at = row * (width + 2 * affine)
              target.baseAddress!.advanced(by: at)
                .copyMemory(from: c.baseAddress!.advanced(by: row * width), byteCount: width)
              target.baseAddress!.advanced(by: at + width)
                .copyMemory(from: sc.baseAddress!.advanced(by: row * affine), byteCount: affine)
              target.baseAddress!.advanced(by: at + width + affine)
                .copyMemory(from: bi.baseAddress!.advanced(by: row * affine), byteCount: affine)
            }
          }
        }
      }
    }
    return out
  }
}
