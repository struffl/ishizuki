// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DeepSeek's release shards, read without MLX's safetensors loader.

import Cmlx
import Foundation
import MLX

/// A safetensors checkpoint read a tensor at a time, straight from the file.
///
/// MLX's loader refuses a whole shard the moment it meets `F8_E8M0`, and every shard of this
/// release carries those scales. So the header is parsed here and each tensor is read into
/// memory of its own: fp8 codes and scales come back as `uint8`, fp4 pairs as `int8`, and the
/// two 100 GB n-gram tables are read a row range at a time rather than whole.
public final class DeepSeekCheckpoint: @unchecked Sendable {
  public struct Entry: Sendable {
    public var file: URL
    public var dtype: String
    public var shape: [Int]
    public var offset: Int
    public var byteCount: Int

    public var rowBytes: Int { shape.isEmpty ? byteCount : byteCount / max(shape[0], 1) }
  }

  public let directory: URL
  public let config: DeepSeekConfig
  public let entries: [String: Entry]

  /// `partial` reads whatever shards are already there and leaves the rest out, for a probe
  /// run against a download still in progress.
  public init(directory: URL, partial: Bool = false) throws {
    self.directory = directory
    self.config = try DeepSeekConfig.load(directory: directory)

    let fm = FileManager.default
    let index = directory.appending(path: "model.safetensors.index.json")
    var files: [String]
    if fm.fileExists(atPath: index.path) {
      let object = try JSONSerialization.jsonObject(with: try Data(contentsOf: index))
      guard let map = (object as? [String: Any])?["weight_map"] as? [String: String] else {
        throw BonsaiError.missingWeight("model.safetensors.index.json has no weight_map")
      }
      files = Array(Set(map.values)).sorted()
    } else {
      files = ["model.safetensors"]
    }

    var entries: [String: Entry] = [:]
    for name in files {
      let url = directory.appending(path: name)
      guard fm.fileExists(atPath: url.path) else {
        if partial { continue }
        throw BonsaiError.missingWeight("\(name) is not in \(directory.path)")
      }
      for (tensor, entry) in try Self.header(of: url) { entries[tensor] = entry }
    }
    self.entries = entries
  }

  static func header(of url: URL) throws -> [String: Entry] {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
      throw BonsaiError.missingWeight("\(url.lastPathComponent) has no safetensors header")
    }
    let length = prefix.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
    guard let data = try handle.read(upToCount: length),
      let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
      throw BonsaiError.missingWeight("\(url.lastPathComponent) has an unreadable header")
    }
    var entries: [String: Entry] = [:]
    for (name, raw) in object where name != "__metadata__" {
      guard let fields = raw as? [String: Any], let dtype = fields["dtype"] as? String,
        let shape = fields["shape"] as? [NSNumber],
        let offsets = fields["data_offsets"] as? [NSNumber], offsets.count == 2
      else { continue }
      entries[name] = Entry(
        file: url, dtype: dtype, shape: shape.map(\.intValue),
        offset: 8 + length + offsets[0].intValue,
        byteCount: offsets[1].intValue - offsets[0].intValue)
    }
    return entries
  }

  public var names: [String] { entries.keys.sorted() }

  public func has(_ name: String) -> Bool { entries[name] != nil }

  public func entry(_ name: String) throws -> Entry {
    guard let entry = entries[name] else { throw BonsaiError.missingWeight(name) }
    return entry
  }

  /// What a tensor's bytes are once they are in MLX. Both fp8 flavours are carried as their bits.
  public static func dtype(_ name: String) throws -> DType {
    switch name {
    case "F32": .float32
    case "F16": .float16
    case "BF16": .bfloat16
    case "F8_E4M3", "F8_E8M0", "U8": .uint8
    case "I8": .int8
    case "I16": .int16
    case "I32": .int32
    case "I64": .int64
    case "U32": .uint32
    case "BOOL": .bool
    default: throw BonsaiError.unsupportedModel("a safetensors dtype \(name) is not one this reads")
    }
  }

  /// The whole tensor, read now.
  public func tensor(_ name: String) throws -> MLXArray {
    let entry = try entry(name)
    return try read(entry, rows: 0..<(entry.shape.first ?? 1))
  }

  /// A contiguous run of rows along the first axis, read now.
  public func rows(_ name: String, _ range: Range<Int>) throws -> MLXArray {
    try read(try entry(name), rows: range)
  }

  private func read(_ entry: Entry, rows: Range<Int>) throws -> MLXArray {
    let type = try Self.dtype(entry.dtype)
    let stride = entry.rowBytes
    let count = entry.shape.isEmpty ? entry.byteCount : rows.count * stride
    let shape = entry.shape.isEmpty ? [] : [rows.count] + entry.shape.dropFirst()
    guard count > 0 else { return MLXArray.zeros(shape, dtype: type) }
    let buffer = try ResidentBuffer(byteCount: count)
    let descriptor = open(entry.file.path, O_RDONLY)
    guard descriptor >= 0 else {
      throw BonsaiError.missingWeight("cannot open \(entry.file.lastPathComponent)")
    }
    defer { close(descriptor) }
    try buffer.read(
      from: descriptor, offset: entry.offset + rows.lowerBound * stride, into: 0..<count)
    return buffer.array(shape: shape, dtype: type)
  }
}

/// The two ways DeepSeek stores a matrix, and the MLX modes that read them without a change.
public enum DeepSeekFormat {
  /// An fp8 matrix's 32x32 block scales, repeated down each block's rows. MLX's mxfp8 keeps one
  /// power-of-two scale per 32 inputs of each row; DeepSeek shares it across 32 rows, so the
  /// same bytes serve once every row carries its own copy.
  public static func rowScales(_ blocks: MLXArray, rows: Int) -> MLXArray {
    let repeated = repeated(blocks, count: 32, axis: 0)
    return repeated[..<rows, 0...]
  }

  /// fp8 codes `[out, in]` as the words mxfp8 multiplies, four to a word, first in the low byte.
  public static func mxfp8(codes: MLXArray, blockScales: MLXArray) -> (MLXArray, MLXArray) {
    let words = codes.asType(.uint8).view(dtype: .uint32)
    return (words, rowScales(blockScales.asType(.uint8), rows: codes.dim(0)))
  }

  /// fp4 pairs `[out, in / 2]`, low nibble first, as the words mxfp4 multiplies: eight to a
  /// word in the same order, so this is a view and nothing moves.
  public static func mxfp4(pairs: MLXArray, scales: MLXArray) -> (MLXArray, MLXArray) {
    let words = pairs.view(dtype: .uint8).view(dtype: .uint32)
    return (words, scales.asType(.uint8))
  }

  public static func dense(fp8 codes: MLXArray, blockScales: MLXArray) -> MLXArray {
    let (words, scales) = mxfp8(codes: codes, blockScales: blockScales)
    return dequantized(
      words, scales: scales, biases: nil, groupSize: 32, bits: 8, mode: .mxfp8,
      dtype: .float32)
  }

  public static func dense(fp4 pairs: MLXArray, scales: MLXArray) -> MLXArray {
    let (words, rowScales) = mxfp4(pairs: pairs, scales: scales)
    return dequantized(
      words, scales: rowScales, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4,
      dtype: .float32)
  }

  /// fp8 e4m3 codes, as MLX's own conversion reads them.
  public static func fromFP8(_ codes: MLXArray, dtype: DType = .float32) -> MLXArray {
    var result = mlx_array_new()
    mlx_from_fp8(&result, codes.ctx, dtype.cmlxDtype, StreamOrDevice.default.ctx)
    return MLXArray(result)
  }

  /// Rounded to the nearest e4m3 code, ties to even, the way torch's cast rounds.
  public static func toFP8(_ values: MLXArray) -> MLXArray {
    var result = mlx_array_new()
    mlx_to_fp8(&result, values.ctx, StreamOrDevice.default.ctx)
    return MLXArray(result)
  }
}
