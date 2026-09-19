// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// GGUF container reader — llama.cpp's on-disk format: header, metadata, tensor table.

import Foundation

/// A GGML storage type, with the block geometry that turns an element count into bytes.
///
/// `blockSize` elements share one block of `typeSize` bytes. The scalar types are blocks of
/// one; the k-quants and i-quants pack 256 elements into a super-block whose scales live in
/// the same bytes as the quants, which is why the size cannot be derived from a bit width.
public enum GGMLType: UInt32, Sendable {
  case f32 = 0
  case f16 = 1
  case q4_0 = 2
  case q4_1 = 3
  case q5_0 = 6
  case q5_1 = 7
  case q8_0 = 8
  case q8_1 = 9
  case q2_K = 10
  case q3_K = 11
  case q4_K = 12
  case q5_K = 13
  case q6_K = 14
  case q8_K = 15
  case iq2_xxs = 16
  case iq2_xs = 17
  case iq3_xxs = 18
  case iq1_s = 19
  case iq4_nl = 20
  case iq3_s = 21
  case iq2_s = 22
  case iq4_xs = 23
  case i8 = 24
  case i16 = 25
  case i32 = 26
  case i64 = 27
  case f64 = 28
  case iq1_m = 29
  case bf16 = 30

  public var blockSize: Int {
    switch self {
    case .f32, .f16, .bf16, .f64, .i8, .i16, .i32, .i64: 1
    case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .iq4_nl: 32
    default: 256
    }
  }

  public var typeSize: Int {
    switch self {
    case .f32, .i32: 4
    case .f16, .bf16, .i16: 2
    case .f64, .i64: 8
    case .i8: 1
    case .q4_0: 18
    case .q4_1: 20
    case .q5_0: 22
    case .q5_1: 24
    case .q8_0: 34
    case .q8_1: 36
    case .q2_K: 84
    case .q3_K: 110
    case .q4_K: 144
    case .q5_K: 176
    case .q6_K: 210
    case .q8_K: 292
    case .iq2_xxs: 66
    case .iq2_xs: 74
    case .iq3_xxs: 98
    case .iq1_s: 50
    case .iq4_nl: 18
    case .iq3_s: 110
    case .iq2_s: 82
    case .iq4_xs: 136
    case .iq1_m: 56
    }
  }

  public var isQuantized: Bool { blockSize > 1 }

  public var name: String {
    switch self {
    case .f32: "F32"
    case .f16: "F16"
    case .bf16: "BF16"
    case .f64: "F64"
    case .i8: "I8"
    case .i16: "I16"
    case .i32: "I32"
    case .i64: "I64"
    case .q4_0: "Q4_0"
    case .q4_1: "Q4_1"
    case .q5_0: "Q5_0"
    case .q5_1: "Q5_1"
    case .q8_0: "Q8_0"
    case .q8_1: "Q8_1"
    case .q2_K: "Q2_K"
    case .q3_K: "Q3_K"
    case .q4_K: "Q4_K"
    case .q5_K: "Q5_K"
    case .q6_K: "Q6_K"
    case .q8_K: "Q8_K"
    case .iq2_xxs: "IQ2_XXS"
    case .iq2_xs: "IQ2_XS"
    case .iq3_xxs: "IQ3_XXS"
    case .iq1_s: "IQ1_S"
    case .iq4_nl: "IQ4_NL"
    case .iq3_s: "IQ3_S"
    case .iq2_s: "IQ2_S"
    case .iq4_xs: "IQ4_XS"
    case .iq1_m: "IQ1_M"
    }
  }

  public func byteCount(elements: Int) -> Int {
    elements / blockSize * typeSize
  }
}

/// One metadata value. GGUF's arrays are homogeneous, so a nested array carries its own values
/// rather than a type tag per element.
public enum GGUFValue: Sendable {
  case uint(UInt64)
  case int(Int64)
  case double(Double)
  case bool(Bool)
  case string(String)
  case array([GGUFValue])

  public var intValue: Int? {
    switch self {
    case .uint(let v): Int(exactly: v)
    case .int(let v): Int(exactly: v)
    case .double(let v): Int(exactly: v.rounded())
    case .bool(let v): v ? 1 : 0
    default: nil
    }
  }

  public var floatValue: Float? {
    switch self {
    case .uint(let v): Float(v)
    case .int(let v): Float(v)
    case .double(let v): Float(v)
    default: nil
    }
  }

  public var stringValue: String? {
    if case .string(let v) = self { return v }
    return nil
  }

  public var boolValue: Bool? {
    switch self {
    case .bool(let v): v
    case .uint(let v): v != 0
    case .int(let v): v != 0
    default: nil
    }
  }

  public var arrayValue: [GGUFValue]? {
    if case .array(let v) = self { return v }
    return nil
  }

  public var intArray: [Int]? { arrayValue?.compactMap(\.intValue) }
  public var stringArray: [String]? { arrayValue?.compactMap(\.stringValue) }
}

/// A GGUF file's header and tensor table, with the weight bytes left on disk.
///
/// Reading the table is cheap even for a 27B pack: the tensor data is addressed by offset into
/// a memory-mapped region, so nothing is faulted in until a tensor is actually asked for.
public struct GGUFFile: Sendable {
  public struct TensorInfo: Sendable {
    public let name: String
    /// GGUF stores dimensions fastest-varying first; `shape` is the row-major order the rest
    /// of the runtime uses, so a `[5120, 248320]` table reads as 248320 rows of 5120.
    public let shape: [Int]
    public let type: GGMLType
    public let offset: Int
    public let byteCount: Int

    public var elementCount: Int { shape.reduce(1, *) }
  }

  public static let magic: UInt32 = 0x4655_4747

  public let url: URL
  public let version: UInt32
  public let metadata: [String: GGUFValue]
  public let tensors: [TensorInfo]
  public let dataOffset: Int

  private let index: [String: Int]

  public subscript(tensor name: String) -> TensorInfo? {
    index[name].map { tensors[$0] }
  }

  public subscript(_ key: String) -> GGUFValue? { metadata[key] }

  /// The `general.architecture` prefix every hyperparameter key is namespaced under.
  public var architecture: String {
    metadata["general.architecture"]?.stringValue ?? ""
  }

  public func architectureValue(_ suffix: String) -> GGUFValue? {
    metadata["\(architecture).\(suffix)"]
  }

  public init(url: URL) throws {
    self.url = url
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    var reader = try Reader(handle: handle)
    guard try reader.u32() == Self.magic else {
      throw BonsaiError.unsupportedModel("\(url.lastPathComponent) is not a GGUF file")
    }
    let version = try reader.u32()
    guard version == 2 || version == 3 else {
      throw BonsaiError.unsupportedModel("GGUF version \(version) is not supported")
    }
    self.version = version

    let tensorCount = Int(try reader.u64())
    let metadataCount = Int(try reader.u64())
    guard tensorCount >= 0, tensorCount < 1_000_000, metadataCount < 1_000_000 else {
      throw BonsaiError.unsupportedModel("GGUF header counts are implausible")
    }

    var metadata: [String: GGUFValue] = [:]
    metadata.reserveCapacity(metadataCount)
    for _ in 0..<metadataCount {
      let key = try reader.string()
      metadata[key] = try reader.value(type: try reader.u32())
    }
    self.metadata = metadata

    let alignment = metadata["general.alignment"]?.intValue ?? 32
    guard alignment > 0, alignment & (alignment - 1) == 0 else {
      throw BonsaiError.unsupportedModel("GGUF alignment \(alignment) is not a power of two")
    }

    var tensors: [TensorInfo] = []
    var index: [String: Int] = [:]
    tensors.reserveCapacity(tensorCount)
    for _ in 0..<tensorCount {
      let name = try reader.string()
      let rank = Int(try reader.u32())
      guard rank >= 1, rank <= 4 else {
        throw BonsaiError.unsupportedModel("\(name) has rank \(rank)")
      }
      var dims: [Int] = []
      for _ in 0..<rank { dims.append(Int(try reader.u64())) }
      let raw = try reader.u32()
      guard let type = GGMLType(rawValue: raw) else {
        throw BonsaiError.unsupportedModel("\(name) has unknown GGML type \(raw)")
      }
      let offset = Int(try reader.u64())
      let elements = dims.reduce(1, *)
      guard elements % type.blockSize == 0 else {
        throw BonsaiError.shapeMismatch(
          "\(name) has \(elements) elements, not a multiple of \(type.name)'s "
            + "\(type.blockSize)-element block")
      }
      index[name] = tensors.count
      tensors.append(
        TensorInfo(
          name: name, shape: dims.reversed(), type: type, offset: offset,
          byteCount: type.byteCount(elements: elements)))
    }
    self.tensors = tensors
    self.index = index

    let unaligned = reader.position
    self.dataOffset = (unaligned + alignment - 1) / alignment * alignment
  }

  /// The bytes of one tensor, read from the data section.
  public func data(for tensor: TensorInfo) throws -> Data {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    try handle.seek(toOffset: UInt64(dataOffset + tensor.offset))
    guard let bytes = try handle.read(upToCount: tensor.byteCount),
      bytes.count == tensor.byteCount
    else {
      throw BonsaiError.missingWeight(
        "\(tensor.name) runs past the end of \(url.lastPathComponent)")
    }
    return bytes
  }

  /// Every storage type in the file with how many tensors use it, for reporting.
  public var typeHistogram: [(type: GGMLType, count: Int)] {
    var counts: [UInt32: Int] = [:]
    for tensor in tensors { counts[tensor.type.rawValue, default: 0] += 1 }
    return counts.sorted { $0.value > $1.value }
      .compactMap { key, count in
        GGMLType(rawValue: key).map { (type: $0, count: count) }
      }
  }

  private struct Reader {
    private let handle: FileHandle
    private var buffer: Data
    private var base: Int
    private var cursor: Int

    var position: Int { base + cursor }

    init(handle: FileHandle) throws {
      self.handle = handle
      self.buffer = Data()
      self.base = 0
      self.cursor = 0
    }

    private mutating func need(_ count: Int) throws {
      guard cursor + count > buffer.count else { return }
      if cursor > 0 {
        base += cursor
        buffer = buffer.subdata(in: cursor..<buffer.count)
        cursor = 0
      }
      while buffer.count < count {
        let chunk = max(count - buffer.count, 1 << 20)
        guard let more = try handle.read(upToCount: chunk), !more.isEmpty else {
          throw BonsaiError.unsupportedModel("GGUF header ends mid-field")
        }
        buffer.append(more)
      }
    }

    private mutating func scalar<T>(_ type: T.Type) throws -> T {
      let size = MemoryLayout<T>.size
      try need(size)
      let value = buffer.withUnsafeBytes { raw in
        raw.loadUnaligned(fromByteOffset: cursor, as: T.self)
      }
      cursor += size
      return value
    }

    mutating func u32() throws -> UInt32 { UInt32(littleEndian: try scalar(UInt32.self)) }
    mutating func u64() throws -> UInt64 { UInt64(littleEndian: try scalar(UInt64.self)) }

    mutating func string() throws -> String {
      let length = Int(try u64())
      guard length >= 0, length < 1 << 28 else {
        throw BonsaiError.unsupportedModel("GGUF string length \(length) is implausible")
      }
      try need(length)
      let bytes = buffer.subdata(in: cursor..<(cursor + length))
      cursor += length
      return String(decoding: bytes, as: UTF8.self)
    }

    mutating func value(type: UInt32) throws -> GGUFValue {
      switch type {
      case 0: .uint(UInt64(try scalar(UInt8.self)))
      case 1: .int(Int64(try scalar(Int8.self)))
      case 2: .uint(UInt64(UInt16(littleEndian: try scalar(UInt16.self))))
      case 3: .int(Int64(Int16(littleEndian: try scalar(Int16.self))))
      case 4: .uint(UInt64(try u32()))
      case 5: .int(Int64(Int32(littleEndian: try scalar(Int32.self))))
      case 6: .double(Double(Float(bitPattern: try u32())))
      case 7: .bool(try scalar(UInt8.self) != 0)
      case 8: .string(try string())
      case 9: try array()
      case 10: .uint(try u64())
      case 11: .int(Int64(bitPattern: try u64()))
      case 12: .double(Double(bitPattern: try u64()))
      default:
        throw BonsaiError.unsupportedModel("GGUF metadata type \(type) is unknown")
      }
    }

    private mutating func array() throws -> GGUFValue {
      let element = try u32()
      let count = Int(try u64())
      guard count >= 0, count < 1 << 26 else {
        throw BonsaiError.unsupportedModel("GGUF array of \(count) is implausible")
      }
      var values: [GGUFValue] = []
      values.reserveCapacity(count)
      for _ in 0..<count { values.append(try value(type: element)) }
      return .array(values)
    }
  }
}
