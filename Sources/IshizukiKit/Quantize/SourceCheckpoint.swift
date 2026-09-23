// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

/// A full-precision checkpoint, read a tensor at a time.
///
/// MLX maps safetensors rather than reading them: opening a 5 GB shard costs about a megabyte,
/// and a tensor is only paid for when it is evaluated. That is what lets a 52 GB source be
/// quantized on a machine that could not hold it — nothing is resident but the layer in hand.
public final class SourceCheckpoint: @unchecked Sendable {
  public let directory: URL
  public let config: [String: Any]
  public let textConfig: [String: Any]

  private let shardOf: [String: String]
  /// Names this checkpoint does not hold, each cut from one that it does. A fused expert bank
  /// is the only case: upstream stacks a layer's gate and up projections into one tensor, and
  /// everything downstream reads them apart.
  private var derived: [String: (source: String, half: Int?)] = [:]
  private let lock = NSLock()
  private var opened: [String: [String: MLXArray]] = [:]
  private var order: [String] = []
  private let residentShards: Int
  /// Where each tensor's bytes sit in its shard, read from the headers as shards are opened.
  private var spans: [String: [String: Range<Int>]] = [:]

  public init(directory: URL, residentShards: Int = 2) throws {
    self.directory = directory
    self.residentShards = max(1, residentShards)

    let configURL = directory.appending(path: "config.json")
    guard let raw = try? Data(contentsOf: configURL),
      let object = try? JSONSerialization.jsonObject(with: raw) as? [String: Any]
    else {
      throw BonsaiError.unsupportedModel("no readable config.json in \(directory.path)")
    }
    self.config = object
    self.textConfig = (object["text_config"] as? [String: Any]) ?? object

    let fm = FileManager.default
    let single = directory.appending(path: "model.safetensors")
    let index = directory.appending(path: "model.safetensors.index.json")
    if fm.fileExists(atPath: index.path) {
      guard
        let payload = try? JSONSerialization.jsonObject(with: try Data(contentsOf: index)),
        let map = (payload as? [String: Any])?["weight_map"] as? [String: String]
      else {
        throw BonsaiError.missingWeight("unreadable weight_map in \(index.path)")
      }
      self.shardOf = map
    } else if fm.fileExists(atPath: single.path) {
      let names = try loadArrays(url: single).keys
      self.shardOf = Dictionary(uniqueKeysWithValues: names.map { ($0, "model.safetensors") })
    } else {
      throw BonsaiError.missingWeight("no safetensors in \(directory.path)")
    }

    for name in shardOf.keys {
      guard let base = Self.fusedExpertBase(name) else { continue }
      if name.hasSuffix(".gate_up_proj") {
        derived[base + ".gate_proj.weight"] = (source: name, half: 0)
        derived[base + ".up_proj.weight"] = (source: name, half: 1)
      } else {
        derived[base + ".down_proj.weight"] = (source: name, half: nil)
      }
    }
  }

  /// `…mlp.experts.gate_up_proj` and `…mlp.experts.down_proj` become `…mlp.switch_mlp.*`,
  /// which is where this runtime looks for a bank of experts.
  private static func fusedExpertBase(_ name: String) -> String? {
    for suffix in [".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"]
    where name.hasSuffix(suffix) {
      return String(name.dropLast(suffix.count)) + ".mlp.switch_mlp"
    }
    return nil
  }

  public var tensorNames: [String] {
    Array(shardOf.keys.filter { Self.fusedExpertBase($0) == nil }) + Array(derived.keys)
  }

  public func has(_ name: String) -> Bool {
    (shardOf[name] != nil && Self.fusedExpertBase(name) == nil) || derived[name] != nil
  }

  /// The tensor, still unevaluated. Reading it costs nothing until it is used.
  public func tensor(_ name: String) throws -> MLXArray {
    if let cut = derived[name] {
      let fused = try tensor(cut.source)
      guard let half = cut.half else { return fused }
      // `linear(x, gate_up[e])` writes gate into the first half of its output and up into the
      // second, so the split is along the output axis, not the input one.
      let width = fused.dim(1) / 2
      return fused[0..., (half * width)..<((half + 1) * width), 0...]
    }
    guard let shard = shardOf[name] else {
      throw BonsaiError.missingWeight(name)
    }
    lock.lock()
    defer { lock.unlock() }

    if let cached = opened[shard]?[name] { return cached }
    let arrays = try loadArrays(url: directory.appending(path: shard))
    opened[shard] = arrays
    order.removeAll { $0 == shard }
    order.append(shard)
    // Dropping a shard releases its mapping; anything already taken from it stays valid.
    while order.count > residentShards {
      opened.removeValue(forKey: order.removeFirst())
    }
    guard let array = arrays[name] else {
      throw BonsaiError.missingWeight("\(name) is named by the index but absent from \(shard)")
    }
    return array
  }

  /// The tensor, with its bytes read once from the CPU first so they are in the page cache
  /// before a kernel touches the mapping. A GPU faulting its way through a mapping on a spinning
  /// disk stalls long enough for its command buffer to be killed as hung. Anything about to be
  /// computed on asks for this; anything that only wants a shape asks for `tensor`.
  public func resident(_ name: String) throws -> MLXArray {
    if let cut = derived[name] {
      _ = try resident(cut.source)
      return try tensor(name)
    }
    let array = try tensor(name)
    guard let shard = shardOf[name] else { return array }
    lock.lock()
    defer { lock.unlock() }
    try warm(directory.appending(path: shard), shard: shard, name: name)
    return array
  }

  private func warm(_ url: URL, shard: String, name: String) throws {
    if spans[shard] == nil { spans[shard] = try Self.spans(of: url) }
    guard let span = spans[shard]?[name], !span.isEmpty else { return }
    let handle = open(url.path, O_RDONLY)
    guard handle >= 0 else { throw BonsaiError.missingWeight("cannot open \(shard)") }
    defer { close(handle) }
    let chunk = 16 << 20
    let scratch = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: 16384)
    defer { scratch.deallocate() }
    var offset = span.lowerBound
    while offset < span.upperBound {
      let got = pread(handle, scratch, min(chunk, span.upperBound - offset), off_t(offset))
      guard got > 0 else { throw BonsaiError.missingWeight("\(name) ends early in \(shard)") }
      offset += got
    }
  }

  /// Every tensor's absolute byte range in one safetensors file.
  static func spans(of url: URL) throws -> [String: Range<Int>] {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
      throw BonsaiError.missingWeight("\(url.lastPathComponent) has no safetensors header")
    }
    let length = prefix.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
    guard let header = try handle.read(upToCount: length),
      let entries = try JSONSerialization.jsonObject(with: header) as? [String: Any]
    else {
      throw BonsaiError.missingWeight("\(url.lastPathComponent) has an unreadable header")
    }
    var spans: [String: Range<Int>] = [:]
    for (name, entry) in entries {
      guard let offsets = (entry as? [String: Any])?["data_offsets"] as? [NSNumber],
        offsets.count == 2
      else { continue }
      spans[name] = (8 + length + offsets[0].intValue)..<(8 + length + offsets[1].intValue)
    }
    return spans
  }

  public func optional(_ name: String) -> MLXArray? { try? tensor(name) }

  public var layerCount: Int {
    (textConfig["num_hidden_layers"] as? NSNumber)?.intValue ?? 0
  }

  public var hiddenSize: Int {
    (textConfig["hidden_size"] as? NSNumber)?.intValue ?? 0
  }

  /// The prefix the language model's tensors carry, which a VLM checkpoint nests.
  public var tensorPrefix: String {
    has("language_model.model.norm.weight") ? "language_model." : ""
  }

  public var layerTypes: [String] {
    if let types = textConfig["layer_types"] as? [String], types.count == layerCount {
      return types
    }
    let interval = (textConfig["full_attention_interval"] as? NSNumber)?.intValue ?? 4
    return (0..<layerCount).map { ($0 + 1) % interval == 0 ? "full_attention" : "linear_attention" }
  }
}
