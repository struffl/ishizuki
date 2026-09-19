// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// Cached prefixes written to disk, so a restart or an idle unload does not cost a full
/// re-prefill of a conversation that is still going.
///
/// This is a trade, not a free win: a 32k prefix is a few hundred megabytes, and writing it
/// only pays for itself against the prefill it saves. Short prefixes are left alone.
public final class PrefixStore: @unchecked Sendable {
  public struct Entry: Sendable {
    public let id: String
    public let tokens: [Int]
    public let byteCount: Int
    public let lastUsed: Date
  }

  /// Bumped whenever the archive layout changes, so an old cache is ignored rather than
  /// misread into a live model.
  public static let formatVersion = 1

  public let directory: URL
  public private(set) var byteLimit: Int
  public let minimumTokens: Int

  private let lock = NSLock()
  private let fm = FileManager.default
  private var failure: String?

  /// Why the last save was dropped. Writing a prefix is best-effort — a full disk is not worth
  /// failing a request over — but a silent nil is no way to find that out.
  public var lastError: String? {
    lock.lock()
    defer { lock.unlock() }
    return failure
  }

  public init(
    directory: URL, byteLimit: Int = 8 << 30, minimumTokens: Int = 2048
  ) {
    self.directory = directory
    self.byteLimit = byteLimit
    self.minimumTokens = minimumTokens
  }

  public func setByteLimit(_ bytes: Int) {
    lock.lock()
    byteLimit = max(0, bytes)
    lock.unlock()
    evictToLimit()
  }

  // MARK: - Naming

  /// A prefix belongs to one model and one cache geometry; anything else would be read as
  /// numbers that happen to be the right shape.
  private func fingerprint(modelID: String, kvConfig: KVCacheConfig) -> String {
    let parts = [
      "v\(Self.formatVersion)", modelID,
      "k\(kvConfig.keyBits)", "v\(kvConfig.valueBits)",
      "g\(kvConfig.groupSize)", "w\(kvConfig.residualWindow)",
    ]
    return Self.digest(parts.joined(separator: "/"))
  }

  private static func digest(_ string: String) -> String {
    // FNV-1a: this names a cache file, it does not defend against anything.
    var hash: UInt64 = 0xcbf2_9ce4_8422_2325
    for byte in string.utf8 {
      hash ^= UInt64(byte)
      hash &*= 0x100_0000_01b3
    }
    return String(hash, radix: 36)
  }

  private func url(_ id: String, _ ext: String) -> URL {
    directory.appending(path: "\(id).\(ext)")
  }

  // MARK: - Writing

  /// Returns the id written, or nil when the prefix was not worth keeping.
  @discardableResult
  public func save(
    cache: ModelCache, tokens: [Int], modelID: String, kvConfig: KVCacheConfig
  ) -> String? {
    guard byteLimit > 0, tokens.count >= minimumTokens, cache.offset == tokens.count
    else { return nil }

    var arrays: [String: MLXArray] = [:]
    var offsets: [Int] = []
    for (index, layer) in cache.layers.enumerated() {
      offsets.append(layer.offset)
      guard let exported = layer.export() else { continue }
      for (key, array) in exported { arrays["\(index).\(key)"] = array }
    }
    guard !arrays.isEmpty else { return nil }

    let fingerprint = fingerprint(modelID: modelID, kvConfig: kvConfig)
    let id = "\(fingerprint)-\(Self.digest(tokens.map(String.init).joined(separator: ",")))"

    let metadata: [String: String] = [
      "version": "\(Self.formatVersion)",
      "fingerprint": fingerprint,
      "offsets": offsets.map(String.init).joined(separator: ","),
      "tokens": tokens.map(String.init).joined(separator: ","),
    ]

    lock.lock()
    defer { lock.unlock() }
    do {
      try fm.createDirectory(at: directory, withIntermediateDirectories: true)
      eval(Array(arrays.values))
      // Written beside the target and moved in, so a crash mid-write leaves no half archive
      // for the next start to trust.
      let staging = url("\(id).partial", "safetensors")
      try MLX.save(arrays: arrays, metadata: metadata, url: staging)
      let final = url(id, "safetensors")
      if fm.fileExists(atPath: final.path) { try fm.removeItem(at: final) }
      try fm.moveItem(at: staging, to: final)
      return id
    } catch {
      failure = "\(error)"
      try? fm.removeItem(at: url("\(id).partial", "safetensors"))
      return nil
    }
  }

  // MARK: - Reading

  /// The archive holding the longest prefix of `tokens` for this model, if any.
  public func bestMatch(
    for tokens: [Int], modelID: String, kvConfig: KVCacheConfig
  ) -> Entry? {
    let wanted = fingerprint(modelID: modelID, kvConfig: kvConfig)
    return entries().filter { $0.id.hasPrefix(wanted + "-") }
      .filter { entry in
        let shared = commonPrefixLength(entry.tokens, tokens)
        return shared == entry.tokens.count && shared < tokens.count
      }
      .max { $0.tokens.count < $1.tokens.count }
  }

  /// Fills `cache` from an archive. False means the archive did not fit the cache it was
  /// handed — a stale or truncated file — and the cache is reset rather than left half filled.
  public func load(_ entry: Entry, into cache: ModelCache) -> Bool {
    guard
      let (arrays, metadata) = try? loadArraysAndMetadata(
        url: url(entry.id, "safetensors")),
      metadata["version"] == "\(Self.formatVersion)",
      let offsets = metadata["offsets"]?.split(separator: ",").map({ Int($0) ?? -1 }),
      offsets.count == cache.layers.count, !offsets.contains(-1)
    else { return false }

    var grouped: [Int: [String: MLXArray]] = [:]
    for (key, array) in arrays {
      let parts = key.split(separator: ".", maxSplits: 1)
      guard parts.count == 2, let index = Int(parts[0]) else { return false }
      grouped[index, default: [:]][String(parts[1])] = array
    }

    for (index, layer) in cache.layers.enumerated() {
      guard layer.load(grouped[index] ?? [:], offset: offsets[index]) else {
        cache.reset()
        return false
      }
    }
    touch(entry.id)
    return true
  }

  // MARK: - Housekeeping

  public func entries() -> [Entry] {
    lock.lock()
    defer { lock.unlock() }
    return unlockedEntries()
  }

  private func unlockedEntries() -> [Entry] {
    guard
      let names = try? fm.contentsOfDirectory(atPath: directory.path)
    else { return [] }

    return names.compactMap { name -> Entry? in
      guard name.hasSuffix(".safetensors"), !name.contains(".partial.") else { return nil }
      let id = String(name.dropLast(".safetensors".count))
      let path = directory.appending(path: name)
      let attributes = try? fm.attributesOfItem(atPath: path.path)
      let size = (attributes?[.size] as? NSNumber)?.intValue ?? 0
      let used = (attributes?[.modificationDate] as? Date) ?? .distantPast
      guard
        let (_, metadata) = try? loadArraysAndMetadata(url: path),
        metadata["version"] == "\(Self.formatVersion)",
        let encoded = metadata["tokens"]
      else { return nil }
      let tokens = encoded.split(separator: ",").compactMap { Int($0) }
      return Entry(id: id, tokens: tokens, byteCount: size, lastUsed: used)
    }
  }

  public var totalBytes: Int {
    entries().reduce(0) { $0 + $1.byteCount }
  }

  private func touch(_ id: String) {
    lock.lock()
    defer { lock.unlock() }
    try? fm.setAttributes(
      [.modificationDate: Date()], ofItemAtPath: url(id, "safetensors").path)
  }

  public func evictToLimit() {
    lock.lock()
    defer { lock.unlock() }
    var held = unlockedEntries()
    var total = held.reduce(0) { $0 + $1.byteCount }
    guard byteLimit >= 0, total > byteLimit else { return }
    held.sort { $0.lastUsed < $1.lastUsed }
    for entry in held where total > byteLimit {
      try? fm.removeItem(at: url(entry.id, "safetensors"))
      total -= entry.byteCount
    }
  }

  public func removeAll() {
    lock.lock()
    defer { lock.unlock() }
    for entry in unlockedEntries() {
      try? fm.removeItem(at: url(entry.id, "safetensors"))
    }
  }

  private func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
    var index = 0
    let limit = min(a.count, b.count)
    while index < limit, a[index] == b[index] { index += 1 }
    return index
  }
}
