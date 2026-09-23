// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

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
    /// The conversation this prefix belongs to, when the request that built it named one.
    public var tag: String? = nil
    /// Shared by every conversation, such as the instructions and tool schemas on their own.
    public var pinned = false
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
  private var headers: [String: (size: Int, modified: Date, entry: Entry)] = [:]
  private var changes = 0

  /// Bumped whenever an archive is written or removed, so a reader can tell when to look again.
  public var revision: Int {
    lock.lock()
    defer { lock.unlock() }
    return changes
  }

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
    cache: ModelCache, tokens: [Int], modelID: String, kvConfig: KVCacheConfig,
    tag: String? = nil, pinned: Bool = false
  ) -> String? {
    guard byteLimit > 0, pinned || tokens.count >= minimumTokens, !tokens.isEmpty,
      cache.offset == tokens.count
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

    var metadata: [String: String] = [
      "version": "\(Self.formatVersion)",
      "fingerprint": fingerprint,
      "offsets": offsets.map(String.init).joined(separator: ","),
      "tokens": tokens.map(String.init).joined(separator: ","),
    ]
    if let tag { metadata["tag"] = tag }
    if pinned { metadata["pinned"] = "1" }

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
      changes += 1
      if let tag, !pinned {
        for older in unlockedEntries()
        where older.id != id && older.tag == tag && !older.pinned
          && older.id.hasPrefix(fingerprint + "-")
        {
          try? fm.removeItem(at: url(older.id, "safetensors"))
        }
      }
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
    else {
      headers.removeAll()
      return []
    }

    var seen: Set<String> = []
    let found = names.compactMap { name -> Entry? in
      guard name.hasSuffix(".safetensors"), !name.contains(".partial.") else { return nil }
      let id = String(name.dropLast(".safetensors".count))
      let path = directory.appending(path: name)
      let attributes = try? fm.attributesOfItem(atPath: path.path)
      let size = (attributes?[.size] as? NSNumber)?.intValue ?? 0
      let used = (attributes?[.modificationDate] as? Date) ?? .distantPast
      seen.insert(id)
      if let known = headers[id], known.size == size, known.modified == used {
        return known.entry
      }
      guard
        let metadata = Self.metadata(at: path),
        metadata["version"] == "\(Self.formatVersion)",
        let encoded = metadata["tokens"]
      else { return nil }
      let tokens = encoded.split(separator: ",").compactMap { Int($0) }
      let entry = Entry(
        id: id, tokens: tokens, byteCount: size, lastUsed: used, tag: metadata["tag"],
        pinned: metadata["pinned"] == "1")
      headers[id] = (size, used, entry)
      return entry
    }
    headers = headers.filter { seen.contains($0.key) }
    return found
  }

  /// A safetensors header on its own: eight bytes of length, then that much JSON. Reading only
  /// this keeps a listing from opening every archive's arrays.
  static func metadata(at url: URL) -> [String: String]? {
    guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
    defer { try? handle.close() }
    guard let prefix = try? handle.read(upToCount: 8), prefix.count == 8 else { return nil }
    let length = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
    guard length > 0, length < 256 << 20,
      let header = try? handle.read(upToCount: Int(length)), header.count == Int(length),
      let object = try? JSONSerialization.jsonObject(with: header) as? [String: Any]
    else { return nil }
    return object["__metadata__"] as? [String: String]
  }

  /// What each conversation's archives hold on disk, keyed by tag.
  public func usage() -> [String: (bytes: Int, tokens: Int)] {
    var out: [String: (bytes: Int, tokens: Int)] = [:]
    for entry in entries() {
      guard let tag = entry.tag else { continue }
      let held = out[tag] ?? (0, 0)
      out[tag] = (held.bytes + entry.byteCount, max(held.tokens, entry.tokens.count))
    }
    return out
  }

  /// Drops every archive a conversation owns. Returns the bytes given back.
  @discardableResult
  public func remove(tag: String) -> Int {
    lock.lock()
    defer { lock.unlock() }
    var freed = 0
    for entry in unlockedEntries() where entry.tag == tag {
      if (try? fm.removeItem(at: url(entry.id, "safetensors"))) != nil {
        freed += entry.byteCount
      }
    }
    if freed > 0 { changes += 1 }
    return freed
  }

  public var totalBytes: Int {
    entries().reduce(0) { $0 + $1.byteCount }
  }

  private let glanceLock = NSLock()
  private var lastGlance = (bytes: 0, revision: -1)

  /// Bytes held and the revision they were counted at, without waiting on a save in progress.
  public func glance() -> (bytes: Int, revision: Int) {
    guard lock.try() else { return glanceLock.withLock { lastGlance } }
    let fresh = (unlockedEntries().reduce(0) { $0 + $1.byteCount }, changes)
    lock.unlock()
    glanceLock.withLock { lastGlance = fresh }
    return fresh
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
    held.sort { ($0.pinned ? 1 : 0, $0.lastUsed) < ($1.pinned ? 1 : 0, $1.lastUsed) }
    for entry in held where total > byteLimit {
      try? fm.removeItem(at: url(entry.id, "safetensors"))
      total -= entry.byteCount
    }
    changes += 1
  }

  /// Drops one archive by id.
  @discardableResult
  public func remove(_ id: String) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    changes += 1
    return (try? fm.removeItem(at: url(id, "safetensors"))) != nil
  }

  public func removeAll() {
    lock.lock()
    defer { lock.unlock() }
    for entry in unlockedEntries() {
      try? fm.removeItem(at: url(entry.id, "safetensors"))
    }
    changes += 1
  }

  private func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
    var index = 0
    let limit = min(a.count, b.count)
    while index < limit, a[index] == b[index] { index += 1 }
    return index
  }
}
