// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// What one V4.1 layer keeps between chunks.

import Foundation
import MLX

/// A layer's window, and — on a layer that owns one — its share of the global KV.
///
/// The window is the last `window - 1` keys the next chunk reaches back over. It is replaced
/// rather than written into, so a snapshot can hold on to it for free. The compressed latents and
/// the index keys only grow, and grow in place, so a snapshot of them is just how many there were.
/// The partial compressor group rides along the same way the window does.
public final class DeepSeekLayerCache: LayerCache, @unchecked Sendable {
  public private(set) var offset = 0
  var window: MLXArray?
  private(set) var compressed: MLXArray?
  private(set) var indexKeys: MLXArray?
  public private(set) var compressedCount = 0
  var pendingKV: MLXArray?
  var pendingScore: MLXArray?
  /// The compressed ids the n-gram hash reaches back over. Only the first layer's cache holds them.
  var ngramTokens: [Int32] = []

  private let step = 256

  public init() {}

  public func reset() {
    offset = 0
    window = nil
    compressed = nil
    indexKeys = nil
    compressedCount = 0
    pendingKV = nil
    pendingScore = nil
    ngramTokens = []
  }

  func advance(_ count: Int) { offset += count }

  /// What the layer holds now: the window, the owned latents and keys, the partial group.
  public var byteCount: Int {
    (window?.nbytes ?? 0) + (pendingKV?.nbytes ?? 0) + (pendingScore?.nbytes ?? 0)
      + compressedCount * ((compressed?.dim(2) ?? 0) * (compressed?.dtype.size ?? 0)
        + (indexKeys?.dim(2) ?? 0) * (indexKeys?.dtype.size ?? 0))
  }

  /// The latents this layer owns, as far as they have been written.
  var latents: MLXArray? { compressed.map { $0[0..., ..<compressedCount, 0...] } }
  var keys: MLXArray? { indexKeys.map { $0[0..., ..<compressedCount, 0...] } }

  /// Appends newly completed groups: their latents and, for an indexing owner, their keys.
  func append(latents: MLXArray, keys: MLXArray?) {
    let added = latents.dim(1)
    guard added > 0 else { return }
    let end = compressedCount + added
    compressed = Self.grown(compressed, toHold: end, like: latents, step: step)
    compressed![0..., compressedCount..<end, 0...] = latents
    if let keys {
      indexKeys = Self.grown(indexKeys, toHold: end, like: keys, step: step)
      indexKeys![0..., compressedCount..<end, 0...] = keys
    }
    compressedCount = end
  }

  private static func grown(_ buffer: MLXArray?, toHold count: Int, like rows: MLXArray, step: Int)
    -> MLXArray
  {
    let capacity = buffer?.dim(1) ?? 0
    if let buffer, capacity >= count, buffer.dtype == rows.dtype { return buffer }
    var size = max(capacity, step)
    while size < count { size += max(size / 2, step) }
    let fresh = MLXArray.zeros([rows.dim(0), size, rows.dim(2)], dtype: rows.dtype)
    if let buffer, capacity > 0 {
      fresh[0..., ..<capacity, 0...] = buffer.asType(rows.dtype)
    }
    return fresh
  }

  public final class State: @unchecked Sendable {
    let offset: Int
    let window: MLXArray?
    let compressedCount: Int
    let pendingKV: MLXArray?
    let pendingScore: MLXArray?
    let ngramTokens: [Int32]

    init(_ cache: DeepSeekLayerCache) {
      offset = cache.offset
      window = cache.window
      compressedCount = cache.compressedCount
      pendingKV = cache.pendingKV
      pendingScore = cache.pendingScore
      ngramTokens = cache.ngramTokens
    }

    var byteCount: Int {
      (window?.nbytes ?? 0) + (pendingKV?.nbytes ?? 0) + (pendingScore?.nbytes ?? 0)
    }
  }

  public func snapshot() -> CacheSnapshot { .windowed(State(self)) }

  public func restore(_ snapshot: CacheSnapshot) {
    guard case .windowed(let state) = snapshot, state.offset <= offset else { return }
    offset = state.offset
    window = state.window
    compressedCount = min(state.compressedCount, compressedCount)
    pendingKV = state.pendingKV
    pendingScore = state.pendingScore
    ngramTokens = state.ngramTokens
  }

  public func export() -> [String: MLXArray]? {
    guard offset > 0 else { return nil }
    var arrays: [String: MLXArray] = [:]
    if let window { arrays["window"] = window }
    if let latents, compressedCount > 0 { arrays["compressed"] = latents }
    if let keys, compressedCount > 0 { arrays["index_keys"] = keys }
    if let pendingKV { arrays["pending_kv"] = pendingKV }
    if let pendingScore { arrays["pending_score"] = pendingScore }
    if !ngramTokens.isEmpty { arrays["ngram_tokens"] = MLXArray(ngramTokens) }
    return arrays
  }

  public func load(_ arrays: [String: MLXArray], offset: Int) -> Bool {
    reset()
    self.offset = offset
    window = arrays["window"]
    if let latents = arrays["compressed"] {
      append(latents: latents, keys: arrays["index_keys"])
    }
    pendingKV = arrays["pending_kv"]
    pendingScore = arrays["pending_score"]
    ngramTokens = arrays["ngram_tokens"]?.asArray(Int32.self) ?? []
    return true
  }
}
