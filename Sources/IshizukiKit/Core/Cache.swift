// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public protocol LayerCache: AnyObject {
  var offset: Int { get }
  func reset()

  func snapshot() -> CacheSnapshot
  func restore(_ snapshot: CacheSnapshot)

  /// Everything needed to rebuild this layer, keyed within the layer, or nil if it holds
  /// nothing worth writing. The offset travels separately, in the archive's metadata.
  func export() -> [String: MLXArray]?

  /// Rebuilds the layer from an export. False leaves the cache untouched and the archive
  /// unusable, which the caller treats as a miss rather than an error.
  func load(_ arrays: [String: MLXArray], offset: Int) -> Bool
}

public enum CacheSnapshot: @unchecked Sendable {
  case kv(offset: Int)
  case recurrent(
    conv: MLXArray?, state: MLXArray?, ple: MLXArray?, pleTokens: MLXArray?, offset: Int)
  case windowed(DeepSeekLayerCache.State)

  /// An attention layer rewinds by moving an offset, so its snapshot is free. A recurrent layer
  /// has to keep the state itself, which is what makes holding many of them expensive.
  public var byteCount: Int {
    switch self {
    case .kv: 0
    case .recurrent(let conv, let state, let ple, let tokens, _):
      (conv?.nbytes ?? 0) + (state?.nbytes ?? 0) + (ple?.nbytes ?? 0) + (tokens?.nbytes ?? 0)
    case .windowed(let state): state.byteCount
    }
  }
}

public final class KVCache: LayerCache, @unchecked Sendable {
  public private(set) var keys: MLXArray?
  public private(set) var values: MLXArray?
  public var indexerKeys: MLXArray?
  public private(set) var offset = 0
  public let step: Int

  public init(step: Int = 256) {
    self.step = step
  }

  public func reset() {
    keys = nil
    values = nil
    indexerKeys = nil
    offset = 0
  }

  public func snapshot() -> CacheSnapshot { .kv(offset: offset) }

  public func restore(_ snapshot: CacheSnapshot) {
    guard case .kv(let restored) = snapshot else { return }
    offset = min(restored, offset)
  }

  public func trim(to length: Int) {
    offset = min(length, offset)
  }

  public func export() -> [String: MLXArray]? {
    guard offset > 0, let keys, let values else { return nil }
    var arrays: [String: MLXArray] = [
      "keys": keys[0..., 0..., ..<offset, 0...],
      "values": values[0..., 0..., ..<offset, 0...],
    ]
    if let indexerKeys { arrays["ik"] = indexerKeys[0..., ..<offset, 0...] }
    return arrays
  }

  public func load(_ arrays: [String: MLXArray], offset: Int) -> Bool {
    guard let keys = arrays["keys"], let values = arrays["values"],
      keys.dim(2) >= offset, values.dim(2) >= offset
    else { return false }
    self.keys = keys
    self.values = values
    self.indexerKeys = arrays["ik"]
    self.offset = offset
    return true
  }

  public func reserve(_ tokens: Int, shapedLike keys: MLXArray, values: MLXArray) {
    guard tokens > (self.keys?.dim(2) ?? 0) else { return }
    grow(to: tokens, like: keys, valuesLike: values, keeping: offset)
  }

  private func grow(
    to required: Int, like newKeys: MLXArray, valuesLike newValues: MLXArray, keeping: Int
  ) {
    let b = newKeys.dim(0)
    let h = newKeys.dim(1)
    let d = newKeys.dim(3)
    let dv = newValues.dim(3)

    let current = keys?.dim(2) ?? 0
    var capacity = max(current, step)
    while capacity < required { capacity += max(capacity / 2, step) }

    let grownKeys = MLXArray.zeros([b, h, capacity, d], dtype: newKeys.dtype)
    let grownValues = MLXArray.zeros([b, h, capacity, dv], dtype: newValues.dtype)

    if let existingKeys = keys, let existingValues = values, keeping > 0 {
      grownKeys[0..., 0..., ..<keeping, 0...] = existingKeys[0..., 0..., ..<keeping, 0...]
      grownValues[0..., 0..., ..<keeping, 0...] =
        existingValues[0..., 0..., ..<keeping, 0...]
    }
    keys = grownKeys
    values = grownValues
  }

  public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (
    MLXArray, MLXArray
  ) {
    let previous = offset
    let added = newKeys.dim(2)

    if keys == nil || (previous + added) > keys!.dim(2) {
      grow(
        to: previous + added, like: newKeys, valuesLike: newValues, keeping: previous)
    }

    keys![0..., 0..., previous..<(previous + added), 0...] = newKeys
    values![0..., 0..., previous..<(previous + added), 0...] = newValues
    offset += added

    return (
      keys![0..., 0..., ..<offset, 0...],
      values![0..., 0..., ..<offset, 0...]
    )
  }
}

public final class GatedDeltaNetCache: LayerCache, @unchecked Sendable {
  public var convState: MLXArray?
  public var recurrentState: MLXArray?
  /// The PLE block rides on one linear layer's cache, as it does upstream: a short convolution
  /// state, and the handful of tokens its n-grams reach back over.
  public var pleConvState: MLXArray?
  public var pleTokens: MLXArray?
  public private(set) var offset = 0

  public init() {}

  public func reset() {
    convState = nil
    recurrentState = nil
    pleConvState = nil
    pleTokens = nil
    offset = 0
  }

  public func snapshot() -> CacheSnapshot {
    .recurrent(
      conv: convState, state: recurrentState, ple: pleConvState, pleTokens: pleTokens,
      offset: offset)
  }

  public func restore(_ snapshot: CacheSnapshot) {
    guard case .recurrent(let conv, let state, let ple, let tokens, let restored) = snapshot
    else { return }
    convState = conv
    recurrentState = state
    pleConvState = ple
    pleTokens = tokens
    offset = restored
  }

  public func advance(_ count: Int) {
    offset += count
  }

  public func export() -> [String: MLXArray]? {
    guard offset > 0 else { return nil }
    var arrays: [String: MLXArray] = [:]
    if let convState { arrays["conv"] = convState }
    if let recurrentState { arrays["state"] = recurrentState }
    if let pleConvState { arrays["ple_conv"] = pleConvState }
    if let pleTokens { arrays["ple_tokens"] = pleTokens }
    return arrays.isEmpty ? nil : arrays
  }

  public func load(_ arrays: [String: MLXArray], offset: Int) -> Bool {
    convState = arrays["conv"]
    recurrentState = arrays["state"]
    pleConvState = arrays["ple_conv"]
    pleTokens = arrays["ple_tokens"]
    self.offset = offset
    return true
  }
}

public final class ModelCache: @unchecked Sendable {
  public let layers: [LayerCache]

  public let kvConfig: KVCacheConfig

  public convenience init(
    config: BonsaiConfig.TextConfig, kvConfig: KVCacheConfig = KVCacheConfig()
  ) {
    self.init(fullAttention: config.isFullAttention, kvConfig: kvConfig)
  }

  /// A draft head runs its own short stack, so the schedule is taken as given rather than read
  /// off the backbone's config.
  public init(fullAttention: [Bool], kvConfig: KVCacheConfig = KVCacheConfig()) {
    self.kvConfig = kvConfig
    self.layers = fullAttention.map { isFull in
      guard isFull else { return GatedDeltaNetCache() as LayerCache }
      return kvConfig.isQuantized
        ? QuantizedKVCache(config: kvConfig) as LayerCache
        : KVCache() as LayerCache
    }
  }

  /// Layers built elsewhere, for an architecture whose caches are not the ones a config's
  /// attention schedule describes.
  public init(layers: [LayerCache], kvConfig: KVCacheConfig = KVCacheConfig()) {
    self.layers = layers
    self.kvConfig = kvConfig
  }

  public var byteCount: Int {
    layers.reduce(0) { total, layer in
      if let attention = layer as? AttentionKVCache { return total + attention.byteCount }
      if let windowed = layer as? DeepSeekLayerCache { return total + windowed.byteCount }
      return total
    }
  }

  public var offset: Int {
    layers.first(where: { $0 is AttentionKVCache })?.offset ?? layers.first?.offset ?? 0
  }

  public func reset() {
    for layer in layers { layer.reset() }
  }

  public func snapshot() -> [CacheSnapshot] { layers.map { $0.snapshot() } }

  public func restore(_ snapshots: [CacheSnapshot]) {
    for (layer, snapshot) in zip(layers, snapshots) { layer.restore(snapshot) }
  }

  public var hasRecurrentLayers: Bool {
    layers.contains { $0 is GatedDeltaNetCache || $0 is DeepSeekLayerCache }
  }
}
