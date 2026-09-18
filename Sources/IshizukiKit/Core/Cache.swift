// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

public protocol LayerCache: AnyObject {
  var offset: Int { get }
  func reset()

  func snapshot() -> CacheSnapshot
  func restore(_ snapshot: CacheSnapshot)
}

public enum CacheSnapshot: @unchecked Sendable {
  case kv(offset: Int)
  case recurrent(conv: MLXArray?, state: MLXArray?, offset: Int)
}

public final class KVCache: LayerCache, @unchecked Sendable {
  public private(set) var keys: MLXArray?
  public private(set) var values: MLXArray?
  public private(set) var offset = 0
  public let step: Int

  public init(step: Int = 256) {
    self.step = step
  }

  public func reset() {
    keys = nil
    values = nil
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
  public private(set) var offset = 0

  public init() {}

  public func reset() {
    convState = nil
    recurrentState = nil
    offset = 0
  }

  public func snapshot() -> CacheSnapshot {
    .recurrent(conv: convState, state: recurrentState, offset: offset)
  }

  public func restore(_ snapshot: CacheSnapshot) {
    guard case .recurrent(let conv, let state, let restored) = snapshot else { return }
    convState = conv
    recurrentState = state
    offset = restored
  }

  public func advance(_ count: Int) {
    offset += count
  }
}

public final class ModelCache: @unchecked Sendable {
  public let layers: [LayerCache]

  public let kvConfig: KVCacheConfig

  public init(config: BonsaiConfig.TextConfig, kvConfig: KVCacheConfig = KVCacheConfig()) {
    self.kvConfig = kvConfig
    self.layers = config.isFullAttention.map { isFull in
      guard isFull else { return GatedDeltaNetCache() as LayerCache }
      return kvConfig.isQuantized
        ? QuantizedKVCache(config: kvConfig) as LayerCache
        : KVCache() as LayerCache
    }
  }

  public var byteCount: Int {
    layers.reduce(0) { $0 + (($1 as? AttentionKVCache)?.byteCount ?? 0) }
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
    layers.contains { $0 is GatedDeltaNetCache }
  }
}
