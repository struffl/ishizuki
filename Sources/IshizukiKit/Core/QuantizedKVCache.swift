// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public enum AttentionOperands {
  case dense(keys: MLXArray, values: MLXArray)
  case quantized(
    keys: (MLXArray, MLXArray, MLXArray),
    values: (MLXArray, MLXArray, MLXArray),
    groupSize: Int, keyBits: Int, valueBits: Int)
}

public protocol AttentionKVCache: LayerCache {
  func appendForAttention(keys: MLXArray, values: MLXArray) -> AttentionOperands
  var byteCount: Int { get }

  /// The indexer's own keys, one small head's worth per token, kept unnormalised and
  /// unrotated the way it wants to pool them. Only a layer that indexes writes here.
  var indexerKeys: MLXArray? { get set }
}

extension AttentionKVCache {
  /// Appends this chunk's indexer keys and hands back every key the query can see. Rewinding
  /// a cache moves its offset rather than truncating it, so what comes back is cut to the
  /// offset rather than to whatever the array happens to still hold.
  public func appendIndexerKeys(_ new: MLXArray, upTo length: Int) -> MLXArray {
    let all: MLXArray
    if let held = indexerKeys, held.dim(1) >= length - new.dim(1) {
      all = concatenated([held[0..., 0..<(length - new.dim(1)), 0...], new], axis: 1)
    } else {
      all = new
    }
    indexerKeys = all
    return all
  }
}

public struct KVCacheConfig: Sendable, Equatable {
  public var bits: Float?
  public var keyBits: Int
  public var valueBits: Int
  public var groupSize: Int
  public var residualWindow: Int

  /// 3.5 bits is the house default: keys at 3 and values at 4, behind a dense window wide
  /// enough that recent tokens are never read back through a quantizer. It costs a fifth of the
  /// memory a 16-bit cache does, which is most of what a long context costs on this hardware.
  public static let defaultBits: Float = 3.5

  public init(
    bits: Float? = defaultBits, groupSize: Int = 64, residualWindow: Int = 128
  ) {
    self.bits = bits
    self.groupSize = groupSize
    self.residualWindow = residualWindow
    (self.keyBits, self.valueBits) = Self.resolve(bits)
  }

  /// An unquantized cache, for measuring against.
  public static let full = KVCacheConfig(bits: nil)

  public static func resolve(_ bits: Float?) -> (key: Int, value: Int) {
    guard let bits else { return (16, 16) }
    let rounded = (bits * 2).rounded() / 2
    if rounded == rounded.rounded() {
      return (Int(rounded), Int(rounded))
    }
    return (Int(rounded.rounded(.down)), Int(rounded.rounded(.up)))
  }

  public var isQuantized: Bool { (bits ?? 16) < 16 }

  public static let supportedBits: Set<Int> = [2, 3, 4, 5, 6, 8]

  public func validate() throws {
    guard isQuantized else { return }
    guard Self.supportedBits.contains(keyBits), Self.supportedBits.contains(valueBits) else {
      throw BonsaiError.unsupportedModel(
        "kv-bits \(bits!) resolves to \(keyBits)-bit keys / \(valueBits)-bit values; "
          + "supported widths are \(Self.supportedBits.sorted())")
    }
  }
}

public final class QuantizedKVCache: AttentionKVCache, @unchecked Sendable {
  public let config: KVCacheConfig

  private var quantizedKeys: (MLXArray, MLXArray, MLXArray)?
  private var quantizedValues: (MLXArray, MLXArray, MLXArray)?
  private var quantizedCount = 0
  private var quantizedCapacity = 0
  private let drainBlock = 256

  private var windowKeys: MLXArray?
  private var windowValues: MLXArray?

  public var indexerKeys: MLXArray?

  public private(set) var offset = 0

  public init(config: KVCacheConfig) {
    self.config = config
  }

  public func reset() {
    quantizedKeys = nil
    quantizedValues = nil
    quantizedCount = 0
    quantizedCapacity = 0
    windowKeys = nil
    windowValues = nil
    indexerKeys = nil
    offset = 0
  }

  public func snapshot() -> CacheSnapshot { .kv(offset: offset) }

  /// Rewinding used to be limited to the dense window, and a target behind it was silently
  /// ignored — which left the caller believing in a rewind that never happened. Both stores are
  /// append-ordered, so a target behind the window just drops back into the quantized one.
  public func restore(_ snapshot: CacheSnapshot) {
    guard case .kv(let target) = snapshot, target < offset else { return }
    if target >= quantizedCount {
      let keepInWindow = target - quantizedCount
      windowKeys = windowKeys.map { $0[0..., 0..., ..<keepInWindow, 0...] }
      windowValues = windowValues.map { $0[0..., 0..., ..<keepInWindow, 0...] }
    } else {
      // The window holds only tokens after the quantized store, so rewinding behind it leaves
      // nothing to keep. The store itself is truncated by count; its capacity is reused.
      quantizedCount = target
      windowKeys = nil
      windowValues = nil
    }
    offset = target
  }

  public func appendForAttention(keys newKeys: MLXArray, values newValues: MLXArray)
    -> AttentionOperands
  {
    windowKeys =
      windowKeys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
    windowValues =
      windowValues.map { concatenated([$0, newValues], axis: 2) } ?? newValues
    offset += newKeys.dim(2)

    if let keys = windowKeys, keys.dim(2) >= config.residualWindow + drainBlock {
      compress(count: drainBlock)
    }

    guard quantizedCount > 0, let quantizedKeys, let quantizedValues else {
      return .dense(keys: windowKeys!, values: windowValues!)
    }
    func live(_ store: (MLXArray, MLXArray, MLXArray)) -> (MLXArray, MLXArray, MLXArray) {
      (
        store.0[0..., 0..., ..<quantizedCount, 0...],
        store.1[0..., 0..., ..<quantizedCount, 0...],
        store.2[0..., 0..., ..<quantizedCount, 0...]
      )
    }
    return .quantized(
      keys: live(quantizedKeys), values: live(quantizedValues),
      groupSize: config.groupSize, keyBits: config.keyBits, valueBits: config.valueBits)
  }

  public var window: (keys: MLXArray, values: MLXArray)? {
    guard let windowKeys, let windowValues, windowKeys.dim(2) > 0 else { return nil }
    return (windowKeys, windowValues)
  }

  public var quantizedTokenCount: Int { quantizedCount }

  private func compress(count: Int) {
    guard let keys = windowKeys, let values = windowValues, count > 0 else { return }
    let headKeys = keys[0..., 0..., ..<count, 0...]
    let headValues = values[0..., 0..., ..<count, 0...]

    let newKeys = quantized(
      headKeys, groupSize: config.groupSize, bits: config.keyBits, mode: .affine)
    let newValues = quantized(
      headValues, groupSize: config.groupSize, bits: config.valueBits, mode: .affine)

    reserveStore(for: quantizedCount + count, keys: newKeys, values: newValues)
    write(&quantizedKeys!, newKeys, at: quantizedCount)
    write(&quantizedValues!, newValues, at: quantizedCount)
    quantizedCount += count

    windowKeys = keys[0..., 0..., count..., 0...]
    windowValues = values[0..., 0..., count..., 0...]
  }

  private func reserveStore(
    for required: Int,
    keys: (wq: MLXArray, scales: MLXArray, biases: MLXArray?),
    values: (wq: MLXArray, scales: MLXArray, biases: MLXArray?)
  ) {
    guard required > quantizedCapacity else { return }
    var capacity = max(quantizedCapacity, drainBlock)
    while capacity < required { capacity += max(capacity / 2, drainBlock) }

    func allocate(
      _ template: (wq: MLXArray, scales: MLXArray, biases: MLXArray?),
      existing: (MLXArray, MLXArray, MLXArray)?
    ) -> (MLXArray, MLXArray, MLXArray) {
      func sized(_ array: MLXArray) -> MLXArray {
        var shape = array.shape
        shape[2] = capacity
        return MLXArray.zeros(shape, dtype: array.dtype)
      }
      let biasTemplate = template.biases ?? MLXArray.zeros(like: template.scales)
      let fresh = (sized(template.wq), sized(template.scales), sized(biasTemplate))
      if let existing, quantizedCount > 0 {
        fresh.0[0..., 0..., ..<quantizedCount, 0...] =
          existing.0[0..., 0..., ..<quantizedCount, 0...]
        fresh.1[0..., 0..., ..<quantizedCount, 0...] =
          existing.1[0..., 0..., ..<quantizedCount, 0...]
        fresh.2[0..., 0..., ..<quantizedCount, 0...] =
          existing.2[0..., 0..., ..<quantizedCount, 0...]
      }
      return fresh
    }

    quantizedKeys = allocate(keys, existing: quantizedKeys)
    quantizedValues = allocate(values, existing: quantizedValues)
    quantizedCapacity = capacity
  }

  private func write(
    _ store: inout (MLXArray, MLXArray, MLXArray),
    _ addition: (wq: MLXArray, scales: MLXArray, biases: MLXArray?),
    at index: Int
  ) {
    let count = addition.wq.dim(2)
    store.0[0..., 0..., index..<(index + count), 0...] = addition.wq
    store.1[0..., 0..., index..<(index + count), 0...] = addition.scales
    store.2[0..., 0..., index..<(index + count), 0...] =
      addition.biases ?? MLXArray.zeros(like: addition.scales)
  }

  public func export() -> [String: MLXArray]? {
    guard offset > 0 else { return nil }
    var arrays: [String: MLXArray] = [:]
    if quantizedCount > 0, let quantizedKeys, let quantizedValues {
      let live = ..<quantizedCount
      arrays["qk.w"] = quantizedKeys.0[0..., 0..., live, 0...]
      arrays["qk.s"] = quantizedKeys.1[0..., 0..., live, 0...]
      arrays["qk.b"] = quantizedKeys.2[0..., 0..., live, 0...]
      arrays["qv.w"] = quantizedValues.0[0..., 0..., live, 0...]
      arrays["qv.s"] = quantizedValues.1[0..., 0..., live, 0...]
      arrays["qv.b"] = quantizedValues.2[0..., 0..., live, 0...]
    }
    if let windowKeys, let windowValues, windowKeys.dim(2) > 0 {
      arrays["wk"] = windowKeys
      arrays["wv"] = windowValues
    }
    if let indexerKeys { arrays["ik"] = indexerKeys[0..., 0..<offset, 0...] }
    return arrays.isEmpty ? nil : arrays
  }

  public func load(_ arrays: [String: MLXArray], offset: Int) -> Bool {
    var restoredQuantized = 0
    var keys: (MLXArray, MLXArray, MLXArray)?
    var values: (MLXArray, MLXArray, MLXArray)?
    if let w = arrays["qk.w"], let sc = arrays["qk.s"], let b = arrays["qk.b"],
      let vw = arrays["qv.w"], let vs = arrays["qv.s"], let vb = arrays["qv.b"]
    {
      keys = (w, sc, b)
      values = (vw, vs, vb)
      restoredQuantized = w.dim(2)
    }

    let window: (MLXArray, MLXArray)?
    if let wk = arrays["wk"], let wv = arrays["wv"] {
      window = (wk, wv)
    } else {
      window = nil
    }

    // The two stores have to account for exactly the tokens the archive claims.
    guard restoredQuantized + (window?.0.dim(2) ?? 0) == offset else { return false }

    quantizedKeys = keys
    quantizedValues = values
    quantizedCount = restoredQuantized
    quantizedCapacity = restoredQuantized
    windowKeys = window?.0
    windowValues = window?.1
    indexerKeys = arrays["ik"]
    self.offset = offset
    return true
  }

  public var byteCount: Int {
    var total = 0
    for array in [quantizedKeys, quantizedValues].compactMap({ $0 }) {
      let fraction = quantizedCapacity > 0 ? Double(quantizedCount) / Double(quantizedCapacity) : 0
      total += Int(Double(array.0.nbytes + array.1.nbytes + array.2.nbytes) * fraction)
    }
    total += windowKeys?.nbytes ?? 0
    total += windowValues?.nbytes ?? 0
    return total
  }
}

extension KVCache: AttentionKVCache {
  public func appendForAttention(keys newKeys: MLXArray, values newValues: MLXArray)
    -> AttentionOperands
  {
    let (k, v) = update(keys: newKeys, values: newValues)
    return .dense(keys: k, values: v)
  }

  public var byteCount: Int { (keys?.nbytes ?? 0) + (values?.nbytes ?? 0) }
}

extension MLXArray {
  var nbytes: Int { size * itemSize }
}
