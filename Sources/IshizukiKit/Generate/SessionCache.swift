// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

public final class SessionCache: @unchecked Sendable {
  public struct Lease {
    public let cache: ModelCache
    public let reused: Int
    public let recycled: Bool
    fileprivate let slot: Slot
  }

  fileprivate final class Slot {
    var tokens: [Int]
    let cache: ModelCache
    var lastUsed: Date
    var busy = false

    init(tokens: [Int], cache: ModelCache) {
      self.tokens = tokens
      self.cache = cache
      self.lastUsed = Date()
    }
  }

  private let lock = NSLock()
  private var slots: [Slot] = []

  private var slotCapacity: Int
  public private(set) var lastReusedTokens = 0
  public private(set) var hits = 0
  public private(set) var misses = 0

  public init(capacity: Int = 1) {
    self.slotCapacity = max(1, capacity)
  }

  public var capacity: Int {
    lock.lock()
    defer { lock.unlock() }
    return slotCapacity
  }

  /// Raising it lets the next miss keep its prefix; lowering it drops the coldest idle slots.
  public func setCapacity(_ value: Int) {
    lock.lock()
    defer { lock.unlock() }
    slotCapacity = max(1, value)
    guard slots.count > slotCapacity else { return }
    var surplus = slots.count - slotCapacity
    var dropped: Set<ObjectIdentifier> = []
    for slot in slots.filter({ !$0.busy }).sorted(by: { $0.lastUsed < $1.lastUsed })
    where surplus > 0 {
      slot.cache.reset()
      dropped.insert(ObjectIdentifier(slot))
      surplus -= 1
    }
    slots.removeAll { dropped.contains(ObjectIdentifier($0)) }
  }

  public func prepare(
    for promptTokens: [Int], model: BonsaiModel, kvConfig: KVCacheConfig = KVCacheConfig()
  ) -> Lease {
    lock.lock()
    defer { lock.unlock() }

    var best: Slot?
    var bestReusable = 0
    for slot in slots where !slot.busy {
      let common = min(commonPrefixLength(slot.tokens, promptTokens), promptTokens.count - 1)
      guard common > 0, common == slot.tokens.count, slot.cache.offset == slot.tokens.count
      else { continue }
      if common > bestReusable {
        best = slot
        bestReusable = common
      }
    }

    if let best {
      best.tokens = promptTokens
      best.lastUsed = Date()
      best.busy = true
      lastReusedTokens = bestReusable
      hits += 1
      return Lease(cache: best.cache, reused: bestReusable, recycled: false, slot: best)
    }

    lastReusedTokens = 0
    misses += 1

    if let recycled = evictableSlot(matching: kvConfig) {
      recycled.cache.reset()
      recycled.tokens = promptTokens
      recycled.lastUsed = Date()
      recycled.busy = true
      return Lease(cache: recycled.cache, reused: 0, recycled: true, slot: recycled)
    }

    let slot = Slot(tokens: promptTokens, cache: model.text.makeCache(kvConfig: kvConfig))
    slot.busy = true
    slots.append(slot)
    return Lease(cache: slot.cache, reused: 0, recycled: false, slot: slot)
  }

  public func commit(_ lease: Lease, generated: [Int]) {
    lock.lock()
    defer { lock.unlock() }
    let slot = lease.slot
    slot.tokens.append(contentsOf: generated)
    if slot.tokens.count > slot.cache.offset {
      slot.tokens.removeLast(slot.tokens.count - slot.cache.offset)
    }
    slot.lastUsed = Date()
    slot.busy = false
  }

  public func release(_ lease: Lease) {
    lock.lock()
    defer { lock.unlock() }
    lease.slot.busy = false
  }

  public func evict() {
    lock.lock()
    defer { lock.unlock() }
    slots.removeAll { !$0.busy }
  }

  public func reset() {
    lock.lock()
    defer { lock.unlock() }
    for slot in slots where !slot.busy { slot.cache.reset() }
    slots.removeAll { !$0.busy }
    lastReusedTokens = 0
  }

  public var cachedTokenCount: Int {
    lock.lock()
    defer { lock.unlock() }
    return slots.reduce(0) { $0 + $1.tokens.count }
  }

  public var cachedBytes: Int {
    lock.lock()
    defer { lock.unlock() }
    return slots.reduce(0) { $0 + $1.cache.byteCount }
  }

  public var slotCount: Int {
    lock.lock()
    defer { lock.unlock() }
    return slots.count
  }

  private func evictableSlot(matching kvConfig: KVCacheConfig) -> Slot? {
    guard slots.count >= slotCapacity else { return nil }
    return
      slots
      .filter { !$0.busy && $0.cache.kvConfig == kvConfig }
      .min { $0.lastUsed < $1.lastUsed }
  }

  private func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
    var index = 0
    let limit = min(a.count, b.count)
    while index < limit, a[index] == b[index] { index += 1 }
    return index
  }
}
