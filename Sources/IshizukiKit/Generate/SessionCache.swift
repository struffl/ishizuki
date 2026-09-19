// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation

public final class SessionCache: @unchecked Sendable {
  public struct Lease {
    public let cache: ModelCache
    public let reused: Int
    public let recycled: Bool
    /// True when the prompt diverged from the cached one and the slot was rewound to a
    /// checkpoint rather than continued.
    public let branched: Bool
    fileprivate let slot: Slot
  }

  /// A point a slot can be rewound to: the cache's whole state after some number of tokens.
  /// Attention layers snapshot an offset and cost nothing; a recurrent layer has to keep its
  /// state, so on a hybrid model these are worth real memory and are kept few.
  fileprivate struct Checkpoint {
    let tokens: Int
    let state: [CacheSnapshot]
    let byteCount: Int
  }

  fileprivate final class Slot {
    var tokens: [Int]
    let cache: ModelCache
    var lastUsed: Date
    var busy = false
    var checkpoints: [Checkpoint] = []

    init(tokens: [Int], cache: ModelCache) {
      self.tokens = tokens
      self.cache = cache
      self.lastUsed = Date()
    }

    var checkpointBytes: Int { checkpoints.reduce(0) { $0 + $1.byteCount } }

    func clearCheckpoints() { checkpoints.removeAll() }
  }

  private let lock = NSLock()
  private var slots: [Slot] = []

  private var slotCapacity: Int
  private let checkpointLimit: Int
  public private(set) var lastReusedTokens = 0
  public private(set) var hits = 0
  public private(set) var misses = 0
  public private(set) var branches = 0
  public private(set) var evictions = 0
  private var byteLimit = 0
  private var store: PrefixStore?
  private var modelID = ""
  public private(set) var diskHits = 0

  /// Backs the pool with a disk tier. `modelID` identifies the pack, so an archive is never
  /// read into a model it was not built from.
  public func setStore(_ store: PrefixStore?, modelID: String) {
    lock.lock()
    defer { lock.unlock() }
    self.store = store
    self.modelID = modelID
  }

  /// Writes every idle slot worth keeping to the disk tier. Called before the pool is dropped —
  /// on idle unload or shutdown — rather than on every eviction, because archiving a long
  /// prefix costs real time and should not land in the middle of a request.
  public func persistAll() {
    lock.lock()
    let store = self.store
    let modelID = self.modelID
    let pending = slots.filter { !$0.busy && $0.tokens.count == $0.cache.offset }
      .map { ($0.cache, $0.tokens, $0.cache.kvConfig) }
    lock.unlock()

    guard let store else { return }
    for (cache, tokens, kvConfig) in pending {
      store.save(cache: cache, tokens: tokens, modelID: modelID, kvConfig: kvConfig)
    }
    store.evictToLimit()
  }

  /// `checkpoints` is how many rewind points each slot keeps. On a hybrid model each one holds
  /// every recurrent layer's state, so the default is deliberately small: enough to rewind the
  /// last couple of turns, not enough to outweigh the cache it serves.
  public init(capacity: Int = 1, checkpoints: Int = 2) {
    self.slotCapacity = max(1, capacity)
    self.checkpointLimit = max(0, checkpoints)
  }

  public var capacity: Int {
    lock.lock()
    defer { lock.unlock() }
    return slotCapacity
  }

  /// The ceiling the pool holds itself to, in bytes; 0 leaves it bounded only by slot count.
  /// A slot count is a poor proxy on its own — one 128k prefix outweighs twenty short ones —
  /// so this is what actually keeps the cache inside the budget it was given.
  public func setByteLimit(_ bytes: Int) {
    lock.lock()
    defer { lock.unlock() }
    byteLimit = max(0, bytes)
    enforceByteLimit()
  }

  public var byteLimitBytes: Int {
    lock.lock()
    defer { lock.unlock() }
    return byteLimit
  }

  /// Sheds the cheapest thing first: rewind points are an optimization on top of a held prefix,
  /// so they go before any prefix does. Called with the lock held.
  private func enforceByteLimit() {
    guard byteLimit > 0 else { return }
    func total() -> Int { slots.reduce(0) { $0 + $1.cache.byteCount + $1.checkpointBytes } }
    guard total() > byteLimit else { return }

    let coldestFirst = slots.filter { !$0.busy }.sorted { $0.lastUsed < $1.lastUsed }

    for slot in coldestFirst where !slot.checkpoints.isEmpty {
      guard total() > byteLimit else { return }
      slot.clearCheckpoints()
    }

    var dropped: Set<ObjectIdentifier> = []
    for slot in coldestFirst {
      guard total() > byteLimit else { break }
      slot.cache.reset()
      slot.clearCheckpoints()
      dropped.insert(ObjectIdentifier(slot))
      evictions += 1
    }
    slots.removeAll { dropped.contains(ObjectIdentifier($0)) }
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
    prepare(for: promptTokens, kvConfig: kvConfig) {
      model.text.makeCache(kvConfig: kvConfig)
    }
  }

  /// The allocating form. A slot's cache is only ever built here, so a caller that has a model
  /// and a test that has a bare schedule reach the same pooling.
  public func prepare(
    for promptTokens: [Int], kvConfig: KVCacheConfig = KVCacheConfig(),
    makeCache: () -> ModelCache
  ) -> Lease {
    lock.lock()
    defer { lock.unlock() }

    // A slot is reusable two ways: the prompt continues it, or the prompt branches off it and
    // the slot can be rewound to a checkpoint at or before where they part.
    var best: (slot: Slot, reuse: Int, rewind: Checkpoint?)?
    for slot in slots where !slot.busy && slot.cache.kvConfig == kvConfig {
      let common = min(commonPrefixLength(slot.tokens, promptTokens), promptTokens.count - 1)
      guard common > 0 else { continue }

      var candidate: (Int, Checkpoint?)?
      if common == slot.tokens.count, slot.cache.offset == slot.tokens.count {
        candidate = (common, nil)
      } else if let checkpoint = slot.checkpoints.last(where: { $0.tokens <= common }) {
        candidate = (checkpoint.tokens, checkpoint)
      }

      guard let candidate, candidate.0 > (best?.reuse ?? 0) else { continue }
      best = (slot, candidate.0, candidate.1)
    }

    if let best {
      if let rewind = best.rewind {
        best.slot.cache.restore(rewind.state)
        // Anything past the branch point describes a history this slot no longer has.
        best.slot.checkpoints.removeAll { $0.tokens > rewind.tokens }
        branches += 1
      }
      best.slot.tokens = promptTokens
      best.slot.lastUsed = Date()
      best.slot.busy = true
      lastReusedTokens = best.reuse
      hits += 1
      return Lease(
        cache: best.slot.cache, reused: best.reuse, recycled: false,
        branched: best.rewind != nil, slot: best.slot)
    }

    lastReusedTokens = 0
    misses += 1

    let slot: Slot
    let recycled: Bool
    if let reusable = evictableSlot(matching: kvConfig) {
      reusable.cache.reset()
      reusable.clearCheckpoints()
      slot = reusable
      recycled = true
    } else {
      slot = Slot(tokens: promptTokens, cache: makeCache())
      slots.append(slot)
      recycled = false
    }

    // Nothing in memory continues this prompt; the disk tier may still hold a prefix of it.
    var fromDisk = 0
    if let store,
      let entry = store.bestMatch(for: promptTokens, modelID: modelID, kvConfig: kvConfig),
      store.load(entry, into: slot.cache)
    {
      fromDisk = entry.tokens.count
      lastReusedTokens = fromDisk
      diskHits += 1
    }

    slot.tokens = promptTokens
    slot.lastUsed = Date()
    slot.busy = true
    return Lease(
      cache: slot.cache, reused: fromDisk, recycled: recycled, branched: false, slot: slot)
  }

  public func commit(_ lease: Lease, generated: [Int]) {
    lock.lock()
    defer { lock.unlock() }
    let slot = lease.slot
    slot.tokens.append(contentsOf: generated)
    if slot.tokens.count > slot.cache.offset {
      slot.tokens.removeLast(slot.tokens.count - slot.cache.offset)
    }
    checkpoint(slot)
    slot.lastUsed = Date()
    slot.busy = false
    enforceByteLimit()
  }

  /// A turn boundary is where conversations branch — a retried, edited or forked last message
  /// shares everything before it — so that is where the rewind points are taken.
  private func checkpoint(_ slot: Slot) {
    guard checkpointLimit > 0, slot.tokens.count > 0,
      slot.tokens.count == slot.cache.offset,
      slot.checkpoints.last?.tokens != slot.tokens.count
    else { return }

    let state = slot.cache.snapshot()
    slot.checkpoints.append(
      Checkpoint(
        tokens: slot.tokens.count, state: state,
        byteCount: state.reduce(0) { $0 + $1.byteCount }))
    if slot.checkpoints.count > checkpointLimit {
      slot.checkpoints.removeFirst(slot.checkpoints.count - checkpointLimit)
    }
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
    for slot in slots where !slot.busy {
      slot.cache.reset()
      slot.clearCheckpoints()
    }
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
    return slots.reduce(0) { $0 + $1.cache.byteCount + $1.checkpointBytes }
  }

  /// What the rewind points cost on their own, which is the part that is bought rather than
  /// inherent to holding the prefix.
  public var checkpointBytes: Int {
    lock.lock()
    defer { lock.unlock() }
    return slots.reduce(0) { $0 + $1.checkpointBytes }
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
