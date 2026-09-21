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
  /// What the last lookup saw, so a miss can be read rather than guessed at: the slots that
  /// were there, how much of each agreed with the prompt, and where their rewind points were.
  public private(set) var lastTrace = ""
  /// Rewind points that were asked for and refused, which is silent otherwise.
  public private(set) var refusedCheckpoints = 0
  /// And ones that were taken, so a lookup that finds none can tell a point never taken from
  /// one taken and then shed to stay inside the budget.
  public private(set) var takenCheckpoints = 0
  public private(set) var shedCheckpoints = 0
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
  public init(capacity: Int = 1, checkpoints: Int = 4) {
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

    // Thin before shedding, and keep the lowest point rather than the newest. The useful one
    // is the highest point at or below where the next prompt parts ways, and since a prompt
    // parts ways near its own end, that is the earliest of a turn's points — not the one
    // taken after the reply, which is past the parting and can never be restored.
    for slot in coldestFirst where slot.checkpoints.count > 1 {
      guard total() > byteLimit else { return }
      let kept = slot.checkpoints.first!
      shedCheckpoints += slot.checkpoints.count - 1
      slot.checkpoints = [kept]
    }

    for slot in coldestFirst where !slot.checkpoints.isEmpty {
      guard total() > byteLimit else { return }
      shedCheckpoints += slot.checkpoints.count
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
    var trace: [String] = []
    for slot in slots {
      trace.append(
        "[held \(slot.tokens.count)"
          + (slot.busy ? " busy" : "")
          + (slot.cache.kvConfig == kvConfig ? "" : " other-kv")
          + " offset \(slot.cache.offset)"
          + " agrees \(min(commonPrefixLength(slot.tokens, promptTokens), promptTokens.count - 1))"
          + " points \(slot.checkpoints.map(\.tokens))]")
    }
    lastTrace =
      trace.isEmpty
      ? "no slots"
      : trace.joined(separator: " ")
        + " taken \(takenCheckpoints) shed \(shedCheckpoints) refused \(refusedCheckpoints)"

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
  private func checkpoint(_ slot: Slot, at count: Int? = nil) {
    let position = count ?? slot.tokens.count
    guard checkpointLimit > 0, position > 0,
      position == slot.cache.offset,
      slot.checkpoints.last?.tokens != position
    else {
      // A point asked for and not taken is the difference between a cache that cannot help
      // and a cache that was never given the chance.
      if checkpointLimit > 0, position > 0, position != slot.cache.offset {
        refusedCheckpoints += 1
      }
      return
    }

    takenCheckpoints += 1
    let state = slot.cache.snapshot()
    slot.checkpoints.append(
      Checkpoint(
        tokens: position, state: state,
        byteCount: state.reduce(0) { $0 + $1.byteCount }))
    if slot.checkpoints.count > checkpointLimit {
      slot.checkpoints.removeFirst(slot.checkpoints.count - checkpointLimit)
    }
  }

  /// Takes a rewind point at the prompt boundary, before any reply is decoded into the slot.
  /// This is the one that matters across turns: a harness re-renders the assistant message
  /// rather than replaying the tokens that were sampled, so the next prompt usually parts ways
  /// right here, and without a checkpoint at this length the whole prefix is re-prefilled.
  public func checkpointPrompt(_ lease: Lease) {
    lock.lock()
    defer { lock.unlock() }
    checkpoint(lease.slot)
  }

  /// A rewind point taken mid-prefill, strictly before the prompt ends.
  ///
  /// The prompt-boundary point is one token too late to be used. A rendered prompt ends with
  /// the generation prompt — for a thinking template, the opener the model is meant to continue
  /// from — and on the next turn that position holds the reply's first token instead. So the
  /// two prompts agree on everything but the last token, and a rewind point at the boundary is
  /// above the parting and cannot be restored. This one is below it, so there is always
  /// somewhere to rewind to.
  public func checkpointPrefill(_ lease: Lease, at count: Int) {
    lock.lock()
    defer { lock.unlock() }
    checkpoint(lease.slot, at: count)
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
