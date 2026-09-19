// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

/// What the server is willing to hold, doubled a step at a time as the work asks for it.
///
/// Nothing is sized from the machine up front: a cold server commits to one short
/// conversation and a small buffer pool. Each time a request runs into the current tier -- a
/// prompt longer than the context reserve, a warm prefix evicted for want of a slot, a
/// saturated buffer pool -- that tier doubles, and stops at whatever the GPU wired ceiling
/// can still hold.
public final class MemoryBudget: @unchecked Sendable {
  public struct Tier: Sendable, Equatable {
    public var contextTokens: Int
    public var slots: Int

    /// What this tier has committed to the KV cache: the ceiling the prefix pool holds to.
    public func kvBytes(bytesPerToken: Int) -> Int { slots * contextTokens * bytesPerToken }
    public var bufferCache: Int
  }

  public struct Step: Sendable {
    public let summary: String
    public let tier: Tier
  }

  public static let contextFloor = 8_192
  public static let bufferFloor = 512 * 1_048_576
  public static let bufferCeiling = 8 * 1_073_741_824
  public static let slotCeiling = 8
  public static let workingReserve = 1_073_741_824
  public static let defaultWeights = 9_663_676_416
  public static let saturationsBeforeGrowth = 2

  public let ceiling: Int
  public let bytesPerToken: Int
  public let maxContextTokens: Int

  private let lock = NSLock()
  private let pinnedSlots: Int?
  private let pinnedBufferCache: Int?
  private var weights: Int
  private var contextTokens: Int
  private var slots: Int
  private var poolTier: Int
  private var saturations = 0

  public init(
    kvBits: Float?,
    maxContextTokens: Int,
    weights: Int = MemoryBudget.defaultWeights,
    ceiling: Int = ResidencyManager.gpuCeiling,
    slots: Int? = nil,
    bufferCache: Int? = nil
  ) {
    self.ceiling = max(ceiling, 1)
    self.bytesPerToken = Self.bytesPerToken(kvBits: kvBits)
    self.maxContextTokens = max(maxContextTokens, Self.contextFloor)
    self.weights = max(weights, 0)
    self.pinnedSlots = slots.map { max($0, 1) }
    self.pinnedBufferCache = bufferCache.map { max($0, 0) }
    self.contextTokens = min(Self.contextFloor, max(maxContextTokens, Self.contextFloor))
    self.slots = slots.map { max($0, 1) } ?? 1
    self.poolTier = bufferCache ?? Self.bufferFloor
  }

  public static func tokens(_ count: Int) -> String {
    if count >= 1_048_576 { return String(format: "%.1fM", Double(count) / 1_048_576) }
    return count < 1024 ? "\(count)" : "\(count / 1024)K"
  }

  public static func bytesPerToken(kvBits: Float?) -> Int {
    Int(32_768 * Double(kvBits ?? 16) / 8) + 2_048
  }

  /// Size of the weight files in a model pack, which land in GPU memory as they are mapped.
  public static func weightBytes(in directory: URL) -> Int? {
    let manager = FileManager.default
    guard
      let names = try? manager.contentsOfDirectory(atPath: directory.path)
    else { return nil }
    // HuggingFace's cache stores shards as symlinks into its blob store, and attributesOfItem
    // reports the link rather than what it points at — which read as a pack of no size at all.
    let total = names.filter { $0.hasSuffix(".safetensors") }.reduce(0) { sum, name in
      let url = directory.appending(path: name).resolvingSymlinksInPath()
      let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize
      return sum + (size ?? 0)
    }
    return total > 0 ? total : nil
  }

  public var tier: Tier {
    lock.lock()
    defer { lock.unlock() }
    return currentTier
  }

  public var weightBytes: Int {
    lock.lock()
    defer { lock.unlock() }
    return weights
  }

  public var headroom: Int {
    lock.lock()
    defer { lock.unlock() }
    return max(ceiling - weights - Self.workingReserve - kvCommitment - allowance, 0)
  }

  public var fitsFullContext: Bool {
    lock.lock()
    defer { lock.unlock() }
    return weights + Self.workingReserve + maxContextTokens * bytesPerToken <= ceiling
  }

  public var maxContextThatFits: Int {
    lock.lock()
    defer { lock.unlock() }
    let free = max(ceiling - weights - Self.workingReserve, 0)
    return min(free / max(bytesPerToken, 1), maxContextTokens)
  }

  public func apply() {
    let limit = tier.bufferCache
    guard limit > 0 else { return }
    Memory.cacheLimit = limit
  }

  public func observe(contextTokens demand: Int) -> Step? {
    lock.lock()
    defer { lock.unlock() }
    guard demand > contextTokens, contextTokens < maxContextTokens else { return nil }
    var next = contextTokens
    while next < demand, next < maxContextTokens { next *= 2 }
    next = min(next, maxContextTokens)
    guard next != contextTokens else { return nil }
    let previous = contextTokens
    contextTokens = next
    let shed = refit()
    return step("context reserve \(Self.tokens(previous)) → \(Self.tokens(next)) tok" + shed)
  }

  public func notePrefixEviction() -> Step? {
    lock.lock()
    defer { lock.unlock() }
    guard pinnedSlots == nil, slots < Self.slotCeiling else { return nil }
    let candidate = slots * 2
    guard fits(slots: candidate) else { return nil }
    let previous = slots
    slots = candidate
    return step("prefix slots \(previous) → \(slots)")
  }

  public func notePoolPressure(cacheMemory: Int) -> Step? {
    lock.lock()
    defer { lock.unlock() }
    guard pinnedBufferCache == nil else { return nil }
    let applied = allowance
    guard applied > 0, cacheMemory >= applied - applied / 10 else {
      saturations = 0
      return nil
    }
    saturations += 1
    guard saturations >= Self.saturationsBeforeGrowth else { return nil }
    saturations = 0
    guard poolTier < Self.bufferCeiling else { return nil }
    let previous = poolTier
    poolTier = min(poolTier * 2, Self.bufferCeiling)
    let grown = allowance
    guard grown > applied else {
      poolTier = previous
      return nil
    }
    return step("buffer pool \(gigabytes(applied)) → \(gigabytes(grown))")
  }

  public func reset() -> Step? {
    lock.lock()
    defer { lock.unlock() }
    let before = currentTier
    contextTokens = min(Self.contextFloor, maxContextTokens)
    if pinnedSlots == nil { slots = 1 }
    if pinnedBufferCache == nil { poolTier = Self.bufferFloor }
    saturations = 0
    guard currentTier != before else { return nil }
    return step("back to \(describeLocked())")
  }

  public func describe() -> String {
    lock.lock()
    defer { lock.unlock() }
    return describeLocked()
  }

  private var kvCommitment: Int { slots * contextTokens * bytesPerToken }

  /// The pool is worth no more than the KV it recycles, and no more than the room left.
  private var allowance: Int {
    if let pinnedBufferCache { return pinnedBufferCache }
    let free = max(ceiling - weights - Self.workingReserve - kvCommitment, 0)
    let earned = min(max(kvCommitment, Self.bufferFloor), Self.bufferCeiling)
    return min(max(min(poolTier, free), Self.bufferFloor), earned)
  }

  private var currentTier: Tier {
    Tier(contextTokens: contextTokens, slots: slots, bufferCache: allowance)
  }

  private func fits(slots candidate: Int) -> Bool {
    weights + Self.workingReserve + candidate * contextTokens * bytesPerToken + Self.bufferFloor
      <= ceiling
  }

  private func refit() -> String {
    guard pinnedSlots == nil else { return "" }
    let previous = slots
    while slots > 1, !fits(slots: slots) { slots /= 2 }
    return slots == previous ? "" : ", prefix slots \(previous) → \(slots)"
  }

  private func step(_ summary: String) -> Step {
    Step(summary: summary, tier: currentTier)
  }

  private func describeLocked() -> String {
    "\(slots) slot\(slots == 1 ? "" : "s") × \(Self.tokens(contextTokens)) tok, "
      + "\(gigabytes(allowance)) pool"
  }

  private func gigabytes(_ bytes: Int) -> String {
    bytes == 0 ? "unbounded" : String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
  }

}
