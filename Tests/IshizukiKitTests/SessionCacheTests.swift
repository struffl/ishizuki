// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

@Suite("Session cache")
struct SessionCacheTests {
  // A hybrid schedule, so the recurrent layers that make checkpoints cost something are in play.
  private let schedule = [false, false, false, true]
  private let kv = KVCacheConfig(bits: 3.5, residualWindow: 8)

  private func pool(capacity: Int = 2, checkpoints: Int = 4) -> SessionCache {
    SessionCache(capacity: capacity, checkpoints: checkpoints)
  }

  private func lease(_ cache: SessionCache, _ tokens: [Int]) -> SessionCache.Lease {
    cache.prepare(for: tokens, kvConfig: kv) {
      ModelCache(fullAttention: schedule, kvConfig: kv)
    }
  }

  /// Drives a lease's cache to `total` tokens the way a real turn would — prompt and generated
  /// alike, since commit() trims a slot back to whatever its cache actually holds.
  private func run(_ lease: SessionCache.Lease, to total: Int) {
    let heads = 2
    let dim = 16
    for layer in lease.cache.layers {
      let count = total - layer.offset
      guard count > 0 else { continue }
      switch layer {
      case let attention as AttentionKVCache:
        let k = MLXArray.zeros([1, heads, count, dim], dtype: .float16)
        _ = attention.appendForAttention(keys: k, values: k)
      case let recurrent as GatedDeltaNetCache:
        recurrent.recurrentState = MLXArray.zeros([1, heads, dim, dim], dtype: .float32)
        recurrent.convState = MLXArray.zeros([1, 3, dim], dtype: .float16)
        recurrent.advance(count)
      default: break
      }
    }
  }

  /// One complete turn: lease the slot, run the cache over prompt plus reply, commit.
  @discardableResult
  private func turn(
    _ pool: SessionCache, prompt: [Int], reply: [Int] = []
  ) -> SessionCache.Lease {
    let lease = lease(pool, prompt)
    run(lease, to: prompt.count)
    pool.checkpointPrompt(lease)
    run(lease, to: prompt.count + reply.count)
    pool.commit(lease, generated: reply)
    return lease
  }

  @Test("readout uses published bytes while a lease changes its arrays")
  func readoutDoesNotInspectBusyArrays() {
    let pool = pool(checkpoints: 0)
    let active = lease(pool, [1, 2, 3, 4])
    let before = pool.cachedBytes
    run(active, to: 4)
    #expect(active.cache.byteCount > before)
    // Neither dashboard reads nor budget changes may touch the in-flight arrays.
    #expect(pool.cachedBytes == before)
    pool.setByteLimit(1)
    #expect(pool.slotCount == 1)
    pool.setByteLimit(0)
    pool.release(active)
    #expect(pool.cachedBytes == active.cache.byteCount)
    pool.evict()
    #expect(pool.cachedBytes == 0)
  }

  /// The case an agentic harness lives in: a rendered prompt ends with the generation prompt,
  /// and next turn that position holds the reply's first token instead, so the two prompts
  /// agree on everything but the last token. A rewind point at the prompt boundary is one
  /// token above the parting and cannot be used.
  @Test("a prompt that parts ways one token early still reuses the prefix")
  func partsWaysAtTheGenerationPrompt() {
    let pool = pool()
    let prompt = Array(1...64)
    let reply = Array(200..<210)

    let first = lease(pool, prompt)
    run(first, to: 48)
    pool.checkpointPrefill(first, at: 48)
    run(first, to: prompt.count)
    pool.checkpointPrompt(first)
    run(first, to: prompt.count + reply.count)
    pool.commit(first, generated: reply)

    // Everything but the last token, which is where the generation prompt was.
    var next = Array(prompt.dropLast())
    next.append(999)
    next += Array(300..<320)

    let second = lease(pool, next)
    #expect(second.reused == 48)
    #expect(second.branched)
  }

  @Test("without a point below the parting there is nothing to rewind to")
  func nothingBelowTheParting() {
    let pool = pool()
    let prompt = Array(1...64)

    let first = lease(pool, prompt)
    run(first, to: prompt.count)
    pool.checkpointPrompt(first)
    pool.commit(first, generated: [])

    var next = Array(prompt.dropLast())
    next.append(999)

    let second = lease(pool, next)
    #expect(second.reused == 0)
  }

  @Test("a prompt that continues a cached one reuses all of it")
  func continuation() {
    let pool = pool()
    turn(pool, prompt: [1, 2, 3, 4], reply: [5, 6])

    let second = lease(pool, [1, 2, 3, 4, 5, 6, 7, 8])
    #expect(second.reused == 6)
    #expect(second.branched == false)
    #expect(pool.slotCount == 1)
  }

  @Test("a prompt that branches rewinds to the turn it shares")
  func branching() {
    let pool = pool()
    turn(pool, prompt: [1, 2, 3, 4], reply: [5, 6])
    turn(pool, prompt: [1, 2, 3, 4, 5, 6, 7], reply: [8, 9])

    // Now the last turn is edited: shared up to token 6, different after.
    let branched = lease(pool, [1, 2, 3, 4, 5, 6, 99, 100])
    #expect(branched.branched == true)
    #expect(branched.reused == 6)
    #expect(branched.cache.offset == 6)
    #expect(pool.slotCount == 1)
    #expect(pool.branches == 1)
  }

  @Test("a re-rendered reply still reuses the prompt behind it")
  func rerenderedReply() {
    let pool = pool()
    turn(pool, prompt: [1, 2, 3, 4], reply: [5, 6])

    // What a harness sends next: the same prompt, then its own rendering of the reply, which
    // is not the tokens that were sampled.
    let second = lease(pool, [1, 2, 3, 4, 50, 60, 7, 8])
    #expect(second.reused == 4)
    #expect(second.branched == true)
  }

  @Test("a branch behind every checkpoint is a miss rather than a bad rewind")
  func branchTooEarly() {
    let pool = pool()
    turn(pool, prompt: [1, 2, 3, 4], reply: [5, 6])

    // Diverges at token 2, earlier than the only checkpoint at 6.
    let branched = lease(pool, [1, 2, 77, 78])
    #expect(branched.reused == 0)
    #expect(branched.branched == false)
  }

  @Test("checkpoints are capped, and the oldest go first")
  func checkpointsCapped() {
    let pool = pool(capacity: 1, checkpoints: 2)
    var tokens = [1, 2]
    turn(pool, prompt: tokens)
    for extra in [[3], [4], [5]] {
      tokens += extra
      turn(pool, prompt: tokens)
    }

    // Four turns, two rewind points kept: the earliest is gone, so a branch there misses.
    let early = lease(pool, [1, 2, 90])
    #expect(early.reused == 0)
  }

  @Test("a byte ceiling sheds rewind points before it sheds a prefix")
  func shedsCheckpointsFirst() {
    let pool = pool(capacity: 4)
    turn(pool, prompt: [1, 2, 3, 4], reply: [5, 6])
    let held = pool.cachedBytes
    let checkpoints = pool.checkpointBytes
    #expect(checkpoints > 0)

    // A ceiling just under what is held: enough to force shedding, not enough to need the slot.
    pool.setByteLimit(held - checkpoints)
    #expect(pool.checkpointBytes == 0)
    #expect(pool.slotCount == 1)
    #expect(pool.evictions == 0)
  }

  @Test("a ceiling the prefixes alone overrun evicts the coldest of them")
  func evictsColdest() {
    let pool = pool(capacity: 4)
    turn(pool, prompt: [1, 2, 3, 4], reply: [5, 6])
    turn(pool, prompt: [50, 51, 52, 53], reply: [54, 55])
    #expect(pool.slotCount == 2)

    pool.setByteLimit(1)
    #expect(pool.slotCount == 0)
    #expect(pool.evictions == 2)
    #expect(pool.cachedBytes == 0)
  }

  @Test("a busy slot is never evicted out from under its request")
  func busyIsSafe() {
    let pool = pool(capacity: 4)
    turn(pool, prompt: [1, 2, 3, 4], reply: [5, 6])
    let inFlight = lease(pool, [90, 91, 92])
    run(inFlight, to: 3)

    pool.setByteLimit(1)
    #expect(pool.slotCount == 1)
    #expect(inFlight.cache.offset == 3)
  }

  @Test("a prefix that fell out of memory comes back from disk")
  func diskTier() {
    let dir = URL(filePath: NSTemporaryDirectory())
      .appending(path: "session-disk-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }

    let store = PrefixStore(directory: dir, minimumTokens: 4)
    let pool = pool(capacity: 1)
    pool.setStore(store, modelID: "pack-a")

    let prompt = Array(1...32)
    turn(pool, prompt: prompt)
    pool.persistAll()
    #expect(store.entries().count == 1)

    // Everything in memory is gone, as after an idle unload.
    pool.reset()
    #expect(pool.slotCount == 0)

    let revived = lease(pool, prompt + [99])
    #expect(revived.reused == 32)
    #expect(revived.cache.offset == 32)
    #expect(pool.diskHits == 1)
  }

  @Test("a disk prefix from another pack is not read into this one")
  func diskFingerprint() {
    let dir = URL(filePath: NSTemporaryDirectory())
      .appending(path: "session-disk-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: dir) }

    let store = PrefixStore(directory: dir, minimumTokens: 4)
    let writer = pool(capacity: 1)
    writer.setStore(store, modelID: "pack-a")
    let prompt = Array(1...32)
    turn(writer, prompt: prompt)
    writer.persistAll()

    let reader = pool(capacity: 1)
    reader.setStore(store, modelID: "pack-b")
    let lease = lease(reader, prompt + [99])
    #expect(lease.reused == 0)
    #expect(reader.diskHits == 0)
  }

  @Test("checkpoints are counted against the cache's own memory")
  func checkpointsCost() {
    let pool = pool()
    #expect(pool.checkpointBytes == 0)
    turn(pool, prompt: [1, 2, 3, 4])
    // The recurrent layers' state is what a rewind point has to hold on to.
    #expect(pool.checkpointBytes > 0)
    #expect(pool.cachedBytes >= pool.checkpointBytes)
  }
}
