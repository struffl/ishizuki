// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import MLX
import Testing

@testable import IshizukiKit

@Suite("Session cache")
struct SessionCacheTests {
  // A hybrid schedule, so the recurrent layers that make checkpoints cost something are in play.
  private let schedule = [false, false, false, true]
  private let kv = KVCacheConfig(bits: 3.5, residualWindow: 8)

  private func pool(capacity: Int = 2, checkpoints: Int = 2) -> SessionCache {
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
    run(lease, to: prompt.count + reply.count)
    pool.commit(lease, generated: reply)
    return lease
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
