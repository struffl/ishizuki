// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import Testing

@testable import IshizukiKit

// Seeds a cache directory so the browser can be looked at. Off unless ISHIZUKI_SEED is set.
@Suite("Seed", .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_SEED"] != nil))
struct SeedFixture {
  @Test("seed")
  func seed() throws {
    let dir = URL(filePath: ProcessInfo.processInfo.environment["ISHIZUKI_SEED"]!)
    let store = PrefixStore(directory: dir, minimumTokens: 4)
    let kv = KVCacheConfig(bits: 3.5, residualWindow: 8)
    for (index, count) in [64, 512, 4096].enumerated() {
      let cache = ModelCache(fullAttention: [false, true], kvConfig: kv)
      for layer in cache.layers {
        switch layer {
        case let a as AttentionKVCache:
          let k = MLXArray.zeros([1, 2, count, 64], dtype: .float16)
          _ = a.appendForAttention(keys: k, values: k)
        case let r as GatedDeltaNetCache:
          r.recurrentState = MLXArray.zeros([1, 2, 64, 64], dtype: .float32)
          r.advance(count)
        default: break
        }
      }
      store.save(
        cache: cache, tokens: Array((index * 10_000)..<(index * 10_000 + count)),
        modelID: "pack-a", kvConfig: kv)
    }
  }
}
