// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import MLX
import Testing

@testable import IshizukiKit

@Suite("Quantized KV cache")
struct QuantizedKVTests {
  private func step(_ cache: QuantizedKVCache, tokens: Int, heads: Int = 4, dim: Int = 256) {
    let keys = MLXArray.zeros([1, heads, tokens, dim], dtype: .float16)
    let values = MLXArray.zeros([1, heads, tokens, dim], dtype: .float16)
    let operands = cache.appendForAttention(keys: keys, values: values)
    switch operands {
    case .dense(let k, let v):
      eval(k, v)
    case .quantized(let k, let v, _, _, _):
      eval(k.0, k.1, k.2, v.0, v.1, v.2)
    }
  }

  @Test("the window and the quantized store always account for every token")
  func invariantHolds() {
    let cache = QuantizedKVCache(config: KVCacheConfig(bits: 3.5, residualWindow: 128))
    var total = 0
    for _ in 0..<40 {
      step(cache, tokens: 128)
      total += 128
      #expect(cache.offset == total)
      let windowed = cache.window?.keys.dim(2) ?? 0
      #expect(cache.quantizedTokenCount + windowed == total)
    }
  }

  @Test("a rewind behind the dense window drops back into the quantized store")
  func rewindPastWindow() {
    let cache = QuantizedKVCache(config: KVCacheConfig(bits: 3.5, residualWindow: 128))
    for _ in 0..<16 { step(cache, tokens: 128) }
    #expect(cache.offset == 2048)
    // Everything but the tail has been compressed by now, so this target is well behind it.
    #expect(cache.quantizedTokenCount > 512)

    cache.restore(.kv(offset: 512))
    #expect(cache.offset == 512)
    #expect(cache.quantizedTokenCount == 512)
    #expect(cache.window == nil)

    // The cache has to keep taking tokens afterwards, and keep accounting for them.
    step(cache, tokens: 64)
    #expect(cache.offset == 576)
    #expect(cache.quantizedTokenCount + (cache.window?.keys.dim(2) ?? 0) == 576)
  }

  @Test("a rewind inside the dense window only trims the window")
  func rewindInsideWindow() {
    let cache = QuantizedKVCache(config: KVCacheConfig(bits: 3.5, residualWindow: 128))
    for _ in 0..<16 { step(cache, tokens: 128) }
    let quantized = cache.quantizedTokenCount
    let target = cache.offset - 8

    cache.restore(.kv(offset: target))
    #expect(cache.offset == target)
    #expect(cache.quantizedTokenCount == quantized)
    #expect(cache.quantizedTokenCount + (cache.window?.keys.dim(2) ?? 0) == target)
  }

  @Test("growth across store reallocations keeps the cache consistent")
  func survivesReallocation() {
    let cache = QuantizedKVCache(config: KVCacheConfig(bits: 3.5, residualWindow: 128))
    var total = 0
    // Past 256, 384, 576, 864, 1296, 1944 -- several capacity doublings.
    while total < 2600 {
      step(cache, tokens: 128)
      total += 128
    }
    #expect(cache.offset == total)
    #expect(cache.quantizedTokenCount + (cache.window?.keys.dim(2) ?? 0) == total)
  }

  @Test("a prefill chunk larger than the drain block is accounted for")
  func largeChunk() {
    let cache = QuantizedKVCache(config: KVCacheConfig(bits: 3.5, residualWindow: 128))
    step(cache, tokens: 512)
    #expect(cache.offset == 512)
    #expect(cache.quantizedTokenCount + (cache.window?.keys.dim(2) ?? 0) == 512)
    step(cache, tokens: 1024)
    #expect(cache.offset == 1536)
    #expect(cache.quantizedTokenCount + (cache.window?.keys.dim(2) ?? 0) == 1536)
  }
}

@Suite("MLX error containment")
struct ErrorContainmentTests {
  @Test("a shape error becomes a Swift throw instead of killing the process")
  func shapeErrorThrows() {
    #expect(throws: (any Error).self) {
      try withError {
        let a = MLXArray(0..<10, [2, 5])
        let b = MLXArray(0..<15, [3, 5])
        eval(a + b)
      }
    }
  }

  @Test("the box reports the failure so a loop can stop early")
  func boxReportsEarly() throws {
    var sawError = false
    #expect(throws: (any Error).self) {
      try withError { box in
        let a = MLXArray(0..<10, [2, 5])
        let b = MLXArray(0..<15, [3, 5])
        eval(a + b)
        sawError = box.firstError != nil
      }
    }
    #expect(sawError)
  }
}
