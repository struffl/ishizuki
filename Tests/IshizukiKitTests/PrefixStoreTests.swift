// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import Testing

@testable import IshizukiKit

@Suite("Prefix store")
struct PrefixStoreTests {
  private let schedule = [false, true, false, true]
  private let kv = KVCacheConfig(bits: 3.5, residualWindow: 8)

  private func store(minimumTokens: Int = 4) -> (PrefixStore, URL) {
    let dir = URL(filePath: NSTemporaryDirectory())
      .appending(path: "prefix-\(UUID().uuidString)")
    return (PrefixStore(directory: dir, minimumTokens: minimumTokens), dir)
  }

  /// Fills a cache with values that differ per position, so a round trip that loses or
  /// reorders tokens cannot pass by accident.
  private func fill(_ cache: ModelCache, tokens: Int) {
    let heads = 2
    let dim = 16
    for layer in cache.layers {
      switch layer {
      case let attention as AttentionKVCache:
        for position in 0..<tokens {
          let value = MLXArray(Float(position + 1) / 64)
          let k = MLXArray.zeros([1, heads, 1, dim], dtype: .float16) + value.asType(.float16)
          _ = attention.appendForAttention(keys: k, values: k)
        }
      case let recurrent as GatedDeltaNetCache:
        recurrent.recurrentState =
          MLXArray.zeros([1, heads, dim, dim], dtype: .float32) + Float(tokens)
        recurrent.convState = MLXArray.zeros([1, 3, dim], dtype: .float16)
        recurrent.advance(tokens)
      default: break
      }
    }
  }

  private func keyTrace(_ cache: ModelCache) -> [Float] {
    var trace: [Float] = []
    for layer in cache.layers {
      guard let attention = layer as? AttentionKVCache else { continue }
      let k = MLXArray.zeros([1, 2, 1, 16], dtype: .float16)
      switch attention.appendForAttention(keys: k, values: k) {
      case .dense(let keys, _):
        trace += keys.asType(.float32).mean(axes: [1, 3]).asArray(Float.self)
      case .quantized(let keys, _, let groupSize, let keyBits, _):
        let restored = dequantized(
          keys.0, scales: keys.1, biases: keys.2, groupSize: groupSize, bits: keyBits,
          mode: .affine)
        trace += restored.asType(.float32).mean(axes: [1, 3]).asArray(Float.self)
      }
    }
    return trace
  }

  @Test("a saved prefix comes back with its tokens and its contents")
  func roundTrip() throws {
    let (store, dir) = store()
    defer { try? FileManager.default.removeItem(at: dir) }

    let tokens = Array(1...64)
    let original = ModelCache(fullAttention: schedule, kvConfig: kv)
    fill(original, tokens: tokens.count)

    let id = store.save(cache: original, tokens: tokens, modelID: "pack-a", kvConfig: kv)
    #expect(id != nil, "save failed: \(store.lastError ?? "no error recorded")")

    // Read after the save: taking a trace appends a probe token, which would put the cache
    // out of step with the token list it was archived under.
    let expected = keyTrace(original)

    let entry = try #require(store.bestMatch(for: tokens + [999], modelID: "pack-a", kvConfig: kv))
    #expect(entry.tokens == tokens)

    let restored = ModelCache(fullAttention: schedule, kvConfig: kv)
    #expect(store.load(entry, into: restored))
    #expect(restored.offset == tokens.count)

    // The restored cache has to answer attention the same way the original did.
    #expect(keyTrace(restored) == expected)
  }

  @Test("a prefix from another model or another cache geometry is never matched")
  func fingerprinted() {
    let (store, dir) = store()
    defer { try? FileManager.default.removeItem(at: dir) }

    let tokens = Array(1...64)
    let cache = ModelCache(fullAttention: schedule, kvConfig: kv)
    fill(cache, tokens: tokens.count)
    #expect(store.save(cache: cache, tokens: tokens, modelID: "pack-a", kvConfig: kv) != nil)

    let probe = tokens + [999]
    #expect(store.bestMatch(for: probe, modelID: "pack-b", kvConfig: kv) == nil)
    #expect(
      store.bestMatch(
        for: probe, modelID: "pack-a", kvConfig: KVCacheConfig(bits: 8, residualWindow: 8))
        == nil)
    #expect(store.bestMatch(for: probe, modelID: "pack-a", kvConfig: kv) != nil)
  }

  @Test("only a prefix of the prompt matches, never a divergent or equal one")
  func matching() {
    let (store, dir) = store()
    defer { try? FileManager.default.removeItem(at: dir) }

    let tokens = Array(1...64)
    let cache = ModelCache(fullAttention: schedule, kvConfig: kv)
    fill(cache, tokens: tokens.count)
    store.save(cache: cache, tokens: tokens, modelID: "pack-a", kvConfig: kv)

    // Diverges inside the archive: unusable, because the cache cannot be rewound from disk.
    #expect(store.bestMatch(for: Array(1...32) + [500], modelID: "pack-a", kvConfig: kv) == nil)
    // Exactly equal: nothing left to forward, so there would be no logits to sample.
    #expect(store.bestMatch(for: tokens, modelID: "pack-a", kvConfig: kv) == nil)
    #expect(store.bestMatch(for: tokens + [65], modelID: "pack-a", kvConfig: kv) != nil)
  }

  @Test("a short prefix is not worth a file")
  func skipsShort() {
    let (store, dir) = store(minimumTokens: 128)
    defer { try? FileManager.default.removeItem(at: dir) }

    let tokens = Array(1...64)
    let cache = ModelCache(fullAttention: schedule, kvConfig: kv)
    fill(cache, tokens: tokens.count)
    #expect(store.save(cache: cache, tokens: tokens, modelID: "pack-a", kvConfig: kv) == nil)
    #expect(store.entries().isEmpty)
  }

  @Test("the store holds itself to its byte ceiling, coldest first")
  func evicts() {
    let (store, dir) = store()
    defer { try? FileManager.default.removeItem(at: dir) }

    for run in 0..<3 {
      let tokens = Array((run * 100)...(run * 100 + 63))
      let cache = ModelCache(fullAttention: schedule, kvConfig: kv)
      fill(cache, tokens: tokens.count)
      store.save(cache: cache, tokens: tokens, modelID: "pack-a", kvConfig: kv)
    }
    #expect(store.entries().count == 3)

    store.setByteLimit(1)
    #expect(store.entries().isEmpty)
    #expect(store.totalBytes == 0)
  }
}
