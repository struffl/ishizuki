// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

/// Measure prefill, decode and cache growth against context length.
public struct ContextBench: Sendable {
  public struct Options: Sendable {
    public var model: URL
    public var lengths: [Int] = [1024, 4096, 16384, 32768]
    public var decodeTokens: Int = 8
    public var kvBits: Float = 16
    public var project: Int = 1_048_576

    public init(model: URL) { self.model = model }
  }

  public static func run(
    _ options: Options,
    log: @escaping @Sendable (String) -> Void
  ) throws {
    let kvConfig = KVCacheConfig(bits: options.kvBits >= 16 ? nil : options.kvBits)
    try kvConfig.validate()

    let bonsai = try BonsaiModel(path: options.model)
    let unit = bonsai.tokenizer.encode(
      "The distributed ledger records each transaction in a totally ordered log. ")
    log(
      "kv: "
        + (kvConfig.isQuantized
          ? "\(kvConfig.keyBits)-bit keys / \(kvConfig.valueBits)-bit values" : "fp16"))
    log("")
    log("  tokens   prefill      tok/s    decode     KV cache    KB/token   GDN state")

    var samples: [(length: Int, prefill: Double, kvBytes: Int)] = []

    for target in options.lengths {
      var tokens: [Int] = []
      while tokens.count < target { tokens.append(contentsOf: unit) }
      tokens = Array(tokens.prefix(target))

      let cache = bonsai.backbone.makeCache(kvConfig: kvConfig)
      let generator = Generator(model: bonsai, kvConfig: kvConfig)

      let result = generator.generate(
        promptTokens: tokens, options: .greedy, maxTokens: options.decodeTokens, cache: cache)

      let kvBytes = cache.byteCount
      let gdnBytes = gdnStateBytes(cache)
      samples.append((target, result.stats.promptSeconds, kvBytes))

      log(
        String(
          format: "  %6d   %7.2fs  %8.1f  %6.2f/s   %8.2f GB   %7.2f   %6.1f MB",
          target, result.stats.promptSeconds, result.stats.promptTokensPerSecond,
          result.stats.generationTokensPerSecond,
          Double(kvBytes) / 1_073_741_824,
          Double(kvBytes) / Double(target) / 1024,
          Double(gdnBytes) / 1_048_576))

      cache.reset()
      Memory.clearCache()
    }

    guard samples.count >= 2 else { return }
    extrapolate(samples, options: options, log: log)
  }

  private static func gdnStateBytes(_ cache: ModelCache) -> Int {
    cache.layers.compactMap { $0 as? GatedDeltaNetCache }
      .reduce(0) { total, layer in
        total + (layer.recurrentState?.nbytes ?? 0) + (layer.convState?.nbytes ?? 0)
      }
  }

  private static func extrapolate(
    _ samples: [(length: Int, prefill: Double, kvBytes: Int)],
    options: Options,
    log: (String) -> Void
  ) {
    let first = samples.first!
    let last = samples.last!
    let l1 = Double(first.length)
    let l2 = Double(last.length)
    let determinant = l1 * l2 * l2 - l2 * l1 * l1
    let a = (first.prefill * l2 * l2 - last.prefill * l1 * l1) / determinant
    let b = (last.prefill * l1 - first.prefill * l2) / determinant

    let target = Double(options.project)
    let projectedPrefill = a * target + b * target * target
    let perToken = Double(last.kvBytes) / Double(last.length)
    let projectedCache = perToken * target

    log("")
    log("Extrapolated to \(options.project) tokens (fit t = a·L + b·L² over the measured points):")
    log(
      String(
        format: "  KV cache        %.1f GB   (+ 8.6 GB weights, + ~0.15 GB recurrent state)",
        projectedCache / 1_073_741_824))
    log(
      String(
        format: "  one-shot prefill %.1f h   (linear term %.1f h, quadratic term %.1f h)",
        projectedPrefill / 3600, a * target / 3600,
        b * target * target / 3600))
    log(
      String(
        format: "  quadratic share  %.0f%% of prefill at %d tokens",
        b * target * target / projectedPrefill * 100, options.project))
  }
}
