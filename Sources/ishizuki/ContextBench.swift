// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import IshizukiKit
import MLX

struct ContextBench: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "context-bench",
    abstract: "Measure prefill, decode and cache growth against context length.")

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .long, help: "Context lengths to measure, in tokens.")
  var lengths: String = "1024,4096,16384,32768"
  @Option(name: .long) var decodeTokens: Int = 8
  @Option(name: .long, help: "KV cache bits; pass 16 for fp16.")
  var kvBits: Float = 16
  @Option(name: .long, help: "Extrapolate to this context length.")
  var project: Int = 1_048_576

  func run() throws {
    let kvConfig = KVCacheConfig(bits: kvBits >= 16 ? nil : kvBits)
    try kvConfig.validate()

    let bonsai = try BonsaiModel(directory: URL(filePath: model))
    let unit = bonsai.tokenizer.encode(
      "The distributed ledger records each transaction in a totally ordered log. ")
    print(
      "kv: "
        + (kvConfig.isQuantized
          ? "\(kvConfig.keyBits)-bit keys / \(kvConfig.valueBits)-bit values" : "fp16"))
    print("")
    print("  tokens   prefill      tok/s    decode     KV cache    KB/token   GDN state")

    var samples: [(length: Int, prefill: Double, kvBytes: Int)] = []

    for field in lengths.split(separator: ",") {
      guard let target = Int(field.trimmingCharacters(in: .whitespaces)) else { continue }

      var tokens: [Int] = []
      while tokens.count < target { tokens.append(contentsOf: unit) }
      tokens = Array(tokens.prefix(target))

      let cache = bonsai.text.makeCache(kvConfig: kvConfig)
      let generator = Generator(model: bonsai, kvConfig: kvConfig)

      let result = generator.generate(
        promptTokens: tokens, options: .greedy, maxTokens: decodeTokens, cache: cache)

      let kvBytes = cache.byteCount
      let gdnBytes = gdnStateBytes(cache)
      samples.append((target, result.stats.promptSeconds, kvBytes))

      print(
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
    project(samples)
  }

  private func gdnStateBytes(_ cache: ModelCache) -> Int {
    cache.layers.compactMap { $0 as? GatedDeltaNetCache }
      .reduce(0) { total, layer in
        total + (layer.recurrentState?.nbytes ?? 0) + (layer.convState?.nbytes ?? 0)
      }
  }

  private func project(_ samples: [(length: Int, prefill: Double, kvBytes: Int)]) {
    let first = samples.first!
    let last = samples.last!
    let l1 = Double(first.length)
    let l2 = Double(last.length)
    let determinant = l1 * l2 * l2 - l2 * l1 * l1
    let a = (first.prefill * l2 * l2 - last.prefill * l1 * l1) / determinant
    let b = (last.prefill * l1 - first.prefill * l2) / determinant

    let target = Double(project)
    let projectedPrefill = a * target + b * target * target
    let perToken = Double(last.kvBytes) / Double(last.length)
    let projectedCache = perToken * target

    print("")
    print("Extrapolated to \(project) tokens (fit t = a·L + b·L² over the measured points):")
    print(
      String(
        format: "  KV cache        %.1f GB   (+ 8.6 GB weights, + ~0.15 GB recurrent state)",
        projectedCache / 1_073_741_824))
    print(
      String(
        format: "  one-shot prefill %.1f h   (linear term %.1f h, quadratic term %.1f h)",
        projectedPrefill / 3600, a * target / 3600,
        b * target * target / 3600))
    print(
      String(
        format: "  quadratic share  %.0f%% of prefill at %d tokens",
        b * target * target / projectedPrefill * 100, project))
  }
}
