// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

/// Measure KV-cache quantization: memory saved and quality lost.
public struct KVBench: Sendable {
  public struct Options: Sendable {
    public var model: URL
    public var bits: [Float] = [8, 4, 3.5, 3, 2]
    public var maxTokens: Int = 64
    public var contextTokens: Int = 1024

    public init(model: URL) { self.model = model }
  }

  public static func run(
    _ options: Options,
    log: @escaping @Sendable (String) -> Void
  ) throws {
    let packURL = options.model
    let bonsai = try BonsaiModel(path: packURL)
    let template = try ChatTemplate(path: packURL)

    let filler = String(
      repeating: "The quick brown fox jumps over the lazy dog near the riverbank. ",
      count: max(1, options.contextTokens / 14))
    let rendered = try template.render(
      messages: [.user(filler + "\n\nSummarise the passage above in two sentences.")],
      addGenerationPrompt: true, enableThinking: false)
    let promptTokens = bonsai.tokenizer.encode(rendered)
    log("context: \(promptTokens.count) tokens\n")

    let reference = try measure(bonsai, promptTokens, config: .full, options: options)
    let perToken = Double(reference.bytes) / Double(promptTokens.count)
    log(
      String(
        format: "fp16      keys 16 / values 16  %7.1f MB  %5.1f KB/token  %6.2f tok/s  reference",
        Double(reference.bytes) / 1_048_576, perToken / 1024,
        reference.tokensPerSecond))

    for value in options.bits {
      let config = KVCacheConfig(bits: value)
      do {
        try config.validate()
      } catch {
        log("  \(value): \(error)")
        continue
      }
      let run = try measure(bonsai, promptTokens, config: config, options: options)

      let shared = min(run.tokens.count, reference.tokens.count)
      let matching = zip(run.tokens, reference.tokens).prefix(shared).filter { $0 == $1 }
        .count
      let divergence =
        zip(run.tokens, reference.tokens).enumerated()
        .first { $0.element.0 != $0.element.1 }?.offset
      let label = String(format: "%.1f", value)

      log(
        String(
          format:
            "kv %-5s keys %2d / values %2d  %7.1f MB  %5.1f KB/token  %6.2f tok/s  "
            + "agree %5.1f%%  first diff %@",
          (label as NSString).utf8String!, config.keyBits, config.valueBits,
          Double(run.bytes) / 1_048_576,
          Double(run.bytes) / Double(promptTokens.count) / 1024,
          run.tokensPerSecond,
          shared > 0 ? Double(matching) / Double(shared) * 100 : 0,
          divergence.map(String.init) ?? "none"))
    }

    log("")
    log("Extrapolated to the model's full 262144-token context:")
    log(
      String(
        format: "  fp16 %.1f GB", perToken * 262_144 / 1_073_741_824))
    for value in options.bits {
      let ratio = Double(value) / 16.0
      log(
        String(
          format: "  %.1f-bit ~%.1f GB", value,
          perToken * 262_144 * ratio / 1_073_741_824))
    }
  }

  private static func measure(
    _ bonsai: BonsaiModel, _ promptTokens: [Int], config: KVCacheConfig, options: Options
  ) throws -> (tokens: [Int], bytes: Int, tokensPerSecond: Double) {
    let cache = bonsai.text.makeCache(kvConfig: config)
    let generator = Generator(model: bonsai, kvConfig: config)
    let result = generator.generate(
      promptTokens: promptTokens, options: .greedy, maxTokens: options.maxTokens, cache: cache)
    return (result.tokens, cache.byteCount, result.stats.generationTokensPerSecond)
  }
}
