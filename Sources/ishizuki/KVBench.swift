// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit
import MLX

struct KVBench: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "kv-bench",
    abstract: "Measure KV-cache quantization: memory saved and quality lost.")

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .long, help: "Bit budgets to compare, e.g. 8,4,3.5,3")
  var bits: String = "8,4,3.5,3,2"
  @Option(name: .long) var maxTokens: Int = 64
  @Option(name: .long, help: "Tokens of context to build before measuring.")
  var contextTokens: Int = 1024

  func run() throws {
    let packURL = URL(filePath: model)
    let bonsai = try BonsaiModel(directory: packURL, loadVision: false)
    let template = try ChatTemplate(directory: packURL)

    let filler = String(
      repeating: "The quick brown fox jumps over the lazy dog near the riverbank. ",
      count: max(1, contextTokens / 14))
    let rendered = try template.render(
      messages: [.user(filler + "\n\nSummarise the passage above in two sentences.")],
      addGenerationPrompt: true, enableThinking: false)
    let promptTokens = bonsai.tokenizer.encode(rendered)
    print("context: \(promptTokens.count) tokens\n")

    let reference = try run(bonsai, promptTokens, config: .full)
    let perToken = Double(reference.bytes) / Double(promptTokens.count)
    print(
      String(
        format: "fp16      keys 16 / values 16  %7.1f MB  %5.1f KB/token  %6.2f tok/s  reference",
        Double(reference.bytes) / 1_048_576, perToken / 1024,
        reference.tokensPerSecond))

    for field in bits.split(separator: ",") {
      guard let value = Float(field.trimmingCharacters(in: .whitespaces)) else { continue }
      let config = KVCacheConfig(bits: value)
      do {
        try config.validate()
      } catch {
        print("  \(field): \(error)")
        continue
      }
      let run = try self.run(bonsai, promptTokens, config: config)

      let shared = min(run.tokens.count, reference.tokens.count)
      let matching = zip(run.tokens, reference.tokens).prefix(shared).filter { $0 == $1 }
        .count
      let divergence =
        zip(run.tokens, reference.tokens).enumerated()
        .first { $0.element.0 != $0.element.1 }?.offset
      let label = String(format: "%.1f", value)

      print(
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

    print("")
    print("Extrapolated to the model's full 262144-token context:")
    print(
      String(
        format: "  fp16 %.1f GB", perToken * 262_144 / 1_073_741_824))
    for field in bits.split(separator: ",") {
      guard let value = Float(field.trimmingCharacters(in: .whitespaces)) else { continue }
      let ratio = Double(value) / 16.0
      print(
        String(
          format: "  %.1f-bit ~%.1f GB", value,
          perToken * 262_144 * ratio / 1_073_741_824))
    }
  }

  private func run(
    _ bonsai: BonsaiModel, _ promptTokens: [Int], config: KVCacheConfig
  ) throws -> (tokens: [Int], bytes: Int, tokensPerSecond: Double) {
    let cache = bonsai.text.makeCache(kvConfig: config)
    let generator = Generator(model: bonsai, kvConfig: config)
    let result = generator.generate(
      promptTokens: promptTokens, options: .greedy, maxTokens: maxTokens, cache: cache)
    return (result.tokens, cache.byteCount, result.stats.generationTokensPerSecond)
  }
}
