// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The real DeepSeek-V4.1-Flash, loaded from its release shards and asked something.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// Runs only where every shard of the release is present, since a partial download cannot
/// load. `ISHIZUKI_DEEPSEEK_PROMPT` and `ISHIZUKI_DEEPSEEK_TOKENS` change what it asks and for
/// how long.
@Suite(
  "DeepSeek-V4.1 release",
  .enabled(
    if: DeepSeekRelease.directory.map {
      FileManager.default.fileExists(
        atPath: $0.appending(path: "model-00048-of-00048.safetensors").path)
    } ?? false))
struct DeepSeekReleaseProbe {
  @Test("answers a question from the release shards")
  func answers() throws {
    let directory = try #require(DeepSeekRelease.directory)
    let environment = ProcessInfo.processInfo.environment
    let question = environment["ISHIZUKI_DEEPSEEK_PROMPT"] ?? "What is the capital of France?"
    let budget = Int(environment["ISHIZUKI_DEEPSEEK_TOKENS"] ?? "") ?? 12

    let loadStart = Date()
    let model = try BonsaiModel(directory: directory)
    print(String(format: "loaded in %.1f s", -loadStart.timeIntervalSinceNow))

    let template = try ChatTemplate(directory: directory)
    let prompt = try template.render(
      messages: [.user(question)], addGenerationPrompt: true, enableThinking: false)
    let tokens = model.tokenizer.encode(prompt)
    print("prompt: \(prompt.debugDescription) → \(tokens.count) tokens")

    let generator = Generator(model: model)
    let result = generator.generate(
      promptTokens: tokens, options: SamplingOptions(temperature: 0), maxTokens: budget,
      onToken: { fragment in
        print(fragment, terminator: "")
        fflush(stdout)
        return true
      })
    print()
    print(
      String(
        format: "prefill %d tokens in %.1f s, %d generated at %.3f tok/s", result.stats.promptTokens,
        result.stats.promptSeconds, result.stats.generatedTokens,
        result.stats.generationTokensPerSecond))
    if let traffic = model.deepseek?.expertTraffic {
      print(
        String(
          format: "experts: %d reads, %.1f%% hits", traffic.reads, 100 * traffic.hitRate))
    }
    #expect(!result.tokens.isEmpty)
  }
}
