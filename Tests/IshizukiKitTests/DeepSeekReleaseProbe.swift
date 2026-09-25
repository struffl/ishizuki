// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The real DeepSeek-V4.1-Flash, loaded from its release shards and asked something.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// Runs only when `ISHIZUKI_DEEPSEEK` names a release with every shard present: a partial
/// download cannot load, and a whole one found on its own would have every test run stream
/// hundreds of gigabytes. `ISHIZUKI_DEEPSEEK_PROMPT` and `ISHIZUKI_DEEPSEEK_TOKENS` change what it asks and for
/// how long, `ISHIZUKI_EXPERT_SLOTS` how many experts a layer holds, `ISHIZUKI_DEEPSEEK_RUNS` how
/// many times the same question is asked: the n-gram rows a first run read off disk are in memory
/// for the next, so a second run is what the release makes of a disk that answers at once.
/// Everything MLX holds is wired for the run, the way the app's wired memory setting does it.
@Suite(
  "DeepSeek-V4.1 release",
  .enabled(
    if: ProcessInfo.processInfo.environment["ISHIZUKI_DEEPSEEK"].map {
      FileManager.default.fileExists(
        atPath: URL(fileURLWithPath: $0).appending(path: "model-00048-of-00048.safetensors").path)
    } ?? false))
struct DeepSeekReleaseProbe {
  @Test("answers a question from the release shards")
  func answers() async throws {
    let directory = try #require(DeepSeekRelease.directory)
    let environment = ProcessInfo.processInfo.environment
    let question = environment["ISHIZUKI_DEEPSEEK_PROMPT"] ?? "What is the capital of France?"
    let budget = Int(environment["ISHIZUKI_DEEPSEEK_TOKENS"] ?? "") ?? 12
    let runs = Int(environment["ISHIZUKI_DEEPSEEK_RUNS"] ?? "") ?? 1
    if let slots = Int(environment["ISHIZUKI_EXPERT_SLOTS"] ?? "") {
      BonsaiRuntime.expertSlots = slots
    }
    defer { BonsaiRuntime.expertSlots = 0 }

    let loadStart = Date()
    let model = try BonsaiModel(directory: directory)
    print(String(format: "loaded in %.1f s", -loadStart.timeIntervalSinceNow))
    let held = Memory.activeMemory + Memory.cacheMemory
    let wired = WiredMemoryTicket(size: held + (4 << 30), policy: WiredSumPolicy())
    _ = await wired.start()

    let template = try ChatTemplate(directory: directory)
    let prompt = try template.render(
      messages: [.user(question)], addGenerationPrompt: true, enableThinking: false)
    let tokens = model.tokenizer.encode(prompt)
    print("prompt: \(prompt.debugDescription) → \(tokens.count) tokens")

    for _ in 0..<runs {
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
          format: "prefill %d tokens in %.1f s (%.2f tok/s), %d generated at %.3f tok/s",
          result.stats.promptTokens, result.stats.promptSeconds,
          Double(result.stats.promptTokens) / result.stats.promptSeconds,
          result.stats.generatedTokens, result.stats.generationTokensPerSecond))
      if let traffic = model.deepseek?.expertTraffic {
        print(
          String(
            format: "experts: %d reads, %.1f%% hits", traffic.reads, 100 * traffic.hitRate))
      }
      #expect(!result.tokens.isEmpty)
    }
    _ = await wired.end()
  }
}
