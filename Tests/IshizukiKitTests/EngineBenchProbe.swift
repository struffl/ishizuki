// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// One pack's prompt and generation rates, the way the engine comparison reads them.

import Foundation
import Testing

@testable import IshizukiKit

/// `ISHIZUKI_BENCH_PACK` is any pack this build reads: a GGUF file, an EXL3 or MLX directory.
/// Prompt processing is timed at 512 and 2048 tokens; generation on three prompts, greedy and
/// with reasoning off, plainly, with the pack's MTP head when it has one, and with a DFlash
/// drafter when one is in reach. Every rate is the better of two runs.
@Suite(
  "Engine bench", .serialized,
  .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_BENCH_PACK"] != nil))
struct EngineBenchProbe {
  @Test("reads a pack's prompt and generation rates")
  func bench() throws {
    let environment = ProcessInfo.processInfo.environment
    let url = URL(fileURLWithPath: environment["ISHIZUKI_BENCH_PACK"]!)
    let budget = Int(environment["ISHIZUKI_TOKENS"] ?? "") ?? 256
    if environment["ISHIZUKI_VERIFY"] == "0" { BonsaiRuntime.useVerifyMatmul = false }
    defer { BonsaiRuntime.useVerifyMatmul = true }
    let model = try BonsaiModel(path: url)
    let template = try ChatTemplate(path: url)
    let draft =
      environment["ISHIZUKI_BENCH_DFLASH"] == "0"
      ? nil
      : DFlashDraft.find(for: model.text, beside: url).flatMap { try? DFlashDraft(directory: $0) }
    print("pack \(url.lastPathComponent): mtp \(model.mtp != nil), dflash \(draft != nil)")

    let passage = """
      The lighthouse stood at the end of a long spit of shingle, and for forty years the same \
      family had kept its lamp. Each evening the keeper climbed the iron stair, trimmed the wick, \
      wound the clockwork that turned the lens, and wrote the weather in a ledger bound in green \
      cloth. Ships passed in the dark, and none of them knew his name.
      """
    for length in environment["ISHIZUKI_BENCH_PP"] == "0" ? [] : [512, 2048] {
      var words: [Int] = []
      while words.count < length { words += model.tokenizer.encode(passage) }
      let prompt = Array(words.prefix(length))
      var rate = 0.0
      for _ in 0..<2 {
        let generator = Generator(model: model)
        generator.speculativeDecode = false
        let result = generator.generate(promptTokens: prompt, options: .greedy, maxTokens: 1)
        rate = max(rate, Double(result.stats.promptTokens) / result.stats.promptSeconds)
      }
      print(String(format: "pp%d: %7.1f tok/s", length, rate))
    }

    let prompts = [
      (
        "math",
        "How many positive whole-number divisors does 196 have? Show your reasoning step by step."
      ),
      (
        "code",
        "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and a short usage example."
      ),
      (
        "prose",
        "Explain in a few paragraphs why the sky looks blue during the day and red at sunset."
      ),
    ]
    var modes: [(String, Bool, DFlashDraft?)] = [("plain", false, nil)]
    if model.mtp != nil { modes.append(("mtp", true, nil)) }
    if let draft { modes.append(("dflash", true, draft)) }
    for (label, text) in prompts {
      let tokens = model.tokenizer.encode(
        try template.render(
          messages: [.user(text)], addGenerationPrompt: true, enableThinking: false))
      var line = String(format: "tg %-5@", label)
      for (mode, drafting, drafter) in modes {
        var rate = 0.0
        var perVerify = 0.0
        for _ in 0..<2 {
          model.dflash = drafter
          let generator = Generator(model: model)
          generator.speculativeDecode = drafting
          let result = generator.generate(promptTokens: tokens, options: .greedy, maxTokens: budget)
          model.dflash = nil
          rate = max(rate, result.stats.generationTokensPerSecond)
          perVerify = result.speculative?.tokensPerRound ?? 0
        }
        line += String(format: "  %@ %6.2f", mode, rate)
        if drafting { line += String(format: " (%.2f/verify)", perVerify) }
      }
      print(line)
    }
  }
}
