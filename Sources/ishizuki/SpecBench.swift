// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import IshizukiKit
import MLX

struct SpecBench: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "spec-bench",
    abstract: "Compare speculative and plain greedy decoding for speed and equality.")

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .shortAndLong) var maxTokens: Int = 128
  @Option(name: .long, help: "Tokens drafted per round.") var draftLength: Int = 4

  @Option(
    name: .long,
    help: "Where drafts come from: ngram, or mtp for the head the pack ships.")
  var drafter: String = "ngram"

  @Option(
    name: .long,
    help: "Prompt to use. Defaults to a repetitive one, where n-gram drafting should win.")
  var prompt: String?

  func run() throws {
    let packURL = URL(filePath: model)
    let bonsai = try BonsaiModel(directory: packURL)
    let template = try ChatTemplate(directory: packURL)

    let text =
      prompt
        ?? """
        Here is a config block:

        host = localhost
        port = 8080
        timeout = 30
        retries = 3

        Repeat that config block back to me exactly three times, unchanged.
        """

    let rendered = try template.render(
      messages: [.user(text)], addGenerationPrompt: true, enableThinking: false)
    let promptTokens = bonsai.tokenizer.encode(rendered)
    print("prompt: \(promptTokens.count) tokens, max \(maxTokens) generated\n")

    let generator = Generator(model: bonsai)
    let baseline = generator.generate(
      promptTokens: promptTokens, options: .greedy, maxTokens: maxTokens)
    print(
      String(
        format: "plain greedy      : %6.2f tok/s  (%d tokens in %.2fs)",
        baseline.stats.generationTokensPerSecond, baseline.stats.generatedTokens,
        baseline.stats.generationSeconds))

    let drafting: Drafter
    let label: String
    switch drafter {
    case "ngram":
      drafting = NgramDrafter()
      label = "n-gram"
    case "mtp":
      drafting = try MTPDrafter(model: bonsai)
      label = "mtp"
    default:
      throw ValidationError("unknown drafter '\(drafter)'; expected ngram or mtp")
    }

    let decoder = SpeculativeDecoder(
      model: bonsai, drafter: drafting, draftLength: draftLength)
    let (speculative, stats) = decoder.generate(
      promptTokens: promptTokens, maxTokens: maxTokens)
    print(
      String(
        format: "speculative (%@) : %6.2f tok/s  (%d tokens in %.2fs)",
        label, speculative.stats.generationTokensPerSecond,
        speculative.stats.generatedTokens, speculative.stats.generationSeconds))

    print("")
    print(
      String(
        format: "  acceptance      : %.1f%% (%d of %d drafted)",
        stats.acceptanceRate * 100, stats.accepted, stats.proposed))
    print(String(format: "  tokens / round  : %.2f", stats.tokensPerRound))
    print("  rounds          : \(stats.rounds), with \(stats.rollbacks) rollback(s)")

    let speedup =
      baseline.stats.generationTokensPerSecond > 0
      ? speculative.stats.generationTokensPerSecond
        / baseline.stats.generationTokensPerSecond : 0
    print(String(format: "  speedup         : %.2fx", speedup))

    print("")
    let shared = min(baseline.tokens.count, speculative.tokens.count)
    let identical =
      Array(baseline.tokens.prefix(shared))
      == Array(
        speculative.tokens.prefix(shared))
    if identical {
      print("lossless: speculative output matches greedy exactly over \(shared) tokens.")
    } else {
      let firstDifference =
        zip(baseline.tokens, speculative.tokens).enumerated()
        .first { $0.element.0 != $0.element.1 }?.offset ?? shared
      print("MISMATCH at token \(firstDifference) -- speculation is not lossless.")
      throw ExitCode.failure
    }
  }
}
