// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

/// Compare speculative and plain greedy decoding for speed and equality.
public struct SpecBench: Sendable {
  public struct Options: Sendable {
    public var model: URL
    public var maxTokens: Int = 128
    public var draftLength: Int = 4
    /// Where drafts come from: ngram, or mtp for the head the pack ships.
    public var drafter: String = "ngram"
    public var prompt: String?

    public init(model: URL) { self.model = model }
  }

  public static func run(
    _ options: Options,
    log: @escaping @Sendable (String) -> Void
  ) throws {
    let packURL = options.model
    let bonsai = try BonsaiModel(path: packURL)
    let template = try ChatTemplate(path: packURL)

    let text =
      options.prompt
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
    log("prompt: \(promptTokens.count) tokens, max \(options.maxTokens) generated\n")

    let generator = Generator(model: bonsai)
    let baseline = generator.generate(
      promptTokens: promptTokens, options: .greedy, maxTokens: options.maxTokens)
    log(
      String(
        format: "plain greedy      : %6.2f tok/s  (%d tokens in %.2fs)",
        baseline.stats.generationTokensPerSecond, baseline.stats.generatedTokens,
        baseline.stats.generationSeconds))

    let drafting: Drafter
    let label: String
    switch options.drafter {
    case "ngram":
      drafting = NgramDrafter()
      label = "n-gram"
    case "mtp":
      drafting = try MTPDrafter(model: bonsai)
      label = "mtp"
    default:
      throw BenchFailure(
        reason: "unknown drafter '\(options.drafter)'; expected ngram or mtp")
    }

    let decoder = SpeculativeDecoder(
      model: bonsai, drafter: drafting, draftLength: options.draftLength)
    let (speculative, stats) = decoder.generate(
      promptTokens: promptTokens, maxTokens: options.maxTokens)
    log(
      String(
        format: "speculative (%@) : %6.2f tok/s  (%d tokens in %.2fs)",
        label, speculative.stats.generationTokensPerSecond,
        speculative.stats.generatedTokens, speculative.stats.generationSeconds))

    log("")
    log(
      String(
        format: "  acceptance      : %.1f%% (%d of %d drafted)",
        stats.acceptanceRate * 100, stats.accepted, stats.proposed))
    log(String(format: "  tokens / round  : %.2f", stats.tokensPerRound))
    log("  rounds          : \(stats.rounds), with \(stats.rollbacks) rollback(s)")

    let speedup =
      baseline.stats.generationTokensPerSecond > 0
      ? speculative.stats.generationTokensPerSecond
        / baseline.stats.generationTokensPerSecond : 0
    log(String(format: "  speedup         : %.2fx", speedup))

    log("")
    let shared = min(baseline.tokens.count, speculative.tokens.count)
    let identical =
      Array(baseline.tokens.prefix(shared))
      == Array(
        speculative.tokens.prefix(shared))
    if identical {
      log("lossless: speculative output matches greedy exactly over \(shared) tokens.")
    } else {
      let firstDifference =
        zip(baseline.tokens, speculative.tokens).enumerated()
        .first { $0.element.0 != $0.element.1 }?.offset ?? shared
      log("MISMATCH at token \(firstDifference) -- speculation is not lossless.")
      throw BenchFailure(
        reason: "speculative decoding diverged from greedy at token \(firstDifference)")
    }
  }
}
