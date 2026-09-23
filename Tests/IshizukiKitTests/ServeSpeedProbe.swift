// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Testing

@testable import IshizukiKit

// Probe, not a test: a served turn's decode rate with drafting off and on.
@Suite("ServeSpeedProbe", .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_PACK"] != nil))
struct ServeSpeedProbe {
  @Test("probe")
  func probe() throws {
    let url = URL(filePath: ProcessInfo.processInfo.environment["ISHIZUKI_PACK"]!)
    let model = try BonsaiModel(path: url)
    let template = try ChatTemplate(path: url)
    let code = """
      func total(_ items: [Item]) -> Double {
          var sum = 0.0
          for item in items {
              sum += item.price * Double(item.quantity)
          }
          return sum
      }
      """
    let prompts = [
      ("prose", "Explain in a few paragraphs why the sky looks blue during the day and red at sunset."),
      ("edit", "Rename `sum` to `runningTotal` in this function and reply with the whole function only.\n\n\(code)"),
    ]
    let samplers: [(String, SamplingOptions)] = [
      ("greedy", .greedy),
      ("app", SamplingOptions(temperature: 0.7, minP: 0.05, seed: 5)),
      ("temp", SamplingOptions(temperature: 0.7, seed: 5)),
    ]
    if let match = ProcessInfo.processInfo.environment["ISHIZUKI_MIN_MATCH"].flatMap({ Int($0) }) {
      BonsaiRuntime.lookupMinMatch = match
    }
    let saved = BonsaiRuntime.speculativeDecode
    defer { BonsaiRuntime.speculativeDecode = saved }
    let only = ProcessInfo.processInfo.environment["ISHIZUKI_ONLY"]
    for (label, text) in prompts where only == nil || only == label {
      let rendered = try template.render(
        messages: [.user(text)], addGenerationPrompt: true, enableThinking: false)
      let tokens = model.tokenizer.encode(rendered)
      for (sampling, options) in samplers where only == nil || sampling != "greedy" {
        var plain: [Int] = []
        for drafting in [false, true] {
          BonsaiRuntime.speculativeDecode = drafting
          let result = Generator(model: model, politeness: .normal).generate(
            promptTokens: tokens, options: options, maxTokens: 120)
          if !drafting { plain = result.tokens }
          let same = drafting && options.temperature == 0
            ? (result.tokens == plain ? "  same tokens" : "  DIFFERS at \(zip(result.tokens, plain).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? -1)")
            : ""
          let spec = result.speculative.map {
            String(format: "  %.2f tok/round, %d/%d accepted", $0.tokensPerRound, $0.accepted, $0.proposed)
          } ?? ""
          print(String(format: "%-5@ %-6@ drafting %@ : %6.2f tok/s (%d tokens)%@",
            label, sampling, drafting ? "on " : "off", result.stats.generationTokensPerSecond,
            result.tokens.count, spec + same))
        }
      }
    }
  }
}
