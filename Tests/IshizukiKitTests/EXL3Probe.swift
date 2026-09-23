// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Probe, not a test: an EXL3 pack end to end — what it says, and how fast.

import Foundation
import Testing

@testable import IshizukiKit

@Suite("EXL3Probe", .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_EXL3"] != nil))
struct EXL3Probe {
  @Test("probe")
  func probe() throws {
    let url = URL(filePath: ProcessInfo.processInfo.environment["ISHIZUKI_EXL3"]!)
    let started = Date()
    let model = try BonsaiModel(path: url)
    let template = try ChatTemplate(path: url)
    print(String(format: "loaded in %.1f s, mtp %@", Date().timeIntervalSince(started),
      model.mtp == nil ? "absent" : "present"))
    let prompts = [
      "Explain in a few paragraphs why the sky looks blue during the day and red at sunset.",
      "Write a Swift function that returns the n-th Fibonacci number iteratively.",
    ]
    let saved = BonsaiRuntime.speculativeDecode
    defer { BonsaiRuntime.speculativeDecode = saved }
    for text in prompts {
      let rendered = try template.render(
        messages: [.user(text)], addGenerationPrompt: true, enableThinking: false)
      let tokens = model.tokenizer.encode(rendered)
      var plain: [Int] = []
      for drafting in model.mtp == nil ? [false] : [false, true] {
        BonsaiRuntime.speculativeDecode = drafting
        let result = Generator(model: model, politeness: .normal).generate(
          promptTokens: tokens, options: .greedy, maxTokens: 160)
        if !drafting {
          plain = result.tokens
          print("---\n\(model.tokenizer.decode(result.tokens, skipSpecialTokens: true))\n---")
        }
        let spec = result.speculative.map {
          String(format: "  %.2f tok/round, %d/%d accepted", $0.tokensPerRound, $0.accepted, $0.proposed)
        } ?? ""
        let same = drafting ? (result.tokens == plain ? "  same tokens" : "  DIFFERS") : ""
        print(String(format: "drafting %@: prefill %.1f tok/s, decode %.2f tok/s (%d tokens)%@",
          drafting ? "on " : "off", result.stats.promptTokensPerSecond,
          result.stats.generationTokensPerSecond, result.tokens.count, spec + same))
      }
    }
  }
}
