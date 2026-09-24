// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DFlash drafting on a real pack, against z-lab's own MLX code.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

/// `ISHIZUKI_DFLASH` is a drafter as z-lab ships it. `ISHIZUKI_DFLASH_GOLDEN` is what their code
/// drafted from the same taps, and `ISHIZUKI_PACK` the backbone the head comes from.
@Suite("DFlash", .serialized, .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_DFLASH"] != nil))
struct DFlashProbe {
  private let environment = ProcessInfo.processInfo.environment

  private func drift(_ got: MLXArray, _ want: MLXArray) -> Float {
    let worst = (got.asType(.float32) - want).abs().max().item(Float.self)
    return worst / max(want.abs().max().item(Float.self), 1e-12)
  }

  @Test("drafts from the taps z-lab's code drafts from")
  func matchesTheReference() throws {
    let golden = try loadArrays(
      url: URL(fileURLWithPath: try #require(environment["ISHIZUKI_DFLASH_GOLDEN"])))
    let draft = try DFlashDraft(directory: URL(fileURLWithPath: environment["ISHIZUKI_DFLASH"]!))
    let caches = draft.makeCaches()
    let taps = try #require(golden["taps"])
    let hidden = draft.hidden(
      block: try #require(golden["embedded"]), taps: taps, caches: caches)
    let first = drift(hidden, try #require(golden["hidden"]))
    let again = draft.hidden(
      block: try #require(golden["second_embedded"]), taps: taps[0..., (taps.dim(1) - 1)...],
      caches: caches)
    let second = drift(again, try #require(golden["second_hidden"]))
    print("hidden drift \(first), after one more tap \(second)")
    #expect(first < 2e-2 && second < 2e-2)

    guard let pack = environment["ISHIZUKI_PACK"] else { return }
    let model = try BonsaiModel(path: URL(fileURLWithPath: pack))
    let logits = model.text.lmHead(hidden)
    let wanted = try #require(golden["path"]).asArray(Int32.self)
    let path = draft.drafts(
      hidden: hidden, logits: logits, anchor: try #require(golden["block"])[0..., 0]
    ).asArray(UInt32.self).map(Int32.init)
    print("path \(path), reference \(wanted)")
    #expect(path == wanted)
  }

  /// Verification makes drafting exact, so greedy decoding with the draft has to come out token
  /// for token what it is without; the rates are what the draft is for.
  @Test("drafts to exactly what plain greedy decoding says")
  func draftsExactly() throws {
    let pack = URL(fileURLWithPath: try #require(environment["ISHIZUKI_PACK"]))
    let model = try BonsaiModel(path: pack)
    let template = try ChatTemplate(path: pack)
    let draft = try DFlashDraft(directory: URL(fileURLWithPath: environment["ISHIZUKI_DFLASH"]!))
    let budget = Int(environment["ISHIZUKI_TOKENS"] ?? "") ?? 256
    let prompts = [
      ("prose", "Explain in a few paragraphs why the sky looks blue during the day and red at sunset."),
      ("math", "How many positive whole-number divisors does 196 have? Show your reasoning step by step."),
      ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and a short usage example."),
    ]
    for (label, text) in prompts {
      let tokens = model.tokenizer.encode(
        try template.render(messages: [.user(text)], addGenerationPrompt: true, enableThinking: false))
      func run(_ drafting: Bool) -> GenerationResult {
        model.dflash = drafting ? draft : nil
        defer { model.dflash = nil }
        let generator = Generator(model: model)
        generator.speculativeDecode = drafting
        return generator.generate(promptTokens: tokens, options: .greedy, maxTokens: budget)
      }
      var plain = run(false)
      var drafted = run(true)
      var plainRate = plain.stats.generationTokensPerSecond
      var draftedRate = drafted.stats.generationTokensPerSecond
      plain = run(false)
      drafted = run(true)
      plainRate = max(plainRate, plain.stats.generationTokensPerSecond)
      draftedRate = max(draftedRate, drafted.stats.generationTokensPerSecond)
      let stats = drafted.speculative ?? SpeculativeStats()
      print(String(format: "%-5@ plain %6.2f tok/s   dflash %6.2f tok/s   %.2f tokens/verify over %d rounds",
        label, plainRate, draftedRate, stats.tokensPerRound, stats.rounds))
      if let split = zip(plain.tokens, drafted.tokens).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset {
        let cache = model.text.makeCache()
        let context = tokens + plain.tokens[..<split]
        let logits = model.text(
          MLXArray(context.map { Int32($0) }).reshaped([1, context.count]), cache: cache)[0, -1]
          .asType(.float32)
        let top = sorted(logits)[(logits.dim(0) - 2)...].asArray(Float.self)
        print(String(format: "%@ parts at token %d: plain %d, drafted %d, top two logits %.4f apart",
          label, split, plain.tokens[split], drafted.tokens[split], top[1] - top[0]))
        #expect(top[1] - top[0] < 0.05, "\(label): drafting changed a pick that was not a near tie")
      }
    }
  }

  /// Where a drafted round's time goes: the draft, the verify's trunk, and its head.
  @Test("times a round's parts")
  func timesARound() throws {
    let pack = URL(fileURLWithPath: try #require(environment["ISHIZUKI_PACK"]))
    let model = try BonsaiModel(path: pack)
    let text = try #require(model.text)
    let draft = try DFlashDraft(directory: URL(fileURLWithPath: environment["ISHIZUKI_DFLASH"]!))
    let prompt = model.tokenizer.encode("Write a short story about a lighthouse keeper.")
    let cache = text.makeCache()
    let drafter = DFlashDrafter(draft: draft, backbone: text)
    let prefill = text.trunk(
      inputs: MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count]), cache: cache,
      taps: drafter.taps)
    eval(prefill.trunk, prefill.taps!)
    let block = MLXArray((0..<8).map { Int32(1000 + $0) }).reshaped([1, 8])
    let one = MLXArray([Int32(1000)]).reshaped([1, 1])

    func time(_ label: String, _ work: () -> Void) {
      for _ in 0..<3 { work() }
      let start = Date()
      for _ in 0..<10 { work() }
      print(String(format: "%-34@ %7.2f ms", label, -start.timeIntervalSinceNow * 100))
    }
    let saved = cache.snapshot()
    time("1-row step, trunk + head") {
      eval(text.lmHead(text.normed(text.trunk(inputs: one, cache: cache))))
      cache.restore(saved)
    }
    time("8-row trunk") {
      eval(text.trunk(inputs: block, cache: cache))
      cache.restore(saved)
    }
    time("8-row trunk with taps") {
      let step = text.trunk(inputs: block, cache: cache, taps: drafter.taps)
      eval(step.trunk, step.taps!)
      cache.restore(saved)
    }
    time("8-row trunk + head") {
      eval(text.lmHead(text.normed(text.trunk(inputs: block, cache: cache))))
      cache.restore(saved)
    }
    let hidden = MLXRandom.normal([1, 8, text.config.hiddenSize]).asType(.float16)
    time("8-row head alone") { eval(text.lmHead(hidden)) }
    time("1-row head alone") { eval(text.lmHead(hidden[0..., ..<1])) }
    let taps = prefill.taps![0..., (prefill.taps!.dim(1) - 6)...]
    time("draft, 6 taps in") {
      let caches = draft.makeCaches()
      let embedded = text.embedTokens(block)
      let h = draft.hidden(block: embedded, taps: taps, caches: caches)
      let logits = text.lmHead(h.asType(embedded.dtype))
      eval(draft.drafts(hidden: h, logits: logits, anchor: block[0..., 0]))
    }
    time("draft without the head and selector") {
      let caches = draft.makeCaches()
      eval(draft.hidden(block: text.embedTokens(block), taps: taps, caches: caches))
    }
  }

  /// The verify's trunk by layer kind, one row against eight.
  @Test("times the trunk's layers")
  func timesTheLayers() throws {
    let pack = URL(fileURLWithPath: try #require(environment["ISHIZUKI_PACK"]))
    let model = try BonsaiModel(path: pack)
    let text = try #require(model.text)
    let linear = try #require(text.layers.firstIndex { $0.isLinear })
    let full = try #require(text.layers.firstIndex { !$0.isLinear })
    let width = text.config.hiddenSize
    for rows in [1, 8] {
      let x = MLXRandom.normal([1, rows, width]).asType(.float32) * 0.1
      func time(_ label: String, count: Int, _ build: () -> MLXArray) {
        for _ in 0..<2 { eval(build()) }
        let start = Date()
        for _ in 0..<5 { eval(build()) }
        let total = -start.timeIntervalSinceNow / 5 * 1000
        print(String(format: "%d rows  %-28@ %7.2f ms for %d, %.3f each", rows, label, total, count, total / Double(count)))
      }
      time("delta-net layers", count: 48) {
        var h = x
        for _ in 0..<48 {
          h = text.layers[linear](h, mask: nil, cache: GatedDeltaNetCache(), positions: nil, compute: .float16)
        }
        return h
      }
      time("attention layers", count: 16) {
        var h = x
        for _ in 0..<16 {
          let cache = ModelCache(config: text.config).layers[full]
          let mask = rows > 1 ? causalMask(length: rows, offset: 0, dtype: .float16) : nil
          h = text.layers[full](h, mask: mask, cache: cache, positions: nil, compute: .float16)
        }
        return h
      }
    }
  }

  /// The app's own sampling, which is what a chat in the window runs with.
  @Test("drafts sampled decoding at the app's settings")
  func draftsSampled() throws {
    let pack = URL(fileURLWithPath: try #require(environment["ISHIZUKI_PACK"]))
    let model = try BonsaiModel(path: pack)
    let template = try ChatTemplate(path: pack)
    let found = try #require(DFlashDraft.find(for: model.text, beside: pack))
    let draft = try DFlashDraft(directory: found)
    let options = SamplingOptions(temperature: 0.7, minP: 0.05, seed: 5)
    for (label, text) in [
      ("math", "How many positive whole-number divisors does 196 have? Show your reasoning step by step."),
      ("code", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and a short usage example."),
      ("prose", "Explain in a few paragraphs why the sky looks blue during the day and red at sunset."),
    ] {
      let tokens = model.tokenizer.encode(
        try template.render(messages: [.user(text)], addGenerationPrompt: true, enableThinking: false))
      var rates: [Bool: Double] = [:]
      var stats = SpeculativeStats()
      for drafting in [false, true, false, true] {
        model.dflash = drafting ? draft : nil
        let generator = Generator(model: model)
        generator.speculativeDecode = drafting
        let result = generator.generate(promptTokens: tokens, options: options, maxTokens: 256)
        rates[drafting] = max(rates[drafting] ?? 0, result.stats.generationTokensPerSecond)
        if drafting { stats = result.speculative ?? stats }
        #expect(!result.tokens.isEmpty)
      }
      model.dflash = nil
      print(String(format: "%-5@ sampled  plain %6.2f tok/s   dflash %6.2f tok/s   %.2f tokens/verify",
        label, rates[false] ?? 0, rates[true] ?? 0, stats.tokensPerRound))
    }
  }

  /// The server finds the drafter by itself and a chat turn drafts with it.
  @Test("serves a chat turn with the drafter it found")
  func serves() throws {
    let pack = URL(fileURLWithPath: try #require(environment["ISHIZUKI_PACK"]))
    let server = try APIServer(
      directory: pack, samplingOptions: SamplingOptions(temperature: 0.7, minP: 0.05),
      preload: false)
    let model = try server.model()
    #expect(model.dflash != nil, "the server did not attach a drafter")
    let request = APIServer.Request(
      messages: [.user("Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation.")],
      tools: nil, maxTokens: 200, temperature: nil, stream: false, thinking: false, images: [],
      responseSchema: nil, model: nil, effort: nil)
    let start = Date()
    let out = try server.complete(request)
    let seconds = -start.timeIntervalSinceNow
    print(String(format: "served %d tokens in %.1f s (%.1f tok/s with prefill)", out.completionTokens, seconds, Double(out.completionTokens) / seconds))
    print(out.parsed.content.prefix(300))
    #expect(out.completionTokens > 0)
  }
}
