// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Drafting inside a served turn changes how many tokens a forward pass yields, never which
// ones, and never leaves the cache ahead of what was emitted.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

@Suite("Speculative generation", .serialized)
struct SpeculativeGenerationTests {
  private var fixture: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appending(path: "Fixtures/qwen4-exp")
  }

  private static let alphabet = Array(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#")

  private func pack() throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory())
      .appending(path: "speculative-generation-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    let profile = QuantProfile(
      name: "test", baseBits: 8, boostBits: [], targetBpw: 8, groupSize: 32,
      summary: "as close to the checkpoint as a pack gets")
    _ = try Quantizer(
      source: try SourceCheckpoint(directory: fixture), profile: profile, destination: url
    ).run()
    return url
  }

  private func model(at pack: URL, eos: Int? = nil) throws -> BonsaiModel {
    var vocab: [String: Int] = [:]
    for (id, character) in Self.alphabet.enumerated() where id != eos {
      vocab[String(character)] = id
    }
    var root: [String: Any] = ["model": ["type": "BPE", "vocab": vocab, "merges": [String]()]]
    if let eos { root["added_tokens"] = [["id": eos, "content": "<|im_end|>"]] }
    try JSONSerialization.data(withJSONObject: root)
      .write(to: pack.appending(path: "tokenizer.json"))
    return try BonsaiModel(directory: pack)
  }

  private final class Oracle: Drafter {
    let truth: [Int]
    let spoil: Int
    private var rounds = 0

    init(truth: [Int], spoil: Int) {
      self.truth = truth
      self.spoil = spoil
    }

    func propose(context: [Int], count: Int) -> [Int] {
      rounds += 1
      let from = context.count
      guard from < truth.count else { return [] }
      var draft = Array(truth[from..<min(from + count, truth.count)])
      if rounds % 3 == 0, draft.count > spoil { draft[spoil] = (draft[spoil] + 1) % 64 }
      return draft
    }

    func commit(tokens: [Int]) {}
    func reset() {}
  }

  private struct Run {
    var result: GenerationResult
    var prompt: Int
    var offset: Int
  }

  private func run(
    _ model: BonsaiModel, drafting: Bool, options: SamplingOptions = .greedy,
    maxTokens: Int = 40, stopAfter: Int? = nil, oracle: [Int]? = nil
  ) throws -> Run {
    let saved = BonsaiRuntime.speculativeDecode
    BonsaiRuntime.speculativeDecode = drafting
    defer { BonsaiRuntime.speculativeDecode = saved }
    let reference = try loadArrays(url: fixture.appending(path: "reference.safetensors"))
    let seed = try #require(reference["tokens"]).asType(.int32).asArray(Int32.self).map(Int.init)
    let prompt = seed + seed + seed
    let cache = model.text.makeCache()
    var fragments = 0
    let generator = Generator(model: model, politeness: .normal)
    if let oracle { generator.lookup = { Oracle(truth: prompt + oracle, spoil: 2) } }
    let result = generator.generate(
      promptTokens: prompt, options: options, maxTokens: maxTokens, cache: cache,
      onToken: { _ in
        fragments += 1
        return stopAfter.map { fragments < $0 } ?? true
      })
    return Run(result: result, prompt: prompt.count, offset: cache.offset)
  }

  @Test("greedy drafting emits the tokens plain decoding does, in fewer rounds")
  func greedyMatches() throws {
    let pack = try pack()
    defer { try? FileManager.default.removeItem(at: pack) }
    let model = try model(at: pack)

    let plain = try run(model, drafting: false)
    let drafted = try run(model, drafting: true, oracle: plain.result.tokens)
    #expect(drafted.result.tokens == plain.result.tokens)
    let stats = try #require(drafted.result.speculative)
    #expect(stats.rounds < drafted.result.tokens.count / 2, "\(stats)")
    #expect(stats.rollbacks > 0, "no draft was ever spoiled, so no rollback was checked")
    #expect(drafted.offset <= drafted.prompt + drafted.result.tokens.count)
  }

  @Test("prompt lookup alone still decodes what plain decoding does")
  func lookupMatches() throws {
    let pack = try pack()
    defer { try? FileManager.default.removeItem(at: pack) }
    let model = try model(at: pack)

    let plain = try run(model, drafting: false)
    let drafted = try run(model, drafting: true)
    #expect(drafted.result.tokens == plain.result.tokens)
    #expect(drafted.offset <= drafted.prompt + drafted.result.tokens.count)
  }

  @Test("an end token inside an accepted block stops there, and the cache stays behind it")
  func stopsOnEOS() throws {
    let pack = try pack()
    defer { try? FileManager.default.removeItem(at: pack) }
    let open = try run(try model(at: pack), drafting: false).result.tokens
    let index = try #require(
      open.indices.dropFirst(4).first { !open[..<$0].contains(open[$0]) })

    let model = try model(at: pack, eos: open[index])
    let drafted = try run(model, drafting: true, oracle: open)
    #expect(drafted.result.stoppedOnEOS)
    #expect(drafted.result.tokens == Array(open[..<index]))
    #expect(drafted.offset <= drafted.prompt + drafted.result.tokens.count)
  }

  @Test("a caller stopping mid-block is honoured at the token it stopped on")
  func callerStops() throws {
    let pack = try pack()
    defer { try? FileManager.default.removeItem(at: pack) }
    let model = try model(at: pack)

    let plain = try run(model, drafting: false, stopAfter: 7)
    let drafted = try run(model, drafting: true, stopAfter: 7, oracle: plain.result.tokens)
    #expect(drafted.result.tokens == plain.result.tokens)
    #expect(drafted.offset <= drafted.prompt + drafted.result.tokens.count)
  }

  @Test("sampled drafting runs to its limit and keeps the cache behind what it emitted")
  func sampled() throws {
    let pack = try pack()
    defer { try? FileManager.default.removeItem(at: pack) }
    let model = try model(at: pack)

    let plain = try run(model, drafting: false)
    let drafted = try run(
      model, drafting: true, options: SamplingOptions(temperature: 0.8, minP: 0.05, seed: 11),
      oracle: plain.result.tokens)
    #expect(drafted.result.tokens.count == 40 || drafted.result.stoppedOnEOS)
    #expect(drafted.offset <= drafted.prompt + drafted.result.tokens.count)
    #expect((drafted.result.speculative?.proposed ?? 0) > 0)
  }
}
