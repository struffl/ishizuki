// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Queuing the next step before the token is read changes when the host waits, never what it
// gets: the same tokens, and a cache left where the one-step-at-a-time loop leaves it.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

@Suite("Pipelined decode", .serialized)
struct PipelinedDecodeTests {
  private var fixture: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appending(path: "Fixtures/qwen4-exp")
  }

  private static let alphabet = Array(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#")

  private func pack() throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory())
      .appending(path: "pipelined-decode-\(UUID().uuidString)")
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

  private struct Run: Equatable {
    var tokens: [Int]
    var offset: Int
    var stoppedOnEOS: Bool
  }

  private func run(
    _ model: BonsaiModel, pipelined: Bool, options: SamplingOptions = .greedy,
    maxTokens: Int = 12, stopAfter: Int? = nil
  ) throws -> Run {
    BonsaiRuntime.pipelineDecode = pipelined
    defer { BonsaiRuntime.pipelineDecode = true }
    let reference = try loadArrays(url: fixture.appending(path: "reference.safetensors"))
    let prompt = try #require(reference["tokens"]).asType(.int32).asArray(Int32.self).map(Int.init)
    let cache = model.text.makeCache()
    var fragments = 0
    let result = Generator(model: model, politeness: .normal).generate(
      promptTokens: prompt, options: options, maxTokens: maxTokens, cache: cache,
      onToken: { _ in
        fragments += 1
        return stopAfter.map { fragments < $0 } ?? true
      })
    return Run(tokens: result.tokens, offset: cache.offset, stoppedOnEOS: result.stoppedOnEOS)
  }

  @Test("greedy decoding picks the same tokens and leaves the cache in the same place")
  func greedy() throws {
    let pack = try pack()
    defer { try? FileManager.default.removeItem(at: pack) }
    let model = try model(at: pack)

    let serial = try run(model, pipelined: false)
    #expect(serial.tokens.count == 12)
    #expect(try run(model, pipelined: true) == serial)
  }

  @Test("an end token is not fed, even though the step after it was already queued")
  func stopsOnEOS() throws {
    let pack = try pack()
    defer { try? FileManager.default.removeItem(at: pack) }
    let open = try run(try model(at: pack), pipelined: false)
    let index = try #require(
      open.tokens.indices.dropFirst(2).first { !open.tokens[..<$0].contains(open.tokens[$0]) })

    let model = try model(at: pack, eos: open.tokens[index])
    let serial = try run(model, pipelined: false)
    #expect(serial.stoppedOnEOS)
    #expect(serial.tokens == Array(open.tokens[..<index]))
    #expect(try run(model, pipelined: true) == serial)
  }

  @Test("a caller that stops the stream leaves the cache where the serial loop does")
  func callerStops() throws {
    let pack = try pack()
    defer { try? FileManager.default.removeItem(at: pack) }
    let model = try model(at: pack)

    let serial = try run(model, pipelined: false, stopAfter: 4)
    #expect(serial.tokens.count == 4)
    #expect(try run(model, pipelined: true, stopAfter: 4) == serial)
  }

  @Test("sampling with penalties sees the unread token the way the serial loop sees it")
  func sampledWithPenalties() throws {
    let pack = try pack()
    defer { try? FileManager.default.removeItem(at: pack) }
    let model = try model(at: pack)
    let options = SamplingOptions(
      temperature: 0.9, repetitionPenalty: 1.4, repetitionContext: 6, presencePenalty: 0.8,
      seed: 7)

    let serial = try run(model, pipelined: false, options: options, maxTokens: 16)
    #expect(try run(model, pipelined: true, options: options, maxTokens: 16) == serial)
  }
}
