// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Decoding a quantized pack, token by token, the way a served turn does.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// The dense path is checked for this elsewhere, and a pack is checked for one prefill. Neither
/// covers what a served turn actually does: load a pack, prefill a prompt, then step a token at
/// a time against the state the prefill left behind.
///
/// That gap is where a rank-2 slice in the PLE convolution's carried state trapped the whole
/// process rather than failing a request. The pack is built once and stepped a handful of
/// times, so this stays a fraction of a second.
@Suite("Packed decode")
struct PackedDecodeTests {
  private var fixture: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appending(path: "Fixtures/qwen4-exp")
  }

  private func scratch() throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory())
      .appending(path: "packed-decode-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  private func pack(at destination: URL) throws -> TextModel {
    let profile = QuantProfile(
      name: "test", baseBits: 8, boostBits: [], targetBpw: 8, groupSize: 32,
      summary: "as close to the checkpoint as a pack gets")
    _ = try Quantizer(
      source: try SourceCheckpoint(directory: fixture), profile: profile,
      destination: destination
    ).run()

    let config = try BonsaiConfig.load(directory: destination)
    try config.validate()
    let store = try WeightStore(directory: destination)
      .openingEngrams(at: destination, capacity: BonsaiRuntime.engramRows)
    let factory = PackedModuleFactory(
      store: store, config: config, tensorPrefix: "", dense: false)
    return try TextModel(config: config, factory: factory, store: store)
  }

  @Test("a pack steps past its prompt to the same place a prefill of it reaches")
  func decodeMatchesPrefillOnAPack() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }
    let model = try pack(at: scratch)

    let reference = try loadArrays(url: fixture.appending(path: "reference.safetensors"))
    let tokens = try #require(reference["tokens"]).asType(.int32).reshaped([1, -1])

    let whole = model.hidden(inputs: tokens, cache: model.makeCache())
    let cache = model.makeCache()
    var stepped: MLXArray?
    for index in 0..<tokens.dim(1) {
      stepped = model.hidden(inputs: tokens[0..., index..<(index + 1)], cache: cache)
    }
    let last = try #require(stepped)
    eval(whole, last)

    let want = whole[0..., -1, 0...].reshaped([-1])
    let worst = (last.reshaped([-1]) - want).abs().max().item(Float.self)
    #expect(
      worst / want.abs().max().item(Float.self) < 1e-2,
      "the decoded state drifted by \(worst)")
  }

  /// Generation proper: the prompt is prefilled whole and the answer is stepped on top of it,
  /// which is the order that leaves a carried convolution state behind to be indexed.
  @Test("and keeps stepping after a whole-prompt prefill")
  func stepsAfterAPrefill() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }
    let model = try pack(at: scratch)

    let reference = try loadArrays(url: fixture.appending(path: "reference.safetensors"))
    let tokens = try #require(reference["tokens"]).asType(.int32).reshaped([1, -1])

    let cache = model.makeCache()
    var logits = model(tokens, cache: cache)
    eval(logits)

    for _ in 0..<4 {
      let next = logits[0..., -1, 0...].argMax(axis: -1).reshaped([1, 1]).asType(.int32)
      eval(next)
      logits = model(next, cache: cache)
      eval(logits)
      #expect(logits.dim(1) == 1)
    }
  }
}
