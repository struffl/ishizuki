// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A whole hyper-connected model, against the reference that wrote the fixture.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// The pieces of this architecture are each checked on their own elsewhere. What cannot be
/// checked that way is the assembly: four residual streams carried the depth of the model, an
/// n-gram block folded into them at one layer, no layer norms of the ordinary kind, and a
/// mixer instead of a final norm. Any one of those wired wrongly still produces a plausible
/// tensor of the right shape.
///
/// So the fixture is a tiny random model saved by `Scripts/make_qwen4_exp_fixture.py`, run
/// through the reference implementation in transformers, and what it produced on the way —
/// every layer's streams, not only the last — is what this compares against. Every parameter
/// is random, including the gates and the convolution the reference initialises to zero, so
/// nothing here can pass by being skipped.
@Suite("Qwen4-Exp")
struct Qwen4ExpTests {
  private var fixture: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appending(path: "Fixtures/qwen4-exp")
  }

  /// The pack as the quantizer would lay it out, but still at full width and run at full
  /// width: what is being measured is the architecture, not the packing.
  private func load(into scratch: URL) throws -> (TextModel, [String: MLXArray]) {
    let source = try SourceCheckpoint(directory: fixture)
    _ = try EngramRepack.run(source: source, destination: scratch)

    var arrays: [String: MLXArray] = [:]
    for name in source.tensorNames
    where !name.hasPrefix("model.ngram_embedding.") && !name.hasPrefix("model.ple_embedding.") {
      arrays[name] = TensorNaming.relayout(
        name, try source.tensor(name), zeroCentredNorms: true
      ).asType(.float32)
    }
    let store = try WeightStore(arrays: arrays).openingEngrams(at: scratch)

    let config = try BonsaiConfig.load(directory: fixture)
    let factory = PackedModuleFactory(
      store: store, config: config, tensorPrefix: "", dense: true, activationDType: .float32)
    return (
      try TextModel(config: config, factory: factory, store: store),
      try loadArrays(url: fixture.appending(path: "reference.safetensors"))
    )
  }

  private func scratch() throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory())
      .appending(path: "qwen4-exp-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  @Test("reads the n-gram table the checkpoint describes rather than one it recomputes")
  func readsTheTableAsGiven() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }

    let source = try SourceCheckpoint(directory: fixture)
    let plan = try #require(try EngramRepack.run(source: source, destination: scratch))
    #expect(plan.layout.ngramSize == 3)
    #expect(plan.layout.heads == 4)
    #expect(plan.layout.headsPerNgram == 2)
    #expect(plan.layout.headDim == 8)
    #expect(plan.layout.eosTokenId == 1)
    #expect(plan.layout.multipliers.count == 3)
    // Every head the same size, addresses laid end to end. Upstream derives a distinct prime
    // per head; a derived checkpoint need not, and what it ships is what it was trained with.
    #expect(Set(plan.layout.vocabSizes).count == 1)
    #expect(plan.layout.offsets == (0..<4).map { $0 * plan.layout.vocabSizes[0] })

    let config = try BonsaiConfig.load(directory: fixture)
    try config.validate()
    #expect(config.textConfig.usesHyperConnections)
    #expect(config.textConfig.pleLayer == 1)
  }

  @Test("carries every layer's streams where the reference carries them")
  func matchesEveryLayer() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }
    let (model, reference) = try load(into: scratch)
    let tokens = try #require(reference["tokens"]).asType(.int32).reshaped([1, -1])

    let cache = model.makeCache()
    let rows = model.engramRows(inputs: tokens, cache: cache)
    var streams = model.embedTokens(tokens).asType(.float32)
    streams = concatenated(Array(repeating: streams, count: 4), axis: -1)
    let mask = causalMask(length: streams.dim(1), offset: 0, dtype: .float32)

    for (index, layer) in model.layers.enumerated() {
      streams = layer(
        streams, mask: mask, cache: cache.layers[index], positions: nil, compute: .float32,
        engrams: rows)
      eval(streams)
      let want = try #require(reference["stage_\(index)"])
      #expect(streams.shape == want.shape)
      let worst = (streams - want).abs().max().item(Float.self)
      let scale = want.abs().max().item(Float.self)
      #expect(worst / scale < 1e-2, "layer \(index) drifted by \(worst) against \(scale)")
    }
  }

  @Test("folds the streams back into the width the head reads")
  func matchesTheHead() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }
    let (model, reference) = try load(into: scratch)
    let tokens = try #require(reference["tokens"]).asType(.int32).reshaped([1, -1])
    let want = try #require(reference["hidden"])

    let got = model.hidden(inputs: tokens, cache: model.makeCache())
    eval(got)
    #expect(got.shape == want.shape)
    let worst = (got - want).abs().max().item(Float.self)
    #expect(worst / want.abs().max().item(Float.self) < 2e-2, "the mixer drifted by \(worst)")

    // What the head would actually say, which is the only thing a sampler sees.
    let logits = model.lmHead(got)
    let wanted = try #require(reference["logits"])
    eval(logits)
    let mine = argMax(logits, axis: -1)
    let theirs = argMax(wanted, axis: -1)
    #expect((mine .!= theirs).sum().item(Int32.self) == 0, "the head chose a different token")
  }

  /// Prefill and decode have to agree: the n-gram hashes reach two tokens back, and a step that
  /// forgets them reads a different row of the table than a prefill of the same text would.
  @Test("decodes a token to the same place a prefill of it would reach")
  func decodeMatchesPrefill() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }
    let (model, reference) = try load(into: scratch)
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
      worst / want.abs().max().item(Float.self) < 1e-3,
      "the decoded state drifted by \(worst)")
  }
  /// The pack the quantizer actually writes, loaded the way `ishizuki serve` loads one.
  ///
  /// This is where the checkpoint's own layout has to survive: a bank of experts arrives
  /// fused into one tensor per layer and has to come apart, the n-gram table has to be lifted
  /// out beside the shards rather than into them, and the convolutions and zero-centred norms
  /// have to be relaid out — none of which the dense path above exercises.
  @Test("converts a checkpoint the way it is shipped, and reads the pack back")
  func convertsAndLoads() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }

    let source = try SourceCheckpoint(directory: fixture)
    // The fused bank comes apart into the three projections a router reads.
    #expect(source.has("model.layers.0.mlp.switch_mlp.gate_proj.weight"))
    #expect(source.has("model.layers.0.mlp.switch_mlp.up_proj.weight"))
    #expect(!source.has("model.layers.0.mlp.experts.gate_up_proj"))
    let gate = try source.tensor("model.layers.0.mlp.switch_mlp.gate_proj.weight")
    let fused = try source.tensor("model.layers.0.mlp.experts.gate_up_proj")
    #expect(gate.shape == [fused.dim(0), fused.dim(1) / 2, fused.dim(2)])

    // Eight bits so what this measures is the conversion, not the width.
    let profile = QuantProfile(
      name: "test", baseBits: 8, boostBits: [], targetBpw: 8, groupSize: 32,
      summary: "as close to the checkpoint as a pack gets")
    let outcome = try Quantizer(
      source: source, profile: profile, destination: scratch
    ).run()
    #expect(outcome.byteCount > 0)

    let config = try BonsaiConfig.load(directory: scratch)
    try config.validate()
    let store = try WeightStore(directory: scratch)
      .openingEngrams(at: scratch, capacity: BonsaiRuntime.engramRows)
    #expect(store.engrams != nil)
    // The table travels beside the shards, never inside them.
    #expect(!store.has("model.ngram_embedding.shard_0.weight"))
    #expect(!store.has("model.ple_embedding.ngram_heads_offsets"))

    let factory = PackedModuleFactory(
      store: store, config: config, tensorPrefix: "", dense: false)
    let model = try TextModel(config: config, factory: factory, store: store)

    let reference = try loadArrays(url: fixture.appending(path: "reference.safetensors"))
    let tokens = try #require(reference["tokens"]).asType(.int32).reshaped([1, -1])
    let logits = model(tokens, cache: model.makeCache())
    eval(logits)
    #expect(logits.shape == [1, tokens.dim(1), config.textConfig.vocabSize])

    // An 8-bit pack of a random model will not agree token for token with float32, but it has
    // to stay in the same neighbourhood: a wiring mistake here reads as noise, not as drift.
    let want = try #require(reference["logits"])
    let error = (logits.asType(.float32) - want).square().mean().sqrt().item(Float.self)
    let scale = want.square().mean().sqrt().item(Float.self)
    #expect(error / scale < 0.2, "the pack's logits drifted by \(error) against \(scale)")
  }

  /// A table of any size lands in files of a fixed number of rows, and a row's address has to
  /// survive being cut across them. The shipped tables run to nineteen parts; the fixture's is
  /// one, so the split is driven here rather than left to the only size a test would see.
  @Test("cuts a table across parts and still addresses every row")
  func splitsAcrossParts() throws {
    let flat = try scratch()
    defer { try? FileManager.default.removeItem(at: flat) }

    let source = try SourceCheckpoint(directory: fixture)
    let whole = try #require(try EngramRepack.run(source: source, destination: flat))
    let rows = (0..<whole.layout.totalRows).map { $0 }
    let reference = try EngramStore(
      directory: flat, layout: whole.layout, capacity: rows.count
    ).rows(rows)
    eval(reference)

    for perPart in [64, 97, whole.layout.totalRows - 1] {
      let cut = try scratch()
      defer { try? FileManager.default.removeItem(at: cut) }
      let plan = try #require(
        try EngramRepack.run(source: source, destination: cut, rowsPerPart: perPart))
      #expect(plan.layout.parts == (whole.layout.totalRows + perPart - 1) / perPart)
      #expect(plan.layout.totalRows == whole.layout.totalRows)

      let store = try EngramStore(
        directory: cut, layout: plan.layout, capacity: rows.count)
      let got = try store.rows(rows)
      eval(got)
      #expect(
        (got - reference).abs().max().item(Float.self) == 0,
        "a table in \(plan.layout.parts) parts read back differently")
    }
  }

}
