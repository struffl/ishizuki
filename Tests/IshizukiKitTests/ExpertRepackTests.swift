// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A pack split onto disk answers exactly as the pack it was split from.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// The repacker's whole claim is that it moves bytes and changes nothing. So a real pack is
/// built from the fixture, split, and both halves run: same logits, to the last bit, with the
/// routed experts coming off a file a few slots at a time rather than out of memory.
@Suite("Expert repack")
struct ExpertRepackTests {
  private var fixture: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appending(path: "Fixtures/qwen4-exp")
  }

  private func scratch() throws -> URL {
    let url = URL(fileURLWithPath: NSTemporaryDirectory())
      .appending(path: "expert-repack-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  /// An 8-bit pack of the fixture, which is what the repacker takes as its source.
  private func pack(at destination: URL) throws {
    let profile = QuantProfile(
      name: "test", baseBits: 8, boostBits: [], targetBpw: 8, groupSize: 32,
      summary: "as close to the checkpoint as a pack gets")
    _ = try Quantizer(
      source: try SourceCheckpoint(directory: fixture), profile: profile,
      destination: destination
    ).run()
  }

  private func model(at directory: URL, slots: Int?) throws -> (TextModel, WeightStore) {
    let config = try BonsaiConfig.load(directory: directory)
    try config.validate()
    var store = try WeightStore(directory: directory)
      .openingEngrams(at: directory, capacity: BonsaiRuntime.engramRows)
    if let slots { store = try store.openingExperts(at: directory, slots: slots) }
    let factory = PackedModuleFactory(
      store: store, config: config, tensorPrefix: "", dense: false)
    return (try TextModel(config: config, factory: factory, store: store), store)
  }

  private func logits(_ model: TextModel) throws -> MLXArray {
    let reference = try loadArrays(url: fixture.appending(path: "reference.safetensors"))
    let tokens = try #require(reference["tokens"]).asType(.int32).reshaped([1, -1])
    let out = model(tokens, cache: model.makeCache())
    eval(out)
    return out
  }

  @Test("a split pack gives the same logits as the pack it came from")
  func matchesTheResidentPack() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }

    let whole = scratch.appending(path: "whole")
    let split = scratch.appending(path: "split")
    try pack(at: whole)

    let plan = try ExpertRepack.run(source: whole, destination: split)
    #expect(plan.layers.count == 4)
    #expect(plan.expertCount == 4)
    #expect(plan.expertBytes > 0)

    let (resident, residentStore) = try model(at: whole, slots: nil)
    #expect(residentStore.experts(layer: 0) == nil)

    let (streamed, streamedStore) = try model(at: split, slots: 2)
    let store = try #require(streamedStore.experts(layer: 0))
    #expect(store.slotCount == 2)
    #expect(!streamedStore.has("model.layers.0.mlp.switch_mlp.gate_proj.weight"))

    let want = try logits(resident)
    let got = try logits(streamed)
    #expect(got.shape == want.shape)
    #expect((got - want).abs().max().item(Float.self) == 0)
    #expect(store.misses > 0)
  }

  @Test("the split carries everything but the experts across")
  func carriesTheRest() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }

    let whole = scratch.appending(path: "whole")
    let split = scratch.appending(path: "split")
    try pack(at: whole)

    let preview = try ExpertRepack.preview(source: whole)
    #expect(preview.layers.count == 4)
    #expect(preview.expertBytes > 0)
    #expect(preview.residentBytes > 0)

    let plan = try ExpertRepack.run(source: whole, destination: split)
    let fm = FileManager.default
    for layer in plan.layers {
      let url = split.appending(path: ExpertRepack.layerFile(layer))
      #expect(fm.fileExists(atPath: url.path))
      let size =
        ((try? fm.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.intValue ?? 0
      #expect(size == plan.layout.expertCount * plan.layout.stride)
    }
    #expect(ExpertRepack.isSplit(split))
    #expect(fm.fileExists(atPath: split.appending(path: "config.json").path))
    // The n-gram table this architecture streams has to travel with the pack, or the split
    // half loads without the rows every token reads.
    #expect(fm.fileExists(atPath: split.appending(path: EngramLayout.layoutFile).path))

    let bytes = try Data(contentsOf: whole.appending(path: "config.json"))
    #expect(try Data(contentsOf: split.appending(path: "config.json")) == bytes)
  }

  @Test("refuses a pack that is already split, and refuses to write over itself")
  func refusesTheImpossible() throws {
    let scratch = try scratch()
    defer { try? FileManager.default.removeItem(at: scratch) }

    let whole = scratch.appending(path: "whole")
    let split = scratch.appending(path: "split")
    try pack(at: whole)
    _ = try ExpertRepack.run(source: whole, destination: split)

    #expect(throws: BonsaiError.self) {
      _ = try ExpertRepack.run(source: split, destination: scratch.appending(path: "again"))
    }
    #expect(throws: BonsaiError.self) {
      _ = try ExpertRepack.run(source: whole, destination: whole)
    }
  }
}
