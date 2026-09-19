// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT
//
// A whole model, loaded from one file and run.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// The end of the GGUF path: architecture, weights and vocabulary out of a single container,
/// through the same modules a pack builds, to logits.
@Suite("GGUF model")
struct GGUFModelTests {
  @Test("loads a whole model out of one file and produces logits")
  func loadsAndRuns() throws {
    let url = GGUFFixture.temporaryURL("model")
    defer { try? FileManager.default.removeItem(at: url) }
    let tiny = GGUFFixture.TinyModel()
    try tiny.write(to: url)

    let model = try BonsaiModel(gguf: url)
    #expect(model.config.textConfig.hiddenSize == tiny.hidden)
    #expect(model.config.textConfig.vocabSize == tiny.vocab)
    #expect(model.tokenizer.eosTokenIds.contains(tiny.vocab - 1))
    #expect(model.mtp == nil)
    #expect(!model.hasVision)
    // The delta-net has to know it is reading llama.cpp's head order, not a checkpoint's.
    #expect(model.store.valueHeadLayout == .tiled)

    let ids = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
    let logits = model.text(ids, cache: nil).asType(.float32)
    eval(logits)

    #expect(logits.shape == [1, 3, tiny.vocab])
    let finite = logits.sum().item(Float.self)
    #expect(finite.isFinite, "logits are \(finite)")
  }

  @Test("gains a tower when the mmproj is handed over with it")
  func loadsTheTower() throws {
    let url = GGUFFixture.temporaryURL("model")
    let projector = GGUFFixture.temporaryURL("mmproj")
    defer {
      try? FileManager.default.removeItem(at: url)
      try? FileManager.default.removeItem(at: projector)
    }
    try GGUFFixture.TinyModel().write(to: url)
    try GGUFFixture.TinyTower().write(to: projector)

    let blind = try BonsaiModel(gguf: url)
    #expect(!blind.hasVision)

    let seeing = try BonsaiModel(gguf: url, mmproj: projector)
    #expect(seeing.hasVision)
    #expect(seeing.config.components?.vision == true)
    #expect(try seeing.vision() != nil)
  }
}
