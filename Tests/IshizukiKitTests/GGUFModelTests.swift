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

  @Test("finds the tower beside the model when handed only a path")
  func pairsTheProjector() throws {
    let root = URL(filePath: NSTemporaryDirectory())
      .appending(path: "pair-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    let url = root.appending(path: "Tiny-IQ4_XS.gguf")
    try GGUFFixture.TinyModel().write(to: url)
    #expect(try BonsaiModel(path: url).hasVision == false)

    // The layout every publisher of these files uses, and the one `pull --file` writes.
    try GGUFFixture.TinyTower().write(to: root.appending(path: "mmproj-Tiny-BF16.gguf"))
    #expect(try BonsaiModel(path: url).hasVision)
  }

  @Test("serves a GGUF through the same server a pack goes through")
  func servesIt() throws {
    let root = URL(filePath: NSTemporaryDirectory())
      .appending(path: "serve-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    var fixture = GGUFFixture.TinyModel()
    fixture.chatTemplate = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
    let url = root.appending(path: "Tiny-IQ4_XS.gguf")
    try fixture.write(to: url)

    // The template comes out of the metadata rather than a file beside the weights.
    let template = try ChatTemplate(path: url)
    let rendered = try template.render(
      messages: [ChatMessage(role: "user", content: .text("hello"))],
      addGenerationPrompt: false)
    #expect(rendered.contains("user: hello"))

    let catalog = ModelCatalog.discover(in: [root])
    let entry = try #require(catalog["Tiny-IQ4_XS"])
    #expect(entry.format == .gguf)
    // What the server sizes its budget against: the file is its own weights.
    #expect(MemoryBudget.weightBytes(in: entry.url) == entry.byteCount)

    let server = try APIServer(directory: entry.url, preload: false)
    #expect(server.modelPath == entry.url)
    #expect(try server.model().config.textConfig.vocabSize == fixture.vocab)
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
