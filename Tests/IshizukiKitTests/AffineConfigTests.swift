// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import Testing

@testable import IshizukiKit

// The fixture is the published config of an oQ4e pack: a Qwen3.5 hybrid with a vision tower,
// one MTP layer, and an imatrix pass that lifted 166 modules off the pack-wide 4-bit width.
@Suite("Affine packs")
struct AffineConfigTests {
  private func oq4e() throws -> BonsaiConfig {
    let url = try #require(
      Bundle.module.url(forResource: "Fixtures/oq4e-config", withExtension: "json"))
    let dir = URL(filePath: NSTemporaryDirectory())
      .appending(path: "oq4e-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    try FileManager.default.copyItem(at: url, to: dir.appending(path: "config.json"))
    return try BonsaiConfig.load(directory: dir)
  }

  @Test("the nested towers survive the adapter")
  func towers() throws {
    let config = try oq4e()
    #expect(config.modelType == "qwen3_5")
    #expect(config.textConfig.hiddenSize == 5120)
    #expect(config.textConfig.numHiddenLayers == 64)
    #expect(config.textConfig.vocabSize == 248_320)
    #expect(config.textConfig.headDim == 256)
    #expect(config.textConfig.tieWordEmbeddings == false)
    #expect(config.visionConfig?.depth == 27)
    #expect(config.imageTokenId == 248_056)
  }

  @Test("the hybrid layer schedule is read off layer_types")
  func schedule() throws {
    let full = try oq4e().textConfig.isFullAttention
    #expect(full.count == 64)
    #expect(full.filter { $0 }.count == 16)
    #expect(full[3] && !full[2])
  }

  @Test("rope is read through MLX's spelling of the family")
  func rope() throws {
    let rope = try oq4e().textConfig.ropeParameters
    #expect(rope.ropeType == "default")
    #expect(rope.ropeTheta == 10_000_000)
    #expect(rope.mropeSection == [11, 11, 10])
    #expect(rope.mropeInterleaved == true)
    #expect(try oq4e().textConfig.ropeDimensions == 64)
  }

  @Test("the imatrix overrides land on the modules they name")
  func overrides() throws {
    let quant = try oq4e().quantization
    #expect(quant.bits == 4)
    #expect(quant.groupSize == 64)
    #expect(quant.overrides.count == 166)
    #expect(quant.widths == [4, 5])
    #expect(
      quant.module("language_model.model.layers.0.mlp.down_proj")
        == BonsaiConfig.ModuleQuant(bits: 5, groupSize: 64))
    // A module the pass left alone falls through to the pack-wide width.
    #expect(
      quant.module("language_model.model.layers.0.mlp.up_proj")
        == BonsaiConfig.ModuleQuant(bits: 4, groupSize: 64))
  }

  @Test("the components the pack ships are inferred from the config")
  func components() throws {
    let config = try oq4e()
    #expect(config.components?.vision == true)
    #expect(config.components?.mtp == true)
    #expect(config.profile == .affine)
  }

  @Test("a mixed-width affine pack validates")
  func validates() throws {
    #expect(throws: Never.self) { try oq4e().validate() }
  }
}
