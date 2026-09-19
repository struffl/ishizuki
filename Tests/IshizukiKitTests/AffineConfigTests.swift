// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom
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

  @Test("a width that does not divide 32 still reports its true input size")
  func inputWidth() throws {
    // Shapes taken from the oQ4e pack: down_proj is 5-bit over a 17408-wide input, which packs
    // into 2720 words. Counting values per word would make that 16320.
    let quant = MLXArray.zeros([5120, 2720], dtype: .uint32)
    let scales = MLXArray.zeros([5120, 272], dtype: .float16)
    let linear = try PackedLinear(
      weight: quant, scales: scales, biases: scales, signs: nil, block: 0,
      groupSize: 64, bits: 5)
    #expect(linear.inputDim == 17408)
    #expect(linear.outputDim == 5120)
  }

  @Test("the widths an imatrix pack mixes all run")
  func widthsExecute() {
    // oQ3e and oQ3.5e sit on a 3-bit base with 4- and 5-bit boosts, so all three have to
    // survive a quantized matmul, not merely be accepted by the config.
    let rows = 256
    let width = 512
    let x = MLXRandom.normal([1, width]).asType(.float16)
    let w = MLXRandom.normal([rows, width]).asType(.float16)
    let reference = matmul(x, w.T).asType(.float32)

    for bits in [3, 4, 5] {
      let (wq, scales, biases) = quantized(w, groupSize: 64, bits: bits, mode: .affine)
      let y = quantizedMM(
        x, wq, scales: scales, biases: biases, transpose: true, groupSize: 64,
        bits: bits, mode: .affine
      ).asType(.float32)
      let error = (abs(y - reference).mean() / abs(reference).mean()).item(Float.self)
      #expect(error < 0.35, "\(bits)-bit relative error \(error)")
    }
  }

  @Test("a mixed-width affine pack validates")
  func validates() throws {
    #expect(throws: Never.self) { try oq4e().validate() }
  }
}
