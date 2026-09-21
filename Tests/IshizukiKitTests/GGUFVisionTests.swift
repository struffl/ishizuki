// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The tower out of llama.cpp's separate mmproj file.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

/// A picture costs two files in this format. The names differ from the checkpoint's throughout,
/// and the patch embedding is stored as two Conv2Ds because ggml has no Conv3D — so the test
/// that matters is whether the real tower runs off it, not whether the strings map.
@Suite("GGUF vision")
struct GGUFVisionTests {
  private let tower = GGUFFixture.TinyTower()

  private func write(to url: URL, projector: String = "qwen3vl", deepstack: [Int]? = nil) throws {
    var fixture = tower
    fixture.projector = projector
    fixture.deepstack = deepstack
    try fixture.write(to: url)
  }

  @Test("reads the tower's geometry out of clip's metadata")
  func geometry() throws {
    let url = GGUFFixture.temporaryURL("mmproj")
    defer { try? FileManager.default.removeItem(at: url) }
    try write(to: url)

    let vision = try GGUFVision(file: GGUFFile(url: url))
    #expect(vision.config.depth == tower.depth)
    #expect(vision.config.hiddenSize == tower.hidden)
    #expect(vision.config.intermediateSize == tower.intermediate)
    #expect(vision.config.numHeads == tower.heads)
    #expect(vision.config.patchSize == tower.patch)
    #expect(vision.config.spatialMergeSize == tower.merge)
    #expect(vision.config.outHiddenSize == tower.outHidden)
    // Not written to the file; recovered from image_size, which is how the converter derived it.
    #expect(vision.config.numPositionEmbeddings == tower.side * tower.side)
    #expect(vision.config.inChannels == 3)
    #expect(vision.config.temporalPatchSize == 2)
  }

  @Test("refuses a projector or a tower it cannot run")
  func refusesUnsupported() throws {
    let url = GGUFFixture.temporaryURL("mmproj")
    defer { try? FileManager.default.removeItem(at: url) }

    try write(to: url, projector: "llava")
    #expect(throws: BonsaiError.self) { _ = try GGUFVision(file: GGUFFile(url: url)) }

    // A deepstack tower feeds intermediate blocks back into the language model; dropping those
    // tensors would load cleanly and change what the model sees.
    try write(to: url, deepstack: [0, 1])
    #expect(throws: BonsaiError.self) { _ = try GGUFVision(file: GGUFFile(url: url)) }

    try write(to: url, deepstack: [0, 0])
    #expect(throws: Never.self) { _ = try GGUFVision(file: GGUFFile(url: url)) }
  }

  @Test("the real tower runs off the mmproj's names and its split patch embedding")
  func runsTheTower() throws {
    let url = GGUFFixture.temporaryURL("mmproj")
    defer { try? FileManager.default.removeItem(at: url) }
    try write(to: url)

    let file = try GGUFFile(url: url)
    let vision = try GGUFVision(file: file)
    let store = try GGUFWeights.loadVision(
      file: file, into: WeightStore(arrays: [:]), dtype: .float32)

    // The two Conv2Ds fused, then laid out channels-last like every other path.
    let embedding = try store(GGUFVisionNaming.prefix + "patch_embed.proj.weight")
    #expect(embedding.shape == [tower.hidden, 2, tower.patch, tower.patch, 3])

    let built = try VisionTower(config: vision.config, store: store)
    let grid = (t: 1, h: 4, w: 4)
    let patches = MLXRandom.normal([grid.h * grid.w, 3 * 2 * tower.patch * tower.patch]).asType(
      .float32)
    let out = built(patches: patches, grid: grid)
    eval(out)

    // One token per merged block of patches, at the language model's width.
    #expect(out.shape == [grid.h * grid.w / (tower.merge * tower.merge), tower.outHidden])
    #expect(out.asType(.float32).sum().item(Float.self).isFinite)
  }
}
