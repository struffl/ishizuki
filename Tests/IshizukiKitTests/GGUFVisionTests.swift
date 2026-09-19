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
  private typealias Writer = GGUFFixture.Writer

  private let depth = 2
  private let hidden = 32
  private let intermediate = 64
  private let heads = 2
  private let patch = 16
  private let merge = 2
  private let outHidden = 48
  private let side = 4

  private func floats(_ count: Int, seed: UInt64) -> Data {
    var state = seed
    var out = Data()
    for _ in 0..<count {
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      let value = Float(Int32(bitPattern: UInt32(truncatingIfNeeded: state >> 33))) / 2_147_483_648
      withUnsafeBytes(of: value * 0.1) { out.append(contentsOf: $0) }
    }
    return out
  }

  private func write(to url: URL, projector: String = "qwen3vl", deepstack: [Int]? = nil) throws {
    var writer = Writer()
    writer.metadata = [
      ("clip.has_vision_encoder", 7, Data([1])),
      ("clip.vision.projector_type", 8, Writer.string(projector)),
      ("clip.vision.block_count", 4, Writer.u32(UInt32(depth))),
      ("clip.vision.embedding_length", 4, Writer.u32(UInt32(hidden))),
      ("clip.vision.feed_forward_length", 4, Writer.u32(UInt32(intermediate))),
      ("clip.vision.attention.head_count", 4, Writer.u32(UInt32(heads))),
      ("clip.vision.patch_size", 4, Writer.u32(UInt32(patch))),
      ("clip.vision.image_size", 4, Writer.u32(UInt32(side * patch))),
      ("clip.vision.spatial_merge_size", 4, Writer.u32(UInt32(merge))),
      ("clip.vision.projection_dim", 4, Writer.u32(UInt32(outHidden))),
    ]
    if let deepstack {
      writer.metadata.append(("clip.vision.is_deepstack_layers", 9, GGUFFixture.Builder.intArray(deepstack)))
    }

    var tensors: [(String, [Int], GGMLType, Data)] = [
      // Two Conv2Ds where the checkpoint has one Conv3D.
      ("v.patch_embd.weight", [hidden, 3, patch, patch], .f32,
        floats(hidden * 3 * patch * patch, seed: 1)),
      ("v.patch_embd.weight.1", [hidden, 3, patch, patch], .f32,
        floats(hidden * 3 * patch * patch, seed: 2)),
      ("v.patch_embd.bias", [hidden], .f32, floats(hidden, seed: 3)),
      ("v.position_embd.weight", [side * side, hidden], .f32,
        floats(side * side * hidden, seed: 4)),
      ("v.post_ln.weight", [hidden], .f32, floats(hidden, seed: 5)),
      ("v.post_ln.bias", [hidden], .f32, floats(hidden, seed: 6)),
      ("mm.0.weight", [hidden * merge * merge, hidden * merge * merge], .f32,
        floats(hidden * merge * merge * hidden * merge * merge, seed: 7)),
      ("mm.0.bias", [hidden * merge * merge], .f32, floats(hidden * merge * merge, seed: 8)),
      ("mm.2.weight", [outHidden, hidden * merge * merge], .f32,
        floats(outHidden * hidden * merge * merge, seed: 9)),
      ("mm.2.bias", [outHidden], .f32, floats(outHidden, seed: 10)),
    ]
    for layer in 0..<depth {
      let s = UInt64(layer) * 100 + 20
      tensors += [
        ("v.blk.\(layer).ln1.weight", [hidden], .f32, floats(hidden, seed: s)),
        ("v.blk.\(layer).ln1.bias", [hidden], .f32, floats(hidden, seed: s + 1)),
        ("v.blk.\(layer).ln2.weight", [hidden], .f32, floats(hidden, seed: s + 2)),
        ("v.blk.\(layer).ln2.bias", [hidden], .f32, floats(hidden, seed: s + 3)),
        ("v.blk.\(layer).attn_qkv.weight", [3 * hidden, hidden], .f32,
          floats(3 * hidden * hidden, seed: s + 4)),
        ("v.blk.\(layer).attn_qkv.bias", [3 * hidden], .f32, floats(3 * hidden, seed: s + 5)),
        ("v.blk.\(layer).attn_out.weight", [hidden, hidden], .f32,
          floats(hidden * hidden, seed: s + 6)),
        ("v.blk.\(layer).attn_out.bias", [hidden], .f32, floats(hidden, seed: s + 7)),
        ("v.blk.\(layer).ffn_up.weight", [intermediate, hidden], .f32,
          floats(intermediate * hidden, seed: s + 8)),
        ("v.blk.\(layer).ffn_up.bias", [intermediate], .f32, floats(intermediate, seed: s + 9)),
        ("v.blk.\(layer).ffn_down.weight", [hidden, intermediate], .f32,
          floats(hidden * intermediate, seed: s + 10)),
        ("v.blk.\(layer).ffn_down.bias", [hidden], .f32, floats(hidden, seed: s + 11)),
      ]
    }
    writer.tensors = tensors.map {
      (name: $0.0, dims: $0.1, type: $0.2, payload: $0.3)
    }
    try writer.write(to: url)
  }

  @Test("reads the tower's geometry out of clip's metadata")
  func geometry() throws {
    let url = GGUFFixture.temporaryURL("mmproj")
    defer { try? FileManager.default.removeItem(at: url) }
    try write(to: url)

    let vision = try GGUFVision(file: GGUFFile(url: url))
    #expect(vision.config.depth == depth)
    #expect(vision.config.hiddenSize == hidden)
    #expect(vision.config.intermediateSize == intermediate)
    #expect(vision.config.numHeads == heads)
    #expect(vision.config.patchSize == patch)
    #expect(vision.config.spatialMergeSize == merge)
    #expect(vision.config.outHiddenSize == outHidden)
    // Not written to the file; recovered from image_size, which is how the converter derived it.
    #expect(vision.config.numPositionEmbeddings == side * side)
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
    #expect(embedding.shape == [hidden, 2, patch, patch, 3])

    let tower = try VisionTower(config: vision.config, store: store)
    let grid = (t: 1, h: 4, w: 4)
    let patches = MLXRandom.normal([grid.h * grid.w, 3 * 2 * patch * patch]).asType(.float32)
    let out = tower(patches: patches, grid: grid)
    eval(out)

    // One token per merged block of patches, at the language model's width.
    #expect(out.shape == [grid.h * grid.w / (merge * merge), outHidden])
    #expect(out.asType(.float32).sum().item(Float.self).isFinite)
  }
}
