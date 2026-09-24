// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DSpark, V4.1's draft head: five tokens proposed in one pass, each with a guess at its odds.

import Foundation
import MLX
import MLXFast

/// Three small blocks that read what the backbone's last layers read and propose the next
/// `blockSize` tokens at once.
///
/// The drafts are a block of noise tokens after the one the backbone just chose, run together
/// and seeing each other both ways, over a window made from the backbone's own positions rather
/// than from earlier drafts. What that pass cannot see — how each draft follows the one before
/// it — a low-rank Markov head adds back as a bias on each position's logits, one position at a
/// time. A confidence head then says how likely each draft is to survive verification.
public final class DeepSeekDraft: @unchecked Sendable {
  let blocks: [DeepSeekBlock]
  let mainProjection: any Projection
  let mainNorm: MLXArray
  let norm: MLXArray
  let markovEmbedding: MLXArray
  let markovHead: any Projection
  let confidence: any Projection
  public let blockSize: Int
  public let noiseToken: Int
  let copies: Int
  let eps: Float
  let compute: DType

  init(weights: DeepSeekWeights, rope: DeepSeekRope, compute: DType) throws {
    let config = weights.config
    self.blocks = try (0..<config.draftLayers).map { stage in
      try DeepSeekBlock(
        layer: config.layers + stage, prefix: "mtp.\(stage)", config: config, weights: weights,
        rope: rope)
    }
    let last = "mtp.\(config.draftLayers - 1)"
    self.mainProjection = try weights.linear("mtp.0.main_proj")
    self.mainNorm = try weights.array("mtp.0.main_norm.weight")
    self.norm = try weights.array(last + ".norm.weight")
    self.markovEmbedding = try weights.array(last + ".markov_head.embed.weight")
    self.markovHead = try weights.linear(last + ".markov_head.head", wide: true)
    self.confidence = try weights.linear(last + ".confidence_head.proj", wide: true)
    self.blockSize = config.draftBlockSize
    self.noiseToken = config.draftNoiseTokenId
    self.copies = config.hcMult
    self.eps = config.normEps
    self.compute = compute
  }

  var fakeQuant: Bool = true {
    didSet { for block in blocks { block.attention.fakeQuant = fakeQuant } }
  }

  public func makeCaches() -> [DeepSeekLayerCache] { blocks.map { _ in DeepSeekLayerCache() } }

  /// What the backbone read at its target layers, projected into the draft's width.
  func main(_ hidden: MLXArray) -> MLXArray {
    MLXFast.rmsNorm(
      mainProjection(hidden.asType(compute)), weight: mainNorm.asType(compute), eps: eps)
  }

  /// Lays the backbone's positions `start ..< start + n` into every draft layer's window.
  public func observe(_ hidden: MLXArray, caches: [DeepSeekLayerCache], start: Int) {
    let projected = main(hidden)
    for (block, cache) in zip(blocks, caches) {
      block.attention.observe(projected, cache: cache, start: start)
    }
  }

  public struct Proposal: @unchecked Sendable {
    /// The token the block started from, then one draft per position.
    public var tokens: [Int]
    public var logits: MLXArray
    public var confidence: MLXArray
  }

  /// Drafts the `blockSize` tokens after `token`, which sits at `position`. `pick` turns a
  /// position's logits into its token; each pick feeds the Markov bias of the next position.
  public func propose(
    after token: Int, at position: Int, caches: [DeepSeekLayerCache], model: DeepSeekModel,
    pick: (MLXArray) -> Int
  ) -> Proposal {
    let ids = MLXArray([Int32(token)] + Array(repeating: Int32(noiseToken), count: blockSize - 1))
      .reshaped([1, blockSize])
    let embedded = model.embed(ids).asType(.float32)
    var streams = broadcast(
      embedded.expandedDimensions(axis: 2), to: [1, blockSize, copies, embedded.dim(-1)])
    var pre = SinkhornMixer.identity(batch: 1, length: blockSize, copies: copies)
    let selection = DeepSeekSelection()
    for (block, cache) in zip(blocks, caches) {
      (streams, pre) = block(
        streams, pre: pre, caches: [cache], selection: selection, start: position,
        compute: compute, draft: true)
    }
    let collapsed = SinkhornMixer.collapse(streams, pre: pre).asType(compute)
    let hidden = MLXFast.rmsNorm(collapsed, weight: norm.asType(compute), eps: eps)
    var logits = model.logits(hidden)

    var tokens = [token]
    var biases: [MLXArray] = []
    var markov: [MLXArray] = []
    for index in 0..<blockSize {
      let embedding = markovEmbedding[tokens[index]].reshaped([1, -1])
      let bias = markovHead(embedding.asType(.float32)).asType(.float32)
      markov.append(embedding)
      biases.append(bias)
      let row = logits[0, index].reshaped([1, -1]) + bias
      tokens.append(pick(row))
    }
    logits = logits + concatenated(biases, axis: 0).expandedDimensions(axis: 0)
    let joined = concatenated(
      [collapsed.asType(.float32), stacked(markov, axis: 1).asType(.float32)], axis: -1)
    let odds = confidence(joined).squeezed(axis: -1)
    return Proposal(tokens: tokens, logits: logits, confidence: odds)
  }
}
