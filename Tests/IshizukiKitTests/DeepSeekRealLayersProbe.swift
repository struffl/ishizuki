// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The release's first layers, one at a time, against DeepSeek's reference over the same shards.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// `ISHIZUKI_DEEPSEEK_LAYERS` names what the reference wrote: the streams and the next collapse
/// after each layer it ran, without the n-gram blocks, which live in a late shard. Every layer it
/// holds is checked, so a partial download is enough for as far as it goes.
///
/// Each layer is held to the reference on the reference's own input. Carried up from the
/// embedding instead, a few millionths of drift are enough to break a routing tie the other way:
/// on this prompt the last token's sixth and seventh experts at layer 3 are 1.9e-6 apart, and the
/// other one moves that token by 0.1. The carried drift is printed, not bounded.
///
/// `ISHIZUKI_DEEPSEEK_FROM` skips the layers below it, starting at the last global KV source at
/// or below it so every layer after reads the latents and picks it would have.
@Suite(
  "DeepSeek-V4.1 real layers",
  .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_DEEPSEEK_LAYERS"] != nil))
struct DeepSeekRealLayersProbe {
  @Test("matches each real layer given the reference's own input")
  func matchesTheReferenceOnRealLayers() throws {
    let environment = ProcessInfo.processInfo.environment
    let reference = try loadArrays(url: URL(fileURLWithPath: environment["ISHIZUKI_DEEPSEEK_LAYERS"]!))
    let checkpoint = try DeepSeekCheckpoint(
      directory: try #require(DeepSeekRelease.directory), partial: true)
    let weights = DeepSeekWeights(checkpoint: checkpoint, compute: .float32, expertSlots: 64)
    let config = checkpoint.config
    let window = DeepSeekRope(
      dims: config.ropeDim, base: config.ropeTheta, originalContext: 0, factor: 1,
      betaFast: config.betaFast, betaSlow: config.betaSlow)
    let compressed = DeepSeekRope(
      dims: config.ropeDim, base: config.compressRopeTheta, originalContext: config.originalContext,
      factor: config.ropeFactor, betaFast: config.betaFast, betaSlow: config.betaSlow)
    let tokens = reference["tokens"]!.asType(.int32)
    let embedding = try weights.array("embed.weight")
    let s = tokens.dim(1)
    let embedded = broadcast(
      take(embedding, tokens.reshaped([-1]), axis: 0).reshaped([1, s, 1, -1]).asType(.float32),
      to: [1, s, config.hcMult, config.dim])
    let identity = SinkhornMixer.identity(batch: 1, length: s, copies: config.hcMult)
    let caches = (0..<config.layers).map { _ in DeepSeekLayerCache() }
    let selection = DeepSeekSelection()
    let from = Int(environment["ISHIZUKI_DEEPSEEK_FROM"] ?? "") ?? 0
    let first = config.kvSourceLayers.filter { $0 <= from }.max() ?? 0
    func input(_ layer: Int) -> (streams: MLXArray, pre: MLXArray) {
      layer == 0
        ? (embedded, identity) : (reference["stage_\(layer - 1)"]!, reference["pre_\(layer - 1)"]!)
    }
    var carried = input(first)
    let carriedCaches = (0..<config.layers).map { _ in DeepSeekLayerCache() }
    let carriedSelection = DeepSeekSelection()
    let layers = reference.keys.filter { $0.hasPrefix("stage_") }.count
    for layer in first..<layers {
      let block = try DeepSeekBlock(
        layer: layer, prefix: "layers.\(layer)", config: config, weights: weights,
        rope: config.compressRatio(layer: layer) > 0 ? compressed : window, engram: false)
      let forced = input(layer)
      let (streams, pre) = block(
        forced.streams, pre: forced.pre, caches: caches, selection: selection, start: 0,
        compute: .float32)
      carried = block(
        carried.streams, pre: carried.pre, caches: carriedCaches, selection: carriedSelection,
        start: 0, compute: .float32)
      eval(streams, pre, carried.streams, carried.pre)
      let want = reference["stage_\(layer)"]!
      let scale = want.abs().max().item(Float.self)
      let worst = (streams - want).abs().max().item(Float.self) / scale
      let drift = (carried.streams - want).abs().max().item(Float.self) / scale
      let preWorst = (pre - reference["pre_\(layer)"]!).abs().max().item(Float.self)
      print("layer \(layer): streams \(worst) of \(scale), pre \(preWorst), carried \(drift)")
      #expect(worst < 1e-4, "layer \(layer) drifted by \(worst) on the reference's own input")
    }
  }
}
