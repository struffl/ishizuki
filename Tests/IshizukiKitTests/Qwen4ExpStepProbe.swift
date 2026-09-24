// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import IshizukiKit

// Probe, not a test: one hyper-connected decode step, split into its components on the real weights.
@Suite(
  "Qwen4ExpStepProbe",
  .enabled(if: ProcessInfo.processInfo.environment["ISHIZUKI_STEP_PACK"] != nil))
struct Qwen4ExpStepProbe {
  static func field<T>(_ value: Any, _ name: String) -> T? {
    Mirror(reflecting: value).children.first { $0.label == name }?.value as? T
  }

  static func best(_ runs: Int = 5, _ body: () -> Void) -> Double {
    body()
    var best = Double.infinity
    for _ in 0..<runs {
      let start = Date()
      body()
      best = min(best, -start.timeIntervalSinceNow)
    }
    return best * 1000
  }

  @Test("probe")
  func probe() throws {
    let path = ProcessInfo.processInfo.environment["ISHIZUKI_STEP_PACK"]!
    let steps = Int(ProcessInfo.processInfo.environment["ISHIZUKI_STEPS"] ?? "") ?? 32
    let model = try BonsaiModel(path: URL(filePath: path))
    let text: TextModel = model.text
    let prompt = model.tokenizer.encode("Write a short story about a lighthouse keeper.")

    func prefilled() -> (ModelCache, MLXArray) {
      let cache = text.makeCache()
      let ids = MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])
      let logits = text(ids, cache: cache)
      eval(logits)
      return (cache, logits[0..., -1, 0...].argMax(axis: -1))
    }

    var emitted: [Int32] = []
    func serial() -> Double {
      var (cache, next) = prefilled()
      emitted = []
      let start = Date()
      for _ in 0..<steps {
        let token = next.item(Int32.self)
        emitted.append(token)
        let step = text(MLXArray([token]).reshaped([1, 1]), cache: cache)
        eval(step)
        next = step[0..., -1, 0...].argMax(axis: -1)
      }
      return -start.timeIntervalSinceNow * 1000 / Double(steps)
    }

    func pipelined() -> Double {
      var (cache, next) = prefilled()
      let start = Date()
      for _ in 0..<steps {
        let step = text(next.reshaped([1, 1]), cache: cache)
        let following = step[0..., -1, 0...].argMax(axis: -1)
        asyncEval(following)
        _ = next.item(Int32.self)
        next = following
      }
      return -start.timeIntervalSinceNow * 1000 / Double(steps)
    }

    func graphBuild() -> Double {
      let (cache, next) = prefilled()
      let token = next.item(Int32.self)
      let start = Date()
      let step = text(MLXArray([token]).reshaped([1, 1]), cache: cache)
      let seconds = -start.timeIntervalSinceNow * 1000
      eval(step)
      return seconds
    }

    _ = serial()
    let serialMs = min(serial(), serial())
    let pipeMs = min(pipelined(), pipelined())
    let buildMs = (0..<3).map { _ in graphBuild() }.min()!
    print(String(format: "serial    : %7.2f ms/step  %6.2f tok/s", serialMs, 1000 / serialMs))
    print("greedy tokens: \(emitted.map(String.init).joined(separator: " "))")
    print(String(format: "pipelined : %7.2f ms/step  %6.2f tok/s", pipeMs, 1000 / pipeMs))
    print(String(format: "graph build, host only, incl. engram fetch: %7.2f ms", buildMs))

    let (cache, next) = prefilled()
    let token = MLXArray([next.item(Int32.self)]).reshaped([1, 1])
    let fetchMs = Self.best { _ = text.engramRows(inputs: token, cache: cache) }
    print(String(format: "engram fetch alone (hash + read + eval): %7.2f ms", fetchMs))

    let compute = text.embedTokens(token).dtype
    let width = text.config.hiddenSize
    let count = text.config.hcCount ?? 1
    let x = MLXRandom.normal([1, 1, width]).asType(compute)
    let streams = MLXRandom.normal([1, 1, width * count]).asType(.float32)
    let engrams = text.engramRows(inputs: token, cache: cache)!
    eval(x, streams, engrams)

    struct Parts {
      var residuals: [GatedResidual] = []
      var gdn: [(GatedDeltaNet, GatedDeltaNetCache?)] = []
      var attention: [(Attention, AttentionKVCache?)] = []
      var feeds: [any FeedForward] = []
      var ple: [PLEBlock] = []
    }
    var parts = Parts()
    for (index, layer) in text.layers.enumerated() {
      let layerCache = cache.layers[index]
      if let r: GatedResidual = Self.field(layer, "attnResidual") { parts.residuals.append(r) }
      if let r: GatedResidual = Self.field(layer, "mlpResidual") { parts.residuals.append(r) }
      if let g: GatedDeltaNet = Self.field(layer, "linearAttention") {
        parts.gdn.append((g, layerCache as? GatedDeltaNetCache))
      }
      if let a: Attention = Self.field(layer, "selfAttention") {
        parts.attention.append((a, layerCache as? AttentionKVCache))
      }
      if let f: any FeedForward = Self.field(layer, "mlp") { parts.feeds.append(f) }
      if let p: PLEBlock = Self.field(layer, "ple") { parts.ple.append(p) }
    }
    if let mixer: GatedResidual = Self.field(text, "mixer") { parts.residuals.append(mixer) }

    func group(_ label: String, _ n: Int, _ build: () -> [MLXArray]) -> Double {
      let ms = Self.best { eval(build()) }
      print(String(format: "  %-34@ x%-3d %7.2f ms  (%5.3f each)", label, n, ms, ms / Double(max(n, 1))))
      return ms
    }

    print("components, each group run alone with no host sync inside it:")
    var total = 0.0
    total += group("embedding", 1) { [text.embedTokens(token)] }
    total += group("hyper-connection open", parts.residuals.count) {
      parts.residuals.flatMap { r -> [MLXArray] in
        let o = r(streams)
        return [o.mixed] + (o.injection.map { [$0] } ?? [])
      }
    }
    total += group("hyper-connection close", parts.residuals.count) {
      parts.residuals.map { r in
        GatedResidual.close(
          GatedResidual.Opened(
            mixed: x, streams: streams,
            injection: MLXRandom.normal([1, 1, count]).asType(.float32)),
          with: x.asType(.float32))
      }
    }
    total += group("gated delta net", parts.gdn.count) { parts.gdn.map { $0.0(x, cache: $0.1) } }
    total += group("full attention", parts.attention.count) {
      parts.attention.map { $0.0(x, mask: nil, cache: $0.1, positions: nil) }
    }
    total += group("moe feed-forward", parts.feeds.count) { parts.feeds.map { $0(x) } }
    total += group("ple block", parts.ple.count) {
      parts.ple.map { p in
        var state: MLXArray? = nil
        return p(engrams, streams: streams, state: &state)
      }
    }
    total += group("lm head", 1) { [text.lmHead(x)] }
    print(String(format: "  sum of components: %7.2f ms  against serial step %7.2f ms", total, serialMs))

    print("hyper-connection open, piece by piece, over every instance:")
    let rs = parts.residuals
    let normed = rs[0].normalized(streams)
    let low = rs[0].down(normed)
    eval(normed, low)
    print("  normed \(normed.dtype) \(normed.shape), down out \(low.shape), up out \(rs[0].up(silu(low)).shape)")
    _ = group("grouped rms norm", rs.count) { rs.map { $0.normalized(streams) } }
    _ = group("down projection", rs.count) { rs.map { $0.down(normed) } }
    _ = group("silu + up projection + sigmoid", rs.count) {
      rs.map { sigmoid($0.up(silu(low / Float(count)))) }
    }
    _ = group("inject projection + 2 sigmoid", rs.count) {
      rs.compactMap { r in r.inject.map { 2 * sigmoid($0(normed) / Float(count)) } }
    }
    let mask = sigmoid(rs[0].up(silu(low)))
    eval(mask)
    _ = group("stream mix (reshape, mul, mean)", rs.count) {
      rs.map { _ in (mask.reshaped([1, 1, count, width]) * normed.reshaped([1, 1, count, width])).mean(axis: -2) }
    }

    print("down + up projections, three ways, over every instance:")
    let dense = rs.compactMap { r -> (MLXArray, MLXArray)? in
      guard let d = r.down as? DenseLinear, let u = r.up as? DenseLinear else { return nil }
      return (d.weight, u.weight)
    }
    let wide = dense.map { ($0.0.asType(.float32), $0.1.asType(.float32)) }
    eval(wide.flatMap { [$0.0, $0.1] })
    print("  dense fp16 pairs: \(dense.count), weight dtype \(dense.first?.0.dtype ?? .float16)")
    _ = group("as shipped (fp16 cast to fp32 per call)", dense.count) {
      dense.flatMap { [matmul(normed, $0.0.T.asType(.float32)), matmul(low, $0.1.T.asType(.float32))] }
    }
    _ = group("fp32 weights cast once at load", wide.count) {
      wide.flatMap { [matmul(normed, $0.0.T), matmul(low, $0.1.T)] }
    }
    let normedHalf = normed.asType(.float16)
    let lowHalf = low.asType(.float16)
    _ = group("fp16 activations, fp32 out", dense.count) {
      dense.flatMap {
        [matmul(normedHalf, $0.0.T).asType(.float32), matmul(lowHalf, $0.1.T).asType(.float32)]
      }
    }
    let quantized = dense.map { pair -> ((MLXArray, MLXArray, MLXArray), (MLXArray, MLXArray, MLXArray)) in
      let d = MLX.quantized(pair.0, groupSize: 64, bits: 8)
      let u = MLX.quantized(pair.1, groupSize: 64, bits: 8)
      return ((d.wq, d.scales, d.biases!), (u.wq, u.scales, u.biases!))
    }
    eval(quantized.flatMap { [$0.0.0, $0.0.1, $0.0.2, $0.1.0, $0.1.1, $0.1.2] })
    _ = group("8-bit affine, fp16 activations", quantized.count) {
      quantized.flatMap { q in
        [
          quantizedMatmul(normedHalf, q.0.0, scales: q.0.1, biases: q.0.2, groupSize: 64, bits: 8),
          quantizedMatmul(lowHalf, q.1.0, scales: q.1.1, biases: q.1.2, groupSize: 64, bits: 8),
        ]
      }
    }

    print("single moe feed-forward, one per call:")
    let one = parts.feeds[0]
    let oneMs = Self.best(10) { eval(one(x)) }
    let batchMs = Self.best(10) { eval((0..<8).map { _ in one(x) }) }
    print(String(format: "  1 alone %6.3f ms, 8 in one graph %6.3f ms each", oneMs, batchMs / 8))
  }
}
