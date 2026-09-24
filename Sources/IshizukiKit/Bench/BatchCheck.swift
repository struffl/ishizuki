// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

/// Verify the forward pass is correct with batch size > 1, and measure the win.
public struct BatchCheck: Sendable {
  public struct Options: Sendable {
    public var model: URL
    public var batch: Int = 4
    public var decodeSteps: Int = 16

    public init(model: URL) { self.model = model }
  }

  public static func run(
    _ options: Options,
    log: @escaping @Sendable (String) -> Void
  ) throws {
    let bonsai = try BonsaiModel(path: options.model)

    let prompts = [
      "The capital city of France is called",
      "In distributed systems the term consensus refers",
      "A gated delta network stores information inside",
      "When quantizing neural network weights engineers",
      "The Hadamard transform is useful because it",
      "Apple silicon unifies CPU and GPU memory which",
      "A tokenizer converts raw text into discrete",
      "Speculative decoding accelerates inference by first",
    ]
    let count = min(options.batch, prompts.count)
    var encoded = prompts.prefix(count).map { bonsai.tokenizer.encode($0) }
    let length = encoded.map(\.count).min()!
    encoded = encoded.map { Array($0.prefix(length)) }
    log("batch \(count) x \(length) tokens\n")

    var singleLogits: [MLXArray] = []
    let singleStart = Date()
    for tokens in encoded {
      let cache = bonsai.backbone.makeCache(kvConfig: .full)
      let ids = MLXArray(tokens.map { Int32($0) }).reshaped([1, length])
      let logits = bonsai.backbone(ids, cache: cache)
      eval(logits)
      singleLogits.append(logits[0, -1])
    }
    let singleSeconds = -singleStart.timeIntervalSinceNow

    let batchStart = Date()
    let batchCache = bonsai.backbone.makeCache(kvConfig: .full)
    let batchIds = MLXArray(encoded.flatMap { $0 }.map { Int32($0) })
      .reshaped([count, length])
    let batchLogits = bonsai.backbone(batchIds, cache: batchCache)
    eval(batchLogits)
    let batchSeconds = -batchStart.timeIntervalSinceNow

    var worst: Float = 0
    var mismatches = 0
    for row in 0..<count {
      let got = batchLogits[row, -1].asType(.float32)
      let want = singleLogits[row].asType(.float32)
      let delta = abs(got - want).max().item(Float.self)
      worst = max(worst, delta)
      let gotArgmax = got.argMax().item(Int.self)
      let wantArgmax = want.argMax().item(Int.self)
      if gotArgmax != wantArgmax { mismatches += 1 }
      log(
        String(
          format: "  row %d  argmax %6d vs %6d  %@  max|Δ| %.4f",
          row, gotArgmax, wantArgmax,
          gotArgmax == wantArgmax ? "MATCH" : "MISMATCH", delta))
    }

    let decodeBatchStart = Date()
    var next = batchLogits[0..., -1, 0...].argMax(axis: -1)
    for _ in 0..<options.decodeSteps {
      let step = bonsai.backbone(next.reshaped([count, 1]), cache: batchCache)
      eval(step)
      next = step[0..., -1, 0...].argMax(axis: -1)
    }
    let batchDecodeSeconds = -decodeBatchStart.timeIntervalSinceNow

    let singleCache = bonsai.backbone.makeCache(kvConfig: .full)
    let warm = bonsai.backbone(
      MLXArray(encoded[0].map { Int32($0) }).reshaped([1, length]), cache: singleCache)
    eval(warm)
    var one = warm[0..., -1, 0...].argMax(axis: -1)
    let singleDecodeStart = Date()
    for _ in 0..<options.decodeSteps {
      let step = bonsai.backbone(one.reshaped([1, 1]), cache: singleCache)
      eval(step)
      one = step[0..., -1, 0...].argMax(axis: -1)
    }
    let singleDecodeSeconds = -singleDecodeStart.timeIntervalSinceNow

    log("")
    log(
      String(
        format: "prefill  : %.2fs sequential vs %.2fs batched  (%.2fx)",
        singleSeconds, batchSeconds, singleSeconds / batchSeconds))
    let perStreamSingle = Double(options.decodeSteps) / singleDecodeSeconds
    let perStreamBatch = Double(options.decodeSteps) / batchDecodeSeconds
    log(
      String(
        format: "decode   : %.1f tok/s for 1 stream, %.1f tok/s per stream at batch %d",
        perStreamSingle, perStreamBatch, count))
    log(
      String(
        format: "aggregate: %.1f tok/s at batch %d  (%.2fx total throughput)",
        perStreamBatch * Double(count), count,
        perStreamBatch * Double(count) / perStreamSingle))

    log("")
    if mismatches == 0 {
      log("Batched forward is correct: every row matches its own single-sequence run.")
    } else {
      log("\(mismatches) row(s) diverged -- the batch axis is not respected somewhere.")
      throw BenchFailure(
        reason: "\(mismatches) batched row(s) diverged from their single-sequence run")
    }
  }
}
