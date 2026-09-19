// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import ArgumentParser
import Foundation
import IshizukiKit
import MLX

struct BatchCheck: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "batch-check",
    abstract: "Verify the forward pass is correct with batch size > 1, and measure the win.")

  @Option(name: .long) var model: String = defaultModelPath
  @Option(name: .long, help: "Sequences to batch.") var batch: Int = 4
  @Option(name: .long) var decodeSteps: Int = 16

  func run() throws {
    let bonsai = try BonsaiModel(directory: URL(filePath: model))

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
    let count = min(batch, prompts.count)
    var encoded = prompts.prefix(count).map { bonsai.tokenizer.encode($0) }
    let length = encoded.map(\.count).min()!
    encoded = encoded.map { Array($0.prefix(length)) }
    print("batch \(count) x \(length) tokens\n")

    var singleLogits: [MLXArray] = []
    let singleStart = Date()
    for tokens in encoded {
      let cache = bonsai.text.makeCache(kvConfig: .full)
      let ids = MLXArray(tokens.map { Int32($0) }).reshaped([1, length])
      let logits = bonsai.text(ids, cache: cache)
      eval(logits)
      singleLogits.append(logits[0, -1])
    }
    let singleSeconds = -singleStart.timeIntervalSinceNow

    let batchStart = Date()
    let batchCache = bonsai.text.makeCache(kvConfig: .full)
    let batchIds = MLXArray(encoded.flatMap { $0 }.map { Int32($0) })
      .reshaped([count, length])
    let batchLogits = bonsai.text(batchIds, cache: batchCache)
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
      print(
        String(
          format: "  row %d  argmax %6d vs %6d  %@  max|Δ| %.4f",
          row, gotArgmax, wantArgmax,
          gotArgmax == wantArgmax ? "MATCH" : "MISMATCH", delta))
    }

    let decodeBatchStart = Date()
    var next = batchLogits[0..., -1, 0...].argMax(axis: -1)
    for _ in 0..<decodeSteps {
      let step = bonsai.text(next.reshaped([count, 1]), cache: batchCache)
      eval(step)
      next = step[0..., -1, 0...].argMax(axis: -1)
    }
    let batchDecodeSeconds = -decodeBatchStart.timeIntervalSinceNow

    let singleCache = bonsai.text.makeCache(kvConfig: .full)
    let warm = bonsai.text(
      MLXArray(encoded[0].map { Int32($0) }).reshaped([1, length]), cache: singleCache)
    eval(warm)
    var one = warm[0..., -1, 0...].argMax(axis: -1)
    let singleDecodeStart = Date()
    for _ in 0..<decodeSteps {
      let step = bonsai.text(one.reshaped([1, 1]), cache: singleCache)
      eval(step)
      one = step[0..., -1, 0...].argMax(axis: -1)
    }
    let singleDecodeSeconds = -singleDecodeStart.timeIntervalSinceNow

    print("")
    print(
      String(
        format: "prefill  : %.2fs sequential vs %.2fs batched  (%.2fx)",
        singleSeconds, batchSeconds, singleSeconds / batchSeconds))
    let perStreamSingle = Double(decodeSteps) / singleDecodeSeconds
    let perStreamBatch = Double(decodeSteps) / batchDecodeSeconds
    print(
      String(
        format: "decode   : %.1f tok/s for 1 stream, %.1f tok/s per stream at batch %d",
        perStreamSingle, perStreamBatch, count))
    print(
      String(
        format: "aggregate: %.1f tok/s at batch %d  (%.2fx total throughput)",
        perStreamBatch * Double(count), count,
        perStreamBatch * Double(count) / perStreamSingle))

    print("")
    if mismatches == 0 {
      print("Batched forward is correct: every row matches its own single-sequence run.")
    } else {
      print("\(mismatches) row(s) diverged -- the batch axis is not respected somewhere.")
      throw ExitCode.failure
    }
  }
}
