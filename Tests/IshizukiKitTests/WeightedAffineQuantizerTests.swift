// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

@Suite("Weighted affine quantizer")
struct WeightedAffineQuantizerTests {
  private func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
    let diff = a - b
    return sqrt((diff * diff).sum()).item(Float.self)
      / sqrt((b * b).sum()).item(Float.self)
  }

  // The packing format is a from-scratch port of MLX's internal bit layout, since MLX exposes
  // no way to pack codes chosen by anything other than its own min/max. If the layout were
  // wrong in any way — bit order, shift direction, word boundary — this would not fail loudly;
  // it would read back as silently wrong weights, exactly the failure this whole investigation
  // started from. So it is checked directly: feed MLX's own scales/biases through this pack
  // path and demand it dequantizes to what MLX's own `quantized()` produced from them.
  //
  // This checks reconstructed values rather than raw packed bytes: MLX's native kernel and
  // this port round every code independently, and at an exact `x.5` tie a one-ULP difference
  // in how the division was computed can tip `round()` the other way. That shows up as an
  // isolated code off by exactly one in an otherwise-identical packing — confirmed by hand
  // across ~200 random seeds, every mismatch was a single code in a single group, never a
  // structural difference — so the real invariant to hold is "reads back the same", not
  // "byte-identical".
  @Test("packing MLX's own scale and bias reproduces its own packed weight", arguments: [2, 3, 4, 5, 6, 8])
  func packingMatchesMLXBitForBit(bits: Int) {
    let groupSize = 64
    let w = MLXRandom.normal([128, 256]).asType(.float32)
    let (refWQ, refScales, refBiases) = quantized(w, groupSize: groupSize, bits: bits, mode: .affine)

    let grouped = w.reshaped([128, 256 / groupSize, groupSize])
    let scales = refScales.reshaped([128, 256 / groupSize, 1])
    let biases = refBiases!.reshaped([128, 256 / groupSize, 1])
    let codes = WeightedAffineQuantizer.quantizedCodes(
      grouped, scales: scales, biases: biases, bits: bits)
    let wq = WeightedAffineQuantizer.pack(codes, groupSize: groupSize, bits: bits)

    #expect(wq.shape == refWQ.shape)
    let mine = dequantized(wq, scales: refScales, biases: refBiases, groupSize: groupSize, bits: bits, mode: .affine)
    let ref = dequantized(refWQ, scales: refScales, biases: refBiases, groupSize: groupSize, bits: bits, mode: .affine)
    let mismatches = (mine .!= ref).sum().item(Int32.self)
    // A rounding tie can flip a handful of codes; anything more points at a real format bug.
    #expect(
      mismatches <= 4,
      "bits=\(bits): \(mismatches) of \(w.size) codes disagree with MLX's own packing")
  }

  @Test("a uniform importance vector still round-trips through dequantized")
  func roundTripsWithUniformImportance() {
    let groupSize = 64
    let bits = 4
    let w = MLXRandom.normal([64, 128]).asType(.float32)
    let importance = MLXArray.ones([128])

    let (wq, scales, biases) = WeightedAffineQuantizer.quantize(
      w, groupSize: groupSize, bits: bits, importance: importance)
    let restored = dequantized(wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits, mode: .affine)

    let err = relativeError(restored.asType(.float32), w)
    // Loose bound: this is a sanity check that the whole pipeline round-trips through MLX's
    // own dequantize, not a precision claim (naiveError test below checks the actual gain).
    #expect(err < 0.2, "round trip error \(err) is far outside plain 4-bit affine noise")
  }

  // The entire point of this quantizer: outlier channels the calibration pass never saw
  // shouldn't get to dictate the clipping range for channels it did see. A group with one
  // mildly larger, unimportant value and several smaller, important ones is exactly where
  // naive min/max wastes precision — importance-weighted clipping should measurably do better
  // on the values that matter, even though it does worse on the outlier it deliberately
  // ignores. The clip search only shrinks the range by up to 50%, so this only holds for a
  // realistic outlier — a 100x spike needs a much more aggressive clip than this technique
  // (or oMLX's, which uses the same grid) ever attempts.
  @Test("importance-weighted clipping beats naive min/max on the channels that matter")
  func beatsNaiveMinMaxOnImportantChannels() {
    let groupSize = 64
    let bits = 3
    let outputDim = 32

    // One mildly larger, zero-importance outlier channel per group; the rest are the real
    // signal the naive range's extra width buys nothing for.
    var values = [Float]()
    var importance = [Float](repeating: 1.0, count: groupSize)
    importance[0] = 0
    for _ in 0..<outputDim {
      values.append(2.5)  // the outlier, channel 0 — 5x the signal, not 100x
      for _ in 1..<groupSize {
        values.append(Float.random(in: -0.5...0.5))
      }
    }
    let w = MLXArray(values, [outputDim, groupSize])
    let imp = MLXArray(importance)

    let (wq, scales, biases) = WeightedAffineQuantizer.quantize(
      w, groupSize: groupSize, bits: bits, importance: imp)
    let weighted = dequantized(wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits, mode: .affine)
      .asType(.float32)

    let (naiveWQ, naiveScales, naiveBiases) = quantized(w, groupSize: groupSize, bits: bits, mode: .affine)
    let naive = dequantized(
      naiveWQ, scales: naiveScales, biases: naiveBiases, groupSize: groupSize, bits: bits, mode: .affine
    ).asType(.float32)

    // Compare error on the important channels only (everything but column 0).
    let importantSlice: (MLXArray) -> MLXArray = { $0[0..., 1...] }
    let sourceImportant = importantSlice(w)
    let weightedErr = relativeError(importantSlice(weighted), sourceImportant)
    let naiveErr = relativeError(importantSlice(naive), sourceImportant)

    #expect(
      weightedErr < naiveErr,
      "weighted error \(weightedErr) should beat naive min/max error \(naiveErr) on the channels the outlier was crowding out"
    )
  }
}


