// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

/// Affine quantization guided by a per-input-channel importance vector, instead of the raw
/// per-group min/max MLX's own `quantized(mode: .affine)` takes outright.
///
/// This is oMLX's oQe scheme (`~/genAI/omlx/omlx/oq.py`, `_weighted_affine_quantize` and
/// `_pack_affine_codes`), ported so ishizuki's native quantizer can approach its quality without
/// a Python dependency. It is not GPTQ: there is no Hessian and no sequential error correction
/// across columns, just a small search over how far each group's quantization range should be
/// clipped, scored by activation-energy-weighted reconstruction error rather than plain
/// Frobenius error. Outlier values inflate a naive min/max range and waste most of the group's
/// codes on values that barely occur; clipping the range tighter loses those outliers but
/// packs the values that actually matter (per the importance weights) more precisely.
///
/// The packed format is bit-for-bit MLX's own affine layout — `_pack_affine_codes` exists
/// because MLX does not expose a way to pack codes chosen by anything other than its own
/// min/max, not because the format itself is different. Anything this writes reads back
/// through `dequantized`/`quantizedMM` exactly like a pack `quantized()` wrote.
///
/// Restricted to 2D weights, which is all `Quantizer` ever calls this with.
public enum WeightedAffineQuantizer {
  /// The clipping factors oMLX searches, applied to the naive min/max range on each side.
  private static let clipFactors: [Float] = [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25]

  /// Quantizes `weight` [outputDim, inputDim] with one scale/bias per `groupSize` input
  /// channels, chosen to minimize `importance`-weighted reconstruction error rather than
  /// Frobenius error.
  ///
  /// - Parameters:
  ///   - importance: per-input-channel weights, shape `[inputDim]`. Channels the calibration
  ///     pass saw carry more activation energy and are weighted more heavily; a channel with
  ///     no signal (or no calibration at all) should be handed a uniform weight, which makes
  ///     this degrade to a plain clipping search with no importance signal, not a crash.
  public static func quantize(
    _ weight: MLXArray, groupSize: Int, bits: Int, importance: MLXArray
  ) -> (wq: MLXArray, scales: MLXArray, biases: MLXArray) {
    precondition(weight.ndim == 2, "WeightedAffineQuantizer only quantizes 2D weights")
    let outputDim = weight.dim(0)
    let inputDim = weight.dim(1)
    precondition(
      inputDim % groupSize == 0,
      "input width \(inputDim) does not divide group size \(groupSize)")
    let groups = inputDim / groupSize

    let w = weight.asType(.float32).reshaped([outputDim, groups, groupSize])
    let imp = maximum(
      importance.asType(.float32).reshaped([1, groups, groupSize]), MLXArray(Float(1e-8)))

    let (scales, biases) = search(w, importance: imp, bits: bits)
    let codes = quantizedCodes(w, scales: scales, biases: biases, bits: bits)
    let wq = pack(codes, groupSize: groupSize, bits: bits)

    return (wq, scales.squeezed(axis: -1), biases.squeezed(axis: -1))
  }

  /// The naive per-group min/max scale and bias, oriented so the larger-magnitude edge lands
  /// on an exact integer code — mirrors MLX's own affine quantizer exactly, before any
  /// importance-guided clipping is applied.
  static func minMaxParams(_ grouped: MLXArray, bits: Int) -> (scales: MLXArray, biases: MLXArray)
  {
    let nBins = MLXArray(Float((1 << bits) - 1))
    let eps = MLXArray(Float(1e-7))
    let wMax = grouped.max(axis: -1, keepDims: true)
    let wMin = grouped.min(axis: -1, keepDims: true)
    let mask = abs(wMin) .> abs(wMax)

    var scales = maximum((wMax - wMin) / nBins, eps)
    scales = which(mask, scales, -scales)
    let edge = which(mask, wMin, wMax)

    let q0 = round(edge / scales)
    scales = which(q0 .!= 0, edge / q0, scales)
    let biases = which(q0 .== 0, MLXArray(Float(0)), edge)
    return (scales, biases)
  }

  /// Scores a candidate (scales, biases) pair by importance-weighted reconstruction error.
  static func weightedError(
    _ grouped: MLXArray, scales: MLXArray, biases: MLXArray, importance: MLXArray, bits: Int
  ) -> MLXArray {
    let codes = quantizedCodes(grouped, scales: scales, biases: biases, bits: bits)
    let restored = codes.asType(.float32) * scales + biases
    let diff = grouped - restored
    return (importance * diff * diff).sum(axis: -1, keepDims: true)
  }

  /// Searches both quantization directions (range anchored at the max, or at the min) and a
  /// small grid of clipping factors on each, keeping whichever minimizes weighted error per
  /// group. Groups disagree on which candidate wins — the search runs per-group throughout.
  static func search(
    _ grouped: MLXArray, importance: MLXArray, bits: Int
  ) -> (scales: MLXArray, biases: MLXArray) {
    let (baseScales, baseBiases) = minMaxParams(grouped, bits: bits)
    var bestScales = baseScales
    var bestBiases = baseBiases
    var bestError = weightedError(
      grouped, scales: baseScales, biases: baseBiases, importance: importance, bits: bits)

    let nBins = MLXArray(Float((1 << bits) - 1))
    let eps = MLXArray(Float(1e-7))
    let wMax = grouped.max(axis: -1, keepDims: true)
    let wMin = grouped.min(axis: -1, keepDims: true)

    // (edge, opposite, sign): anchoring the range at the max with the min as the far side, and
    // the mirror image anchoring it at the min.
    let candidates: [(edge: MLXArray, opposite: MLXArray, sign: Float)] = [
      (wMax, wMin, -1.0), (wMin, wMax, 1.0),
    ]
    for (edge, opposite, sign) in candidates {
      let raw = maximum(abs(edge - opposite) / nBins, eps) * MLXArray(sign)
      let q0 = round(edge / raw)
      let scale0 = which(q0 .!= 0, edge / q0, raw)
      let bias0 = which(q0 .== 0, MLXArray(Float(0)), edge)

      for factor in clipFactors {
        let scales = scale0 * MLXArray(factor)
        let error = weightedError(
          grouped, scales: scales, biases: bias0, importance: importance, bits: bits)
        let take = error .< bestError
        bestScales = which(take, scales, bestScales)
        bestBiases = which(take, bias0, bestBiases)
        bestError = which(take, error, bestError)
      }
    }
    return (bestScales, bestBiases)
  }

  static func quantizedCodes(
    _ grouped: MLXArray, scales: MLXArray, biases: MLXArray, bits: Int
  ) -> MLXArray {
    let nBins = MLXArray(Float((1 << bits) - 1))
    let raw = round((grouped - biases) / scales)
    return clip(raw, min: MLXArray(Float(0)), max: nBins).asType(.uint32)
  }

  /// Packs per-element integer codes (one per weight, values in `0..<2^bits`) into MLX's own
  /// affine bit layout: `groupSize` codes packed densely across `groupSize * bits / 32`
  /// 32-bit words. Ported from oMLX's `_pack_affine_codes`.
  static func pack(_ codes: MLXArray, groupSize: Int, bits: Int) -> MLXArray {
    let outputDim = codes.dim(0)
    let inputDim = codes.dim(1) * codes.dim(2)
    let flat = codes.reshaped([outputDim, inputDim])

    let packedWidth = inputDim * bits / 32
    if [2, 4, 8].contains(bits) {
      let elPerInt = 32 / bits
      let shifts = MLXArray(UInt32(2)) ** MLXArray.arange(0, 32, step: bits, dtype: .uint32)
      let grouped = flat.reshaped([outputDim, inputDim / elPerInt, elPerInt])
      return (grouped * shifts).sum(axis: -1).reshaped([outputDim, packedWidth])
    }

    // Widths that do not divide 32 (3, 5, 6 bits) pack bit-by-bit rather than value-by-value.
    let bitsArange = MLXArray.arange(0, bits, dtype: .uint32)
    let bitValues = bitwiseAnd(rightShift(flat.expandedDimensions(axis: -1), bitsArange), 1)
    let flatBits = bitValues.reshaped([outputDim, inputDim * bits / 32, 32])
    let shifts = MLXArray.arange(0, 32, dtype: .uint32)
    return leftShift(flatBits, shifts).sum(axis: -1).reshaped([outputDim, packedWidth])
  }
}
