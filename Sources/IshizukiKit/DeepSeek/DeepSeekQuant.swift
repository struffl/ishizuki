// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The rounding V4.1 was trained to see in its caches, and the rotation on the tail of a head.

import Foundation
import MLX

/// The quantization V4.1's caches went through in training, applied and undone in place.
///
/// Nothing is stored narrower for it: the point is that a key reaches attention as the value it
/// had in training, where the window, the compressed latents and the indexer's queries and keys
/// were all rounded to a few bits. The three rules differ in format, block and scale, and each
/// scale is rounded up to a power of two by its exponent bits the way DeepSeek's kernels do —
/// which is not how MLX's own mxfp quantizer rounds, so none of these borrow it.
public enum DeepSeekQuant {
  /// `2^ceil(log2 r)` for a positive normal `r`, read off its exponent and mantissa.
  static func powerOfTwoCeiling(_ r: MLXArray) -> MLXArray {
    let bits = r.asType(.float32).view(dtype: .int32)
    let exponent = (bits >> 23) & 0xFF
    let carry = ((bits & 0x7FFFFF) .!= 0).asType(.int32)
    return ((exponent + carry) << 23).view(dtype: .float32)
  }

  /// Nearest of 0, ½, 1, 1½, 2, 3, 4, 6, ties to the even code, sign kept.
  static func e2m1(_ v: MLXArray) -> MLXArray {
    let m = abs(v)
    func step(_ c: MLXArray) -> MLXArray { c.asType(.float32) }
    let halves = step(m .> 0.25) + step(m .>= 0.75) + step(m .> 1.25) + step(m .>= 1.75)
    let wholes = step(m .> 2.5) + step(m .>= 3.5)
    return sign(v) * (0.5 * halves + wholes + 2 * step(m .> 5.0))
  }

  private static func blocked(_ x: MLXArray, _ block: Int) -> MLXArray {
    let shape = x.shape
    return x.asType(.float32).reshaped(Array(shape.dropLast()) + [shape.last! / block, block])
  }

  /// fp8 e4m3 in blocks of 32, power-of-two scales: the sliding-window keys.
  public static func fp8(_ x: MLXArray, block: Int = 32) -> MLXArray {
    let blocks = blocked(x, block)
    let amax = maximum(abs(blocks).max(axis: -1, keepDims: true), MLXArray(Float(1e-4)))
    let scale = powerOfTwoCeiling(amax * Float(1.0 / 448.0))
    let codes = DeepSeekFormat.toFP8(clip(blocks / scale, min: -448, max: 448))
    return (DeepSeekFormat.fromFP8(codes) * scale).reshaped(x.shape).asType(x.dtype)
  }

  /// fp4 e2m1 in blocks of 32, power-of-two scales: the indexer's queries and keys.
  public static func fp4(_ x: MLXArray, block: Int = 32) -> MLXArray {
    let blocks = blocked(x, block)
    let floor = Float(6.0) * Float(pow(2.0, -126.0))
    let amax = maximum(abs(blocks).max(axis: -1, keepDims: true), MLXArray(floor))
    let scale = powerOfTwoCeiling(amax * Float(1.0 / 6.0))
    return (e2m1(clip(blocks / scale, min: -6, max: 6)) * scale).reshaped(x.shape)
      .asType(x.dtype)
  }

  /// fp4 e2m1 in blocks of 16 under an e4m3 scale: the compressed latents, which is NVFP4
  /// without its second-level scale.
  public static func fp4E4M3(_ x: MLXArray, block: Int = 16) -> MLXArray {
    let blocks = blocked(x, block)
    let amax = maximum(abs(blocks).max(axis: -1, keepDims: true), MLXArray(Float(6.0 / 512.0)))
    let scale = DeepSeekFormat.fromFP8(DeepSeekFormat.toFP8(amax / 6))
    return (e2m1(clip(blocks / scale, min: -6, max: 6)) * scale).reshaped(x.shape)
      .asType(x.dtype)
  }
}

/// Rotary position on the last `dims` channels of a head, taken as adjacent pairs.
///
/// A V4.1 head carries its position in its tail: the first `headDim - dims` channels are left
/// alone. The rotation is also undone on the attention's output before the output projection,
/// which is what lets every layer share one cache in one rotated form.
public struct DeepSeekRope: Sendable {
  public let dims: Int
  /// Radians per position for each pair, after YaRN has faded the long wavelengths.
  public let rates: [Float]

  public init(
    dims: Int, base: Float, originalContext: Int, factor: Float, betaFast: Float,
    betaSlow: Float
  ) {
    self.dims = dims
    let half = dims / 2
    var rates = (0..<half).map { i -> Float in
      1 / Foundation.pow(base, Float(2 * i) / Float(dims))
    }
    if originalContext > 0 {
      func corrected(_ rotations: Float) -> Double {
        Double(dims) * log(Double(originalContext) / (Double(rotations) * 2 * Double.pi))
          / (2 * log(Double(base)))
      }
      let low = max(Foundation.floor(corrected(betaFast)), 0)
      let high = min(Foundation.ceil(corrected(betaSlow)), Double(dims - 1))
      let span = Float(max(high - low, 1e-3))
      for i in 0..<half {
        let ramp = min(max((Float(i) - Float(low)) / span, 0), 1)
        let smooth = 1 - ramp
        rates[i] = rates[i] / factor * (1 - smooth) + rates[i] * smooth
      }
    }
    self.rates = rates
  }

  /// Cosine and sine for each position, `[positions, dims / 2]`.
  public func table(_ positions: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
    let angles =
      positions.asType(.float32).expandedDimensions(axis: -1)
      * MLXArray(rates).expandedDimensions(axis: 0)
    return (cos(angles), sin(angles))
  }

  public func table(from start: Int, count: Int, stride: Int = 1) -> (cos: MLXArray, sin: MLXArray)
  {
    table(MLXArray((0..<count).map { Int32(start + $0 * stride) }))
  }

  /// `x` is `[b, s, d]` or `[b, s, heads, d]`; the table has one row per `s`.
  public func callAsFunction(
    _ x: MLXArray, _ table: (cos: MLXArray, sin: MLXArray), inverse: Bool = false
  ) -> MLXArray {
    let width = x.dim(-1)
    let leading = Array(x.shape.dropLast())
    var cosine = table.cos
    var sine = inverse ? -table.sin : table.sin
    if x.ndim == 4 {
      cosine = cosine.expandedDimensions(axis: 1)
      sine = sine.expandedDimensions(axis: 1)
    }
    let tail = x[.ellipsis, (width - dims)...].asType(.float32)
      .reshaped(leading + [dims / 2, 2])
    let even = tail[.ellipsis, 0]
    let odd = tail[.ellipsis, 1]
    let rotated = stacked([even * cosine - odd * sine, even * sine + odd * cosine], axis: -1)
      .reshaped(leading + [dims]).asType(x.dtype)
    guard width > dims else { return rotated }
    return concatenated([x[.ellipsis, ..<(width - dims)], rotated], axis: -1)
  }
}
