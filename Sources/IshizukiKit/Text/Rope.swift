// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXFast

public final class RotaryEmbedding: @unchecked Sendable {
  public let dimensions: Int
  public let base: Float
  public let axisForFrequency: [Int]
  private let invFreq: MLXArray
  public let scaling: RopeScaling
  private let scaledFrequencies: MLXArray?

  public init(
    dimensions: Int, base: Float, mropeSection: [Int]?, interleaved: Bool,
    scaling: RopeScaling = .none
  ) {
    self.dimensions = dimensions
    self.base = base
    self.scaling = scaling
    self.scaledFrequencies = scaling.frequencies(dimensions: dimensions, base: base)

    let half = dimensions / 2
    var axes = [Int](repeating: 0, count: half)
    if let section = mropeSection, section.count == 3 {
      if interleaved {
        for (axis, offset) in [(1, 1), (2, 2)] {
          let limit = min(section[axis] * 3, half)
          var i = offset
          while i < limit {
            axes[i] = axis
            i += 3
          }
        }
      } else {
        var i = 0
        for (axis, count) in section.enumerated() {
          for _ in 0..<count where i < half {
            axes[i] = axis
            i += 1
          }
        }
      }
    }
    self.axisForFrequency = axes

    if let scaledFrequencies {
      self.invFreq = 1.0 / scaledFrequencies
    } else {
      let exponents =
        MLXArray(stride(from: 0, to: dimensions, by: 2).map { Float($0) })
        / Float(dimensions)
      self.invFreq = 1.0 / pow(MLXArray(base), exponents)
    }
  }

  public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
    var x = x
    if scaling.attentionScale != 1 {
      x = x * scaling.attentionScale
    }
    return MLXFast.RoPE(
      x, dimensions: dimensions, traditional: false,
      base: scaledFrequencies == nil ? base : nil,
      scale: 1.0, offset: offset, freqs: scaledFrequencies)
  }

  public func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
    var x = x
    if scaling.attentionScale != 1 {
      x = x * scaling.attentionScale
    }
    let headDim = x.dim(-1)
    let (cosine, sine) = cosSin(positions: positions, dtype: x.dtype)

    if dimensions == headDim {
      return rotate(x, cosine: cosine, sine: sine)
    }
    let rotated = rotate(x[.ellipsis, 0..<dimensions], cosine: cosine, sine: sine)
    let passthrough = x[.ellipsis, dimensions..<headDim]
    return concatenated([rotated, passthrough], axis: -1)
  }

  private func cosSin(positions: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
    let angles = positions.asType(.float32).expandedDimensions(axis: -1) * invFreq

    let axisIndex = MLXArray(axisForFrequency.map { Int32($0) })
    let freqIndex = MLXArray((0..<dimensions / 2).map { Int32($0) })
    let selected = angles[axisIndex, 0..., freqIndex]
    let freqs = selected.transposed(1, 0)

    let doubled = concatenated([freqs, freqs], axis: -1)
    let cosine = cos(doubled).asType(dtype).expandedDimensions(axes: [0, 1])
    let sine = sin(doubled).asType(dtype).expandedDimensions(axes: [0, 1])
    return (cosine, sine)
  }

  private func rotate(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0..<half]
    let x2 = x[.ellipsis, half..<x.dim(-1)]
    let rotatedHalves = concatenated([-x2, x1], axis: -1)
    return x * cosine + rotatedHalves * sine
  }
}
