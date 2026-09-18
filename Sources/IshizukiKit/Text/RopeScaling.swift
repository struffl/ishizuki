// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: MIT

import Foundation
import MLX

public struct RopeScaling: Sendable, Equatable {
  public enum Method: String, Sendable, CaseIterable {
    case none
    case linear
    case ntk
    case yarn
  }

  public var method: Method
  public var factor: Float
  public var originalContext: Int

  public init(method: Method = .yarn, factor: Float = 1, originalContext: Int = 262_144) {
    self.method = method
    self.factor = factor
    self.originalContext = originalContext
  }

  public static let none = RopeScaling(method: .none, factor: 1)

  public var isActive: Bool { method != .none && factor > 1 }

  public var effectiveContext: Int { Int(Float(originalContext) * max(factor, 1)) }

  public func frequencies(dimensions: Int, base: Float) -> MLXArray? {
    guard isActive else { return nil }

    let exponents =
      MLXArray(stride(from: 0, to: dimensions, by: 2).map { Float($0) })
      / Float(dimensions)
    let extrapolation = pow(MLXArray(base), exponents)

    switch method {
    case .none:
      return nil

    case .linear:
      return extrapolation * factor

    case .ntk:
      let adjusted = base * pow(factor, Float(dimensions) / Float(dimensions - 2))
      return pow(MLXArray(adjusted), exponents)

    case .yarn:
      let interpolation = extrapolation * factor
      let (low, high) = correctionRange(dimensions: dimensions, base: base)
      let ramp = linearRamp(low: low, high: high, count: dimensions / 2)
      let mask = 1.0 - ramp
      return (interpolation * extrapolation)
        / (interpolation * mask + extrapolation * (1 - mask))
    }
  }

  public var attentionScale: Float {
    guard method == .yarn, factor > 1 else { return 1 }
    return 0.1 * log(factor) + 1.0
  }

  private func correctionRange(
    dimensions: Int, base: Float, betaFast: Float = 32, betaSlow: Float = 1
  ) -> (Float, Float) {
    func correctionDimension(_ rotations: Float) -> Float {
      Float(dimensions)
        * log(Float(originalContext) / (rotations * 2 * .pi))
        / (2 * log(base))
    }
    let low = max(correctionDimension(betaFast).rounded(.down), 0)
    let high = min(correctionDimension(betaSlow).rounded(.up), Float(dimensions - 1))
    return (low, high)
  }

  private func linearRamp(low: Float, high: Float, count: Int) -> MLXArray {
    let high = low == high ? high + 0.001 : high
    let positions = MLXArray((0..<count).map { Float($0) })
    return clip((positions - low) / (high - low), min: 0, max: 1)
  }
}
