// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

/// What a module costs and what it loses, measured at each width it might be given.
public struct ModuleMeasurement: Sendable {
  public let path: String
  public let elements: Int
  /// Relative Frobenius error of the round trip at each candidate width.
  public let errorAt: [Int: Double]

  public func error(_ bits: Int) -> Double { errorAt[bits] ?? 0 }

  /// Bits per weight once the group's scale and bias are counted. At group 64 that is half a
  /// bit on top of the nominal width, which is most of the gap between a "4-bit" pack and the
  /// 4.5 bpw it actually occupies.
  public static func bpw(bits: Int, groupSize: Int) -> Double {
    Double(bits) + 32.0 / Double(groupSize)
  }

  public func bytes(bits: Int, groupSize: Int) -> Double {
    Double(elements) * Self.bpw(bits: bits, groupSize: groupSize) / 8
  }
}

/// Measures every quantizable module by quantizing it and comparing the result to the original.
///
/// This is the honest part of the job and the slow part: each module is read, quantized at every
/// width it might be given, and dequantized to see what it lost. Nothing is inferred from the
/// module's name or position — a projection is boosted because it measurably suffers at the
/// base width, not because projections of that name usually do.
public enum ModuleSurvey {
  public static func measure(
    _ weight: MLXArray, path: String, widths: [Int], groupSize: Int
  ) -> ModuleMeasurement {
    let reference = weight.asType(.float32)
    let denominator = sqrt((reference * reference).sum()).item(Float.self)

    var errors: [Int: Double] = [:]
    for bits in widths {
      let (wq, scales, biases) = quantized(
        weight, groupSize: groupSize, bits: bits, mode: .affine)
      let restored = dequantized(
        wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits, mode: .affine
      ).asType(.float32)
      let difference = restored - reference
      let numerator = sqrt((difference * difference).sum()).item(Float.self)
      errors[bits] = denominator > 0 ? Double(numerator / denominator) : 0
    }

    return ModuleMeasurement(
      path: path, elements: weight.size, errorAt: errors)
  }
}
