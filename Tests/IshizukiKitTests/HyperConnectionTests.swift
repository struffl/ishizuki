// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Four residual streams, against values worked out from the reference by hand.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// A widened residual cannot be checked by looking at it: every stream is plausible whatever the
/// gates do. So the numbers here come from the reference formulation transcribed independently
/// in float64 — the grouped norm, the low-rank gate, the doubled sigmoid and the write-back —
/// on shapes small enough to be exact.
@Suite("Hyper connection")
struct HyperConnectionTests {
  private let count = 4
  private let width = 3
  private let eps: Float = 1e-6

  private func residual(withInject: Bool = true) -> GatedResidual {
    GatedResidual(
      norm: MLXArray(Golden.norm),
      down: DenseLinear(
        weight: MLXArray(Golden.down.flatMap { $0 }, [2, count * width]), bias: nil),
      up: DenseLinear(weight: MLXArray(Golden.up.flatMap { $0 }, [count * width, 2]), bias: nil),
      inject: withInject
        ? DenseLinear(
          weight: MLXArray(Golden.inject.flatMap { $0 }, [count, count * width]), bias: nil)
        : nil,
      count: count, width: width, eps: eps)
  }

  private func streams() -> MLXArray {
    MLXArray(Golden.x.flatMap { $0 }, [Golden.x.count, count * width])
  }

  @Test("mixes the streams into the one width a block reads")
  func mixes() {
    let opened = residual()(streams())
    eval(opened.mixed)

    #expect(opened.mixed.shape == [Golden.x.count, width])
    for (token, row) in Golden.mixed.enumerated() {
      for (i, want) in row.enumerated() {
        let got = opened.mixed[token, i].item(Float.self)
        #expect(abs(got - want) < 2e-6, "mixed[\(token)][\(i)] was \(got), not \(want)")
      }
    }
  }

  @Test("gates each stream's share of the answer")
  func injects() {
    let opened = residual()(streams())
    let injection = try! #require(opened.injection)
    eval(injection)

    #expect(injection.shape == [Golden.x.count, count])
    for (token, row) in Golden.injection.enumerated() {
      for (c, want) in row.enumerated() {
        let got = injection[token, c].item(Float.self)
        #expect(abs(got - want) < 2e-6, "injection[\(token)][\(c)] was \(got), not \(want)")
      }
    }
  }

  @Test("writes a block's answer back into every stream")
  func closes() {
    let opened = residual()(streams())
    let answer = MLXArray(Golden.answer.flatMap { $0 }, [Golden.answer.count, width])
    let closed = GatedResidual.close(opened, with: answer)
    eval(closed)

    #expect(closed.shape == [Golden.x.count, count * width])
    for (token, row) in Golden.closed.enumerated() {
      for (i, want) in row.enumerated() {
        let got = closed[token, i].item(Float.self)
        #expect(abs(got - want) < 2e-6, "closed[\(token)][\(i)] was \(got), not \(want)")
      }
    }
  }

  /// The model-level mixer has no block to inject into, so it only narrows.
  @Test("without a block to feed, it mixes and stops")
  func mixerOnly() {
    let opened = residual(withInject: false)(streams())
    #expect(opened.injection == nil)
    eval(opened.mixed)
    #expect(opened.mixed.shape == [Golden.x.count, width])
    // The mix does not depend on the injection, so it is the same either way.
    for (token, row) in Golden.mixed.enumerated() {
      for (i, want) in row.enumerated() {
        #expect(abs(opened.mixed[token, i].item(Float.self) - want) < 2e-6)
      }
    }
    // With nothing gated, closing is the block's own answer.
    let answer = MLXArray(Golden.answer.flatMap { $0 }, [Golden.answer.count, width])
    #expect(GatedResidual.close(opened, with: answer).shape == answer.shape)
  }

  private enum Golden {
    static let x: [[Float]] = [
      [-0.6875, 0.0, 0.6875, -0.0625, 0.625, -0.125, 0.5625, -0.1875, 0.5, -0.25, 0.4375, -0.3125],
      [
        0.1875, -0.5625, 0.125, -0.625, 0.0625, -0.6875, 0.0, 0.6875, -0.0625, 0.625, -0.125,
        0.5625,
      ],
    ]
    static let norm: [Float] = [
      -0.625, 0.0625, -0.6875, 0.0, 0.6875, -0.0625, 0.625, -0.125, 0.5625, -0.1875, 0.5, -0.25,
    ]
    static let down: [[Float]] = [
      [
        -0.3125, 0.375, -0.375, 0.3125, -0.4375, 0.25, -0.5, 0.1875, -0.5625, 0.125, -0.625, 0.0625,
      ],
      [0.5625, -0.1875, 0.5, -0.25, 0.4375, -0.3125, 0.375, -0.375, 0.3125, -0.4375, 0.25, -0.5],
    ]
    static let up: [[Float]] = [
      [0.625, -0.125], [0.0625, -0.6875], [-0.5, 0.1875], [0.375, -0.375], [-0.1875, 0.5],
      [0.6875, -0.0625], [0.125, -0.625], [-0.4375, 0.25], [0.4375, -0.3125], [-0.125, 0.5625],
      [-0.6875, 0.0], [0.1875, -0.5625],
    ]
    static let inject: [[Float]] = [
      [0.125, -0.625, 0.0625, -0.6875, 0.0, 0.6875, -0.0625, 0.625, -0.125, 0.5625, -0.1875, 0.5],
      [-0.4375, 0.25, -0.5, 0.1875, -0.5625, 0.125, -0.625, 0.0625, -0.6875, 0.0, 0.6875, -0.0625],
      [
        0.4375, -0.3125, 0.375, -0.375, 0.3125, -0.4375, 0.25, -0.5, 0.1875, -0.5625, 0.125, -0.625,
      ],
      [-0.125, 0.5625, -0.1875, 0.5, -0.25, 0.4375, -0.3125, 0.375, -0.375, 0.3125, -0.4375, 0.25],
    ]
    static let answer: [[Float]] = [[0.25, -0.5, 0.1875], [0.25, -0.5, 0.1875]]
    static let mixed: [[Float]] = [
      [0.2012042207, 0.2432992873, -0.0069667389], [-0.0719723579, -0.0452089715, -0.0674435915],
    ]
    static let injection: [[Float]] = [
      [1.0042544249, 0.8683834435, 1.0650491081, 0.8929064635],
      [0.9604085658, 1.0233462635, 1.0214324532, 0.9885100599],
    ]
    static let closed: [[Float]] = [
      [
        -0.4364363938, -0.5021272125, 0.8757977047, 0.1545958609, 0.1908082782, 0.0378218957,
        0.8287622770, -0.7200245540, 0.6996967078, -0.0267733841, -0.0089532318, -0.1450800381,
      ],
      [
        0.4276021414, -1.0427042829, 0.3050766061, -0.3691634341, -0.4491731317, -0.4956225756,
        0.2553581133, 0.1767837734, 0.1290185850, 0.8721275150, -0.6192550299, 0.7478456362,
      ],
    ]
  }
}
