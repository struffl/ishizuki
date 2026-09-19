// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The two value-head orders the delta rule has to serve.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

/// A checkpoint groups value heads by key head; llama.cpp's converter tiles them. The weights
/// carry no mark of which, so the only thing that can go wrong is silent: every head pairs with
/// a plausible neighbour and the model reads fluent.
@Suite("Gated delta net")
struct GatedDeltaNetTests {
  private let batch = 2
  private let steps = 3
  private let keyHeads = 4
  private let repeats = 3
  private let keyDim = 32
  private let valueDim = 32

  private var valueHeads: Int { keyHeads * repeats }

  /// Grouped position of the head that tiled position `t` holds.
  private func grouped(from tiled: Int) -> Int {
    (tiled % keyHeads) * repeats + tiled / keyHeads
  }

  private func retiled(_ x: MLXArray, axis: Int) -> MLXArray {
    let order = MLXArray((0..<valueHeads).map { Int32(grouped(from: $0)) })
    return take(x, order, axis: axis)
  }

  @Test("tiled heads give the same answer as grouped ones, head for head")
  func layoutsAgree() {
    MLXRandom.seed(90210)
    let q = MLXRandom.normal([batch, steps, keyHeads, keyDim]).asType(.float32)
    let k = MLXRandom.normal([batch, steps, keyHeads, keyDim]).asType(.float32)
    let v = MLXRandom.normal([batch, steps, valueHeads, valueDim]).asType(.float32)
    let g = MLXRandom.uniform(low: 0.8, high: 1.0, [batch, steps, valueHeads])
      .asType(.float32)
    let beta = MLXRandom.uniform(low: 0.0, high: 1.0, [batch, steps, valueHeads])
      .asType(.float32)
    let state = MLXRandom.normal([batch, valueHeads, valueDim, keyDim]).asType(.float32)

    func run(_ layout: ValueHeadLayout, _ inputs: (MLXArray, MLXArray, MLXArray, MLXArray))
      -> (MLXArray, MLXArray)
    {
      GatedDeltaNet.deltaRule(
        q: q, k: k, v: inputs.0, g: inputs.1, beta: inputs.2, state: inputs.3,
        headRepeat: repeats, layout: layout)
    }

    let (yGrouped, stateGrouped) = run(.grouped, (v, g, beta, state))
    let (yTiled, stateTiled) = run(
      .tiled,
      (
        retiled(v, axis: 2), retiled(g, axis: 2), retiled(beta, axis: 2),
        retiled(state, axis: 1)
      ))

    // Whichever path `deltaRule` picked, the other one has to agree with it: on a GPU that is
    // the Metal kernel's own `hv % Hk` against the ops fallback's broadcast.
    let (yOps, stateOps) = GatedDeltaNet.opsDeltaRule(
      q: q, k: k, v: retiled(v, axis: 2), g: retiled(g, axis: 2),
      beta: retiled(beta, axis: 2), state: retiled(state, axis: 1),
      headRepeat: repeats, layout: .tiled)
    // Loose, because the kernel sums across a simdgroup and the fallback does not: the bound
    // is on the pairing, not on the arithmetic.
    let paths = abs(yOps - yTiled).max().item(Float.self)
    #expect(paths < 1e-2, "kernel and fallback differ by \(paths)")
    #expect(abs(stateOps - stateTiled).max().item(Float.self) < 1e-2)

    let y = abs(retiled(yGrouped, axis: 2) - yTiled).max().item(Float.self)
    let s = abs(retiled(stateGrouped, axis: 1) - stateTiled).max().item(Float.self)
    #expect(y < 1e-2, "outputs differ by \(y)")
    #expect(s < 1e-2, "states differ by \(s)")
  }

  @Test("the fallback broadcasts each order the way its kernel indexes it")
  func broadcastMatchesIndexing() {
    let x = MLXArray((0..<keyHeads).map { Float($0) }).reshaped([1, 1, keyHeads, 1])

    let grouped = GatedDeltaNet.broadcastHeads(x, repeat: repeats, layout: .grouped)
      .reshaped([valueHeads])
    let tiled = GatedDeltaNet.broadcastHeads(x, repeat: repeats, layout: .tiled)
      .reshaped([valueHeads])
    eval(grouped, tiled)

    for head in 0..<valueHeads {
      #expect(grouped[head].item(Float.self) == Float(head / repeats))
      #expect(tiled[head].item(Float.self) == Float(head % keyHeads))
    }
  }
}
