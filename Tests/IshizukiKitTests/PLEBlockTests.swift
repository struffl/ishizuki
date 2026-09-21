// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// N-gram rows gated into the streams, against values worked out from the reference by hand.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

/// Every stage here is plausible when wrong: a gate that saturates, a convolution whose taps are
/// the wrong distance apart, a norm applied across the streams instead of within them. So the
/// numbers come from the reference transcribed independently in float64, on a block small enough
/// that the dilation reaches a state it has to have carried.
@Suite("PLE block")
struct PLEBlockTests {
  private let count = 2
  private let width = 3
  private let embed = 4
  private let kernel = 2
  private let dilation = 2
  private let eps: Float = 1e-6

  private func block() -> PLEBlock {
    PLEBlock(
      keyProj: DenseLinear(
        weight: MLXArray(Golden.keyw.flatMap { $0 }, [count * width, embed]), bias: nil),
      valueProj: DenseLinear(
        weight: MLXArray(Golden.valw.flatMap { $0 }, [width, embed]), bias: nil),
      normKey: MLXArray(Golden.nk), normQuery: MLXArray(Golden.nq),
      normConv: MLXArray(Golden.nc),
      // MLX wants a depthwise kernel as [channels, taps, 1]; a pack is relaid that way on load.
      conv: MLXArray(Golden.convw.flatMap { $0 }, [count * width, kernel, 1]),
      count: count, width: width, kernel: kernel, dilation: dilation, eps: eps)
  }

  private func inputs() -> (MLXArray, MLXArray) {
    (
      MLXArray(Golden.emb.flatMap { $0 }, [1, Golden.emb.count, embed]),
      MLXArray(Golden.streams.flatMap { $0 }, [1, Golden.streams.count, count * width])
    )
  }

  @Test("gates the n-gram value by what each stream asks for")
  func matchesReference() {
    let (embeddings, streams) = inputs()
    var state: MLXArray? = nil
    let out = block()(embeddings, streams: streams, state: &state)
    eval(out)

    #expect(out.shape == [1, Golden.emb.count, count * width])
    for (token, row) in Golden.output.enumerated() {
      for (i, want) in row.enumerated() {
        let got = out[0, token, i].item(Float.self)
        #expect(abs(got - want) < 2e-6, "output[\(token)][\(i)] was \(got), not \(want)")
      }
    }
  }

  /// The convolution is dilated by the n-gram order, so it has to carry `(kernel - 1) * dilation`
  /// positions to answer for the next token.
  @Test("keeps exactly the positions the dilation reaches back for")
  func carriesItsState() {
    let block = block()
    #expect(block.stateLength == 2)

    let (embeddings, streams) = inputs()
    var state: MLXArray? = nil
    _ = block(embeddings, streams: streams, state: &state)
    let carried = try! #require(state)
    eval(carried)
    #expect(carried.shape == [1, block.stateLength, count * width])
  }

  /// A token decoded one at a time must see the neighbours a prefilled one saw, or the
  /// convolution quietly means something different at every step.
  @Test("one token at a time matches the whole run")
  func decodeMatchesPrefill() {
    let (embeddings, streams) = inputs()
    var whole: MLXArray? = nil
    let all = block()(embeddings, streams: streams, state: &whole)
    eval(all)

    var stepwise: MLXArray? = nil
    let piece = block()
    for token in 0..<Golden.emb.count {
      let out = piece(
        embeddings[0..., token..<(token + 1), 0...],
        streams: streams[0..., token..<(token + 1), 0...], state: &stepwise)
      eval(out)
      for i in 0..<(count * width) {
        let one = out[0, 0, i].item(Float.self)
        let run = all[0, token, i].item(Float.self)
        #expect(abs(one - run) < 2e-6, "token \(token) channel \(i): \(one) vs \(run)")
      }
    }
  }

  private enum Golden {
    static let emb: [[Float]] = [
      [0.1875, -0.5625, 0.125, -0.625], [-0.375, 0.3125, -0.4375, 0.25],
      [0.5, -0.25, 0.4375, -0.3125],
    ]
    static let streams: [[Float]] = [
      [0.5, -0.25, 0.4375, -0.3125, 0.375, -0.375], [-0.0625, 0.625, -0.125, 0.5625, -0.1875, 0.5],
      [-0.625, 0.0625, -0.6875, 0.0, 0.6875, -0.0625],
    ]
    static let keyw: [[Float]] = [
      [-0.3125, 0.375, -0.375, 0.3125], [0.5625, -0.1875, 0.5, -0.25],
      [0.0, 0.6875, -0.0625, 0.625], [-0.5625, 0.125, -0.625, 0.0625],
      [0.3125, -0.4375, 0.25, -0.5], [-0.25, 0.4375, -0.3125, 0.375],
    ]
    static let valw: [[Float]] = [
      [0.3125, -0.4375, 0.25, -0.5], [-0.25, 0.4375, -0.3125, 0.375],
      [0.625, -0.125, 0.5625, -0.1875],
    ]
    static let nk: [Float] = [-0.1875, 0.5, -0.25, 0.4375, -0.3125, 0.375]
    static let nq: [Float] = [0.125, -0.625, 0.0625, -0.6875, 0.0, 0.6875]
    static let nc: [Float] = [-0.6875, 0.0, 0.6875, -0.0625, 0.625, -0.125]
    static let convw: [[Float]] = [
      [0.25, -0.5], [-0.3125, 0.375], [0.5625, -0.1875], [0.0, 0.6875], [-0.5625, 0.125],
      [0.3125, -0.4375],
    ]
    static let output: [[Float]] = [
      [0.6238403291, -0.3289741209, 0.1752113993, 0.3459035830, -0.3631545824, 0.2337877668],
      [-0.4531542350, 0.2971879841, -0.2880373654, -0.1882201018, 0.2343690958, -0.2724286854],
      [0.3000576358, -0.2170839640, 0.3508935233, 0.2648555710, -0.0899962438, 0.3661230713],
    ]
  }
}
