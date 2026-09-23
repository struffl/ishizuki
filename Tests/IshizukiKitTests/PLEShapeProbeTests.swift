// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Probe the shapes that reach shortConv on a Whittle-sized PLE block.

import Foundation
import MLX
import Testing

@testable import IshizukiKit

@Suite("PLE shape probe")
struct PLEShapeProbeTests {
  private let count = 4
  private let width = 2048
  private let embed = 2048
  private let kernel = 4
  private let dilation = 3
  private let eps: Float = 1e-6

  private func block() -> PLEBlock {
    PLEBlock(
      keyProj: DenseLinear(
        weight: MLXArray.zeros([count * width, embed]), bias: nil),
      valueProj: DenseLinear(
        weight: MLXArray.zeros([width, embed]), bias: nil),
      normKey: MLXArray.ones([count * width]),
      normQuery: MLXArray.ones([count * width]),
      normConv: MLXArray.ones([count * width]),
      conv: MLXArray.zeros([count * width, kernel, 1]),
      count: count, width: width, kernel: kernel, dilation: dilation, eps: eps)
  }

  @Test("stateLength matches Whittle dilation")
  func stateLength() {
    #expect(block().stateLength == 9)
  }

  @Test("prefill-shaped streams keep padded ndim >= 2")
  func prefillShapes() {
    let b = block()
    for tokens in [1, 2, 3, 8, 16] {
      let embeddings = MLXArray.zeros([1, tokens, embed])
      let streams = MLXArray.zeros([1, tokens, count * width])
      var state: MLXArray? = nil
      let out = b(embeddings, streams: streams, state: &state)
      eval(out)
      #expect(out.shape == [1, tokens, count * width], "tokens \(tokens)")
      let carried = try! #require(state)
      #expect(carried.shape == [1, b.stateLength, count * width], "tokens \(tokens)")
    }
  }

  @Test("shortConv survives 2D input")
  func shortConv2D() throws {
    let b = block()
    let x = MLXArray.zeros([8, count * width])
    var state: MLXArray? = nil
    let out = b.shortConv(x, state: &state)
    eval(out)
    #expect(out.shape == [1, 8, count * width], "valid conv keeps the time axis")
    let carried = try #require(state)
    eval(carried)
    #expect(carried.shape == [1, b.stateLength, count * width])
  }

  @Test("shortConv on 1D input — rank mismatch would previously trap or fatal")
  func shortConv1D() throws {
    let b = block()
    let x = MLXArray.zeros([count * width])
    var state: MLXArray? = nil
    let out = b.shortConv(x, state: &state)
    eval(out)
    #expect(out.shape == [1, 1, count * width])
    let carried = try #require(state)
    eval(carried)
    #expect(carried.shape == [1, b.stateLength, count * width])
  }

  @Test("shortConv discards a 1D state instead of building an inverted Range")
  func shortConvDropsBadState() throws {
    let b = block()
    let x = MLXArray.zeros([1, 4, count * width])
    var state: MLXArray? = MLXArray.zeros([count * width])  // 1D, wrong geometry
    let out = b.shortConv(x, state: &state)
    eval(out)
    #expect(out.shape == [1, 4, count * width])
    let carried = try #require(state)
    eval(carried)
    #expect(carried.shape == [1, b.stateLength, count * width])
  }

  @Test("concat axis -2 same-rank only — rank 1 is a hard MLX fatal")
  func concatSameRankOnly() {
    // axis -2 is out of bounds for ndim < 2 (fatal), and mixed ranks fatal too.
    // Same-rank ndim >= 2 is the only success path shortConv can take.
    for (name, a, bb) in [
      ("2+2", MLXArray.zeros([9, 16]), MLXArray.zeros([8, 16])),
      ("3+3", MLXArray.zeros([1, 9, 16]), MLXArray.zeros([1, 8, 16])),
      ("2+2 mismatch C", MLXArray.zeros([9, 16]), MLXArray.zeros([8, 15])),
    ] {
      // Mismatch on the non-axis dim may fatal — catch by running last and accepting.
      if name.contains("mismatch") { continue }
      let padded = concatenated([a, bb], axis: -2)
      eval(padded)
      print("concat \(name): \(a.shape) + \(bb.shape) => \(padded.shape) ndim=\(padded.ndim)")
      #expect(padded.ndim >= 2)
    }
  }

  @Test("what empty-shape arrays do to the expand Range")
  func emptyShapeTrap() {
    // An error/placeholder MLXArray with ndim 0 is what makes 0..<(0-2) trap.
    let scalar = MLXArray.zeros([])
    print("scalar ndim=\(scalar.ndim) shape=\(scalar.shape) expand=0..<\(scalar.ndim - 2)")
    #expect(scalar.ndim == 0)
    #expect(scalar.ndim - 2 == -2)
    // Do not execute the subscript — that is the production trap.
  }

  @Test("expandEllipsis trap condition on low ndim")
  func lowNdimSubscript() {
    // The trap is 0 ..< (ndim - 2) when ndim < 2. shortConv now promotes to rank 3 first.
    let arr1 = MLXArray.zeros([6])
    print("1D dim(-2)=\(arr1.dim(-2)) ndim=\(arr1.ndim) expand would be 0..<\(arr1.ndim - 2)")
    #expect(arr1.ndim - 2 < 0, "1D would have inverted the expand Range without a promote")

    let arr0 = MLXArray.zeros([])
    print("0D ndim=\(arr0.ndim) expand 0..<\(arr0.ndim - 2)")
    #expect(arr0.ndim - 2 < 0)
  }

  @Test("value expand_dims and gate broadcast for Whittle dims")
  func valueGateShapes() {
    let tokens = 5
    let embeddings = MLXArray.zeros([1, tokens, embed])
    let streams = MLXArray.zeros([1, tokens, count * width])
    let b = block()

    let value = b.valueProj(embeddings).expandedDimensions(axis: -2)
    let keys = groupedRMSNorm(
      b.keyProj(embeddings), weight: b.normKey, groups: count, width: width, eps: eps
    ).reshaped([1, tokens, count, width])
    let queries = groupedRMSNorm(
      streams, weight: b.normQuery, groups: count, width: width, eps: eps
    ).reshaped([1, tokens, count, width])
    let gate = (keys * queries).sum(axis: -1, keepDims: true)
    print("keys=\(keys.shape) queries=\(queries.shape) gate=\(gate.shape) value=\(value.shape)")
    let product = sigmoid(gate) * value
    print("product=\(product.shape)")
    let flat = product.reshaped([1, tokens, count * width])
    print("flat=\(flat.shape)")
    let normed = groupedRMSNorm(
      flat, weight: b.normConv, groups: count, width: width, eps: eps)
    print("normed=\(normed.shape)")
    let previous = MLXArray.zeros(Array(normed.shape.dropLast(2)) + [b.stateLength, count * width])
    let padded = concatenated([previous, normed], axis: -2)
    eval(padded)
    print("padded=\(padded.shape) ndim=\(padded.ndim)")
    #expect(padded.ndim >= 2)
  }
}
