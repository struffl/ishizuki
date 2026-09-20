// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

/// The GPU decode against the CPU one, which is itself held to ggml's output. A kernel that
/// misreads a block's scales still produces plausible numbers, so the check is per element
/// rather than on a norm.
@Suite("GGML kernels")
struct GGMLKernelTests {
  private struct Reference {
    let type: GGMLType
    let bytes: [UInt8]
    let count: Int
  }

  private static func load() throws -> [Reference] {
    let url = try #require(
      Bundle.module.url(
        forResource: "ggml-reference", withExtension: "bin", subdirectory: "Fixtures")
        ?? Bundle.module.url(forResource: "ggml-reference", withExtension: "bin"))
    let data = try Data(contentsOf: url)
    var cursor = 8
    func u32() -> Int {
      let value = data.withUnsafeBytes {
        UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: cursor, as: UInt32.self))
      }
      cursor += 4
      return Int(value)
    }
    let count = u32()
    var out: [Reference] = []
    for _ in 0..<count {
      let raw = UInt32(u32())
      _ = u32()
      let byteCount = u32()
      let floatCount = u32()
      let type = try #require(GGMLType(rawValue: raw))
      let bytes = [UInt8](data.subdata(in: cursor..<(cursor + byteCount)))
      cursor += byteCount + 4 * floatCount
      out.append(Reference(type: type, bytes: bytes, count: floatCount))
    }
    return out
  }

  @Test("the Metal decode agrees with the reference decode on every block type")
  func matchesReference() throws {
    for reference in try Self.load() where reference.type.isQuantized {
      let blocks = MLXArray(reference.bytes)
      guard
        let gpu = GGMLKernels.dequantize(
          blocks: blocks, type: reference.type, shape: [reference.count], dtype: .float32)
      else {
        Issue.record("\(reference.type.name) produced no kernel")
        continue
      }
      eval(gpu)
      let actual = gpu.asArray(Float.self)
      let expected = try GGMLDequant.dequantize(
        Data(reference.bytes), type: reference.type, count: reference.count)

      var worst: Float = 0
      var worstIndex = -1
      for i in 0..<reference.count {
        let scale = max(abs(expected[i]), 1e-6)
        let error = abs(actual[i] - expected[i]) / scale
        if error > worst {
          worst = error
          worstIndex = i
        }
      }
      let index = max(worstIndex, 0)
      let detail =
        "\(reference.type.name): element \(worstIndex) off by \(worst) relative "
        + "(\(actual[index]) vs \(expected[index]))"
      #expect(worst < 1e-5, "\(detail)")
    }
  }

  @Test("a scalar type has no block kernel and falls back")
  func scalarTypesFallBack() {
    let blocks = MLXArray([UInt8](repeating: 0, count: 64))
    #expect(GGMLKernels.dequantize(blocks: blocks, type: .f32, shape: [16]) == nil)
    #expect(GGMLKernels.dequantize(blocks: blocks, type: .q3_K, shape: [256]) == nil)
  }
}

/// The fused matvec against the same decode it is built from, expanded and multiplied the
/// ordinary way. A kernel that walks blocks or rows wrongly still returns numbers of the right
/// magnitude, so the comparison is per output element against the term-by-term bound.
@Suite("GGML matvec")
struct GGMLMatvecTests {
  /// Bit 6 is cleared in every byte so that no two of them can line up as an fp16 with a
  /// saturated exponent. A block scale of infinity would make both sides of the comparison
  /// NaN and prove nothing; the decode itself is already pinned to ggml over the full byte
  /// range by the dequantization suite.
  private func blocks(_ type: GGMLType, rows: Int, k: Int, seed: UInt64) -> [UInt8] {
    var state = seed
    let count = rows * (k / type.blockSize) * type.typeSize
    return (0..<count).map { _ in
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      return UInt8((state >> 33) & 0xbf)
    }
  }

  @Test("the fused matvec matches expanding the weight and multiplying")
  func matchesExpandedMatmul() throws {
    let types: [GGMLType] = [
      .q2_K, .q4_K, .q6_K, .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_xs,
    ]
    let k = 512
    let rows = 17

    for type in types {
      let raw = MLXArray(blocks(type, rows: rows, k: k, seed: 0x5EED))
      let weight = try #require(
        GGMLKernels.dequantize(blocks: raw, type: type, shape: [rows, k], dtype: .float32))

      for m in GGMLKernels.matvecBatch {
        let x = MLXRandom.normal([m, k]).asType(.float32)
        let expected = matmul(x, weight.T)
        let actual = try #require(
          GGMLKernels.matvec(x, blocks: raw, type: type, outputDim: rows))
        eval(expected, actual)

        // Cancellation between large terms makes the result's own magnitude a bad yardstick,
        // so each element is judged against the sum of the magnitudes that formed it.
        let bound = matmul(abs(x), abs(weight).T)
        let error = abs(actual - expected) / maximum(bound, MLXArray(Float(1e-6)))
        eval(error)
        let worst = error.max().item(Float.self)
        #expect(worst < 1e-5, "\(type.name) at batch \(m): worst relative term error \(worst)")
      }
    }
  }

  /// The gather walks `(row, block)` pairs several to a thread and writes into a different
  /// tensor's layout, so a row that lands one slot over still looks like a plausible embedding.
  /// Expanding the whole table and indexing it is the definition it has to match.
  @Test("gathering rows matches indexing the expanded table")
  func gatherMatchesIndexing() throws {
    let types: [GGMLType] = [
      .q2_K, .q4_K, .q6_K, .iq1_s, .iq1_m, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_xs,
    ]
    let k = 512
    let rows = 40
    let ids = MLXArray(
      [0, 39, 1, 17, 17, 38, 2, 23, 9, 4, 31, 12, 30, 5, 21, 8, 11].map(Int32.init))

    for type in types {
      let raw = MLXArray(blocks(type, rows: rows, k: k, seed: 0xC0FFEE))
      let table = try #require(
        GGMLKernels.dequantize(blocks: raw, type: type, shape: [rows, k], dtype: .float32))
      let actual = try #require(
        GGMLKernels.gather(ids: ids, blocks: raw, type: type, inputDim: k, dtype: .float32))
      let expected = table[ids]
      eval(actual, expected)
      #expect(actual.shape == [ids.size, k], "\(type.name) gathered the wrong shape")
      let worst = abs(actual - expected).max().item(Float.self)
      #expect(worst == 0, "\(type.name): gathered rows differ from the expanded table")
    }
  }

  @Test("a batch the fused path does not cover falls back")
  func refusesWideBatch() {
    let type = GGMLType.iq2_xs
    let raw = MLXArray(blocks(type, rows: 4, k: 256, seed: 1))
    let wide = MLXRandom.normal([GGMLKernels.matvecBatch.upperBound + 1, 256])
    #expect(GGMLKernels.matvec(wide, blocks: raw, type: type, outputDim: 4) == nil)

    let ragged = MLXRandom.normal([1, 100])
    #expect(GGMLKernels.matvec(ragged, blocks: raw, type: type, outputDim: 4) == nil)
  }
}
