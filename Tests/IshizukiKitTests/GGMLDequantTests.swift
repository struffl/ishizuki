// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import Testing

@testable import IshizukiKit

/// Checked against llama.cpp itself: `Scripts/gen-ggml-reference.c` feeds pseudo-random blocks
/// through ggml's own `to_float` and records both sides, so a drift in either the block layout
/// or the codebook indexing shows up as an exact mismatch rather than a quality regression.
@Suite("GGML dequantization")
struct GGMLDequantTests {
  private struct Case {
    let type: GGMLType
    let bytes: Data
    let expected: [Float]
  }

  private static func loadReference() throws -> [Case] {
    let url = try #require(
      Bundle.module.url(
        forResource: "ggml-reference", withExtension: "bin", subdirectory: "Fixtures")
        ?? Bundle.module.url(forResource: "ggml-reference", withExtension: "bin"))
    let data = try Data(contentsOf: url)
    #expect(data.prefix(8) == Data("GGMLREF1".utf8))

    var cursor = 8
    func u32() -> Int {
      let value = data.withUnsafeBytes {
        UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: cursor, as: UInt32.self))
      }
      cursor += 4
      return Int(value)
    }

    let count = u32()
    var cases: [Case] = []
    for _ in 0..<count {
      let raw = UInt32(u32())
      _ = u32()
      let byteCount = u32()
      let floatCount = u32()
      let type = try #require(GGMLType(rawValue: raw))
      let bytes = data.subdata(in: cursor..<(cursor + byteCount))
      cursor += byteCount
      var expected = [Float](repeating: 0, count: floatCount)
      data.withUnsafeBytes { source in
        for i in 0..<floatCount {
          expected[i] = source.loadUnaligned(
            fromByteOffset: cursor + 4 * i, as: Float.self)
        }
      }
      cursor += 4 * floatCount
      cases.append(Case(type: type, bytes: bytes, expected: expected))
    }
    return cases
  }

  @Test("every block type reproduces ggml's own to_float bit for bit")
  func matchesGGML() throws {
    let cases = try Self.loadReference()
    #expect(cases.count == 13)

    var seen: Set<GGMLType> = []
    for testCase in cases {
      seen.insert(testCase.type)
      let actual = try GGMLDequant.dequantize(
        testCase.bytes, type: testCase.type, count: testCase.expected.count)
      #expect(actual.count == testCase.expected.count)

      var mismatches: [String] = []
      for (i, (got, want)) in zip(actual, testCase.expected).enumerated()
      where got.bitPattern != want.bitPattern && !(got.isNaN && want.isNaN) {
        if mismatches.count < 4 { mismatches.append("[\(i)] got \(got), want \(want)") }
      }
      #expect(
        mismatches.isEmpty,
        "\(testCase.type.name): \(mismatches.joined(separator: "; "))")
    }

    for type in GGMLDequant.supported {
      #expect(seen.contains(type), "\(type.name) has no reference vector")
    }
  }

  @Test("a type with no dequantizer is refused rather than read as zeroes")
  func refusesUnsupported() {
    let bytes = Data(repeating: 0, count: 1024)
    #expect(throws: BonsaiError.self) {
      _ = try GGMLDequant.dequantize(bytes, type: .q3_K, count: 256)
    }
  }

  @Test("a short buffer is refused rather than read past its end")
  func refusesShortBuffer() {
    let bytes = Data(repeating: 0, count: GGMLType.iq2_xs.typeSize - 1)
    #expect(throws: BonsaiError.self) {
      _ = try GGMLDequant.dequantize(bytes, type: .iq2_xs, count: 256)
    }
    #expect(throws: BonsaiError.self) {
      _ = try GGMLDequant.dequantize(
        Data(repeating: 0, count: 1024), type: .iq2_xs, count: 100)
    }
  }
}
