// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Reference dequantization for the GGML block formats, ported from llama.cpp.

import Foundation

/// Turns one GGML super-block back into floats, one type at a time.
///
/// This is the definition the Metal kernels are checked against, not the path a forward pass
/// takes: it walks a block scalar-for-scalar the way `dequantize_row_*` does in ggml-quants.c,
/// so a kernel that disagrees with it is wrong. Every type here packs 256 elements except the
/// scalar ones, and the scales live inside the block rather than beside it.
public enum GGMLDequant {
  public static let superBlock = 256
  static let iq1Delta: Float = 0.125

  /// Every type this runtime can read. A GGUF naming anything else is refused at load.
  public static let supported: Set<GGMLType> = [
    .f32, .f16, .bf16,
    .q2K, .q4K, .q6K,
    .iq1S, .iq1M, .iq2Xxs, .iq2Xs, .iq2S, .iq3Xxs, .iq3S, .iq4Xs,
  ]

  public static func dequantize(
    _ bytes: UnsafeRawBufferPointer, type: GGMLType, count: Int
  ) throws -> [Float] {
    guard supported.contains(type) else {
      throw BonsaiError.unsupportedModel("GGML type \(type.name) has no dequantizer")
    }
    guard count % type.blockSize == 0 else {
      throw BonsaiError.shapeMismatch(
        "\(count) elements is not a whole number of \(type.name) blocks")
    }
    let blocks = count / type.blockSize
    guard bytes.count >= blocks * type.typeSize else {
      throw BonsaiError.missingWeight(
        "\(type.name) needs \(blocks * type.typeSize) bytes for \(count) elements, "
          + "got \(bytes.count)")
    }

    var out = [Float](repeating: 0, count: count)
    out.withUnsafeMutableBufferPointer { y in
      switch type {
      case .f32: scalar(bytes, y, blocks, 4) { $0.loadUnaligned(as: Float.self) }
      case .f16: scalar(bytes, y, blocks, 2) { half($0.loadUnaligned(as: UInt16.self)) }
      case .bf16:
        scalar(bytes, y, blocks, 2) {
          Float(bitPattern: UInt32($0.loadUnaligned(as: UInt16.self)) << 16)
        }
      case .q2K: blockwise(bytes, y, blocks, type, q2K)
      case .q4K: blockwise(bytes, y, blocks, type, q4K)
      case .q6K: blockwise(bytes, y, blocks, type, q6K)
      case .iq1S: blockwise(bytes, y, blocks, type, iq1S)
      case .iq1M: blockwise(bytes, y, blocks, type, iq1M)
      case .iq2Xxs: blockwise(bytes, y, blocks, type, iq2XXS)
      case .iq2Xs: blockwise(bytes, y, blocks, type, iq2XS)
      case .iq2S: blockwise(bytes, y, blocks, type, iq2S)
      case .iq3Xxs: blockwise(bytes, y, blocks, type, iq3XXS)
      case .iq3S: blockwise(bytes, y, blocks, type, iq3S)
      case .iq4Xs: blockwise(bytes, y, blocks, type, iq4XS)
      default: break
      }
    }
    return out
  }

  public static func dequantize(_ data: Data, type: GGMLType, count: Int) throws -> [Float] {
    try data.withUnsafeBytes { try dequantize($0, type: type, count: count) }
  }

  private static func scalar(
    _ bytes: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>, _ blocks: Int,
    _ stride: Int, _ read: (UnsafeRawPointer) -> Float
  ) {
    let base = bytes.baseAddress!
    for i in 0..<blocks { y[i] = read(base + i * stride) }
  }

  private static func blockwise(
    _ bytes: UnsafeRawBufferPointer, _ y: UnsafeMutableBufferPointer<Float>, _ blocks: Int,
    _ type: GGMLType,
    _ body: (UnsafeRawPointer, UnsafeMutableBufferPointer<Float>, Int) -> Void
  ) {
    let base = bytes.baseAddress!
    for i in 0..<blocks {
      body(base + i * type.typeSize, y, i * type.blockSize)
    }
  }

  static func half(_ bits: UInt16) -> Float { Float(Float16(bitPattern: bits)) }

  private static func u8(_ p: UnsafeRawPointer, _ offset: Int) -> UInt8 {
    p.loadUnaligned(fromByteOffset: offset, as: UInt8.self)
  }

  private static func i8(_ p: UnsafeRawPointer, _ offset: Int) -> Int8 {
    p.loadUnaligned(fromByteOffset: offset, as: Int8.self)
  }

  private static func u16(_ p: UnsafeRawPointer, _ offset: Int) -> UInt16 {
    UInt16(littleEndian: p.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
  }

  private static func u32(_ p: UnsafeRawPointer, _ offset: Int) -> UInt32 {
    UInt32(littleEndian: p.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
  }

  private static func byte(_ value: UInt64, _ index: Int) -> Int {
    Int((value >> (8 * UInt64(index))) & 0xff)
  }

  private static func signedByte(_ value: UInt64, _ index: Int) -> Int {
    Int(Int8(bitPattern: UInt8((value >> (8 * UInt64(index))) & 0xff)))
  }

  private static func byte(_ value: UInt32, _ index: Int) -> Int {
    Int((value >> (8 * UInt32(index))) & 0xff)
  }

  private static func signFlip(_ signs: UInt8, _ j: Int) -> Float {
    signs & GGMLTables.kmaskIq2xs[j] != 0 ? -1 : 1
  }

  private static func q2K(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let d = half(u16(p, 80))
    let dmin = half(u16(p, 82))
    var out = start
    var scale = 0
    for n in stride(from: 0, to: superBlock, by: 128) {
      let quants = 16 + n / 4
      var shift: UInt8 = 0
      for _ in 0..<4 {
        for lane in 0..<2 {
          let sc = u8(p, scale)
          scale += 1
          let dl = d * Float(sc & 0xf)
          let ml = dmin * Float(sc >> 4)
          for l in 0..<16 {
            let q = u8(p, quants + 16 * lane + l)
            y[out] = dl * Float((q >> shift) & 3) - ml
            out += 1
          }
        }
        shift += 2
      }
    }
  }

  /// ql holds the low four bits of every weight, qh the high two, and the sign comes from
  /// subtracting 32 — six bits per weight, in two tables rather than one.
  private static func q6K(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let d = half(u16(p, 208))
    var out = start
    var low = 0
    var high = 128
    var scale = 192

    for _ in stride(from: 0, to: superBlock, by: 128) {
      for l in 0..<32 {
        let bits = u8(p, high + l)
        let q1 = Int(u8(p, low + l) & 0xf) | (Int((bits >> 0) & 3) << 4)
        let q2 = Int(u8(p, low + 32 + l) & 0xf) | (Int((bits >> 2) & 3) << 4)
        let q3 = Int(u8(p, low + l) >> 4) | (Int((bits >> 4) & 3) << 4)
        let q4 = Int(u8(p, low + 32 + l) >> 4) | (Int((bits >> 6) & 3) << 4)

        let group = l / 16
        func scaled(_ index: Int, _ q: Int) -> Float {
          d * Float(i8(p, scale + group + index)) * Float(q - 32)
        }
        y[out + l] = scaled(0, q1)
        y[out + 32 + l] = scaled(2, q2)
        y[out + 64 + l] = scaled(4, q3)
        y[out + 96 + l] = scaled(6, q4)
      }
      out += 128
      low += 64
      high += 32
      scale += 8
    }
  }

  private static func q4K(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let d = half(u16(p, 0))
    let dmin = half(u16(p, 2))

    func scaleMin(_ j: Int) -> (Float, Float) {
      let q = { (i: Int) in u8(p, 4 + i) }
      let sc: UInt8
      let m: UInt8
      if j < 4 {
        sc = q(j) & 63
        m = q(j + 4) & 63
      } else {
        sc = (q(j + 4) & 0xf) | ((q(j - 4) >> 6) << 4)
        m = (q(j + 4) >> 4) | ((q(j) >> 6) << 4)
      }
      return (d * Float(sc), dmin * Float(m))
    }

    var out = start
    var pair = 0
    for j in stride(from: 0, to: superBlock, by: 64) {
      let (d1, m1) = scaleMin(pair)
      let (d2, m2) = scaleMin(pair + 1)
      let quants = 16 + j / 2
      for l in 0..<32 {
        y[out + l] = d1 * Float(u8(p, quants + l) & 0xf) - m1
        y[out + 32 + l] = d2 * Float(u8(p, quants + l) >> 4) - m2
      }
      out += 64
      pair += 2
    }
  }

  private static func iq4XS(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let d = half(u16(p, 0))
    let scalesH = u16(p, 2)
    var out = start
    for ib in 0..<(superBlock / 32) {
      let low = Int((u8(p, 4 + ib / 2) >> (4 * UInt8(ib % 2))) & 0xf)
      let high = Int((scalesH >> (2 * UInt16(ib))) & 3) << 4
      let dl = d * Float((low | high) - 32)
      let qs = 8 + 16 * ib
      for j in 0..<16 {
        let q = u8(p, qs + j)
        y[out + j] = dl * Float(GGMLTables.kvaluesIq4nl[Int(q & 0xf)])
        y[out + 16 + j] = dl * Float(GGMLTables.kvaluesIq4nl[Int(q >> 4)])
      }
      out += 32
    }
  }

  private static func iq2XXS(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let d = half(u16(p, 0))
    var out = start
    for ib32 in 0..<(superBlock / 32) {
      let base = 2 + 8 * ib32
      let aux1 = u32(p, base)
      let aux2 = u32(p, base + 4)
      let db = d * (0.5 + Float(aux2 >> 28)) * 0.25
      for l in 0..<4 {
        let index = Int((aux1 >> (8 * UInt32(l))) & 0xff)
        let entry = GGMLTables.iq2xxsGrid[index]
        let signs = GGMLTables.ksignsIq2xs[Int((aux2 >> (7 * UInt32(l))) & 127)]
        for j in 0..<8 {
          y[out + j] = db * Float(byte(entry, j)) * signFlip(signs, j)
        }
        out += 8
      }
    }
  }

  private static func iq2XS(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let d = half(u16(p, 0))
    var out = start
    for ib32 in 0..<(superBlock / 32) {
      let scale = u8(p, 66 + ib32)
      let db = [
        d * (0.5 + Float(scale & 0xf)) * 0.25,
        d * (0.5 + Float(scale >> 4)) * 0.25,
      ]
      for l in 0..<4 {
        let q = u16(p, 2 + 2 * (4 * ib32 + l))
        let entry = GGMLTables.iq2xsGrid[Int(q & 511)]
        let signs = GGMLTables.ksignsIq2xs[Int(q >> 9)]
        for j in 0..<8 {
          y[out + j] = db[l / 2] * Float(byte(entry, j)) * signFlip(signs, j)
        }
        out += 8
      }
    }
  }

  private static func iq2S(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let d = half(u16(p, 0))
    var out = start
    for ib32 in 0..<(superBlock / 32) {
      let scale = u8(p, 74 + ib32)
      let db = [
        d * (0.5 + Float(scale & 0xf)) * 0.25,
        d * (0.5 + Float(scale >> 4)) * 0.25,
      ]
      let qh = u8(p, 66 + ib32)
      for l in 0..<4 {
        let low = Int(u8(p, 2 + 4 * ib32 + l))
        let high = (Int(qh) << (8 - 2 * l)) & 0x300
        let entry = GGMLTables.iq2sGrid[low | high]
        let signs = u8(p, 34 + 4 * ib32 + l)
        for j in 0..<8 {
          y[out + j] = db[l / 2] * Float(byte(entry, j)) * signFlip(signs, j)
        }
        out += 8
      }
    }
  }

  private static func iq3XXS(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let d = half(u16(p, 0))
    var out = start
    for ib32 in 0..<(superBlock / 32) {
      let aux = u32(p, 2 + 64 + 4 * ib32)
      let db = d * (0.5 + Float(aux >> 28)) * 0.5
      for l in 0..<4 {
        let signs = GGMLTables.ksignsIq2xs[Int((aux >> (7 * UInt32(l))) & 127)]
        let grid1 = GGMLTables.iq3xxsGrid[Int(u8(p, 2 + 8 * ib32 + 2 * l))]
        let grid2 = GGMLTables.iq3xxsGrid[Int(u8(p, 2 + 8 * ib32 + 2 * l + 1))]
        for j in 0..<4 {
          y[out + j] = db * Float(byte(grid1, j)) * signFlip(signs, j)
          y[out + 4 + j] = db * Float(byte(grid2, j)) * signFlip(signs, j + 4)
        }
        out += 8
      }
    }
  }

  private static func iq3S(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let d = half(u16(p, 0))
    let qsBase = 2
    let qhBase = qsBase + 64
    let signBase = qhBase + 8
    let scaleBase = signBase + 32

    var out = start
    for pair in stride(from: 0, to: superBlock / 32, by: 2) {
      let scale = u8(p, scaleBase + pair / 2)
      let db = [
        d * Float(1 + 2 * Int(scale & 0xf)),
        d * Float(1 + 2 * Int(scale >> 4)),
      ]
      for group in 0..<2 {
        let qh = Int(u8(p, qhBase + pair + group))
        let qs = qsBase + 8 * (pair + group)
        let signs = signBase + 4 * (pair + group)
        for l in 0..<4 {
          let grid1 = GGMLTables.iq3sGrid[
            Int(u8(p, qs + 2 * l)) | ((qh << (8 - 2 * l)) & 256)]
          let grid2 = GGMLTables.iq3sGrid[
            Int(u8(p, qs + 2 * l + 1)) | ((qh << (7 - 2 * l)) & 256)]
          let sign = u8(p, signs + l)
          for j in 0..<4 {
            y[out + j] = db[group] * Float(byte(grid1, j)) * signFlip(sign, j)
            y[out + 4 + j] = db[group] * Float(byte(grid2, j)) * signFlip(sign, j + 4)
          }
          out += 8
        }
      }
    }
  }

  private static func iq1S(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let d = half(u16(p, 0))
    var out = start
    for ib in 0..<(superBlock / 32) {
      let qh = u16(p, 34 + 2 * ib)
      let dl = d * Float(2 * Int((qh >> 12) & 7) + 1)
      let delta: Float = qh & 0x8000 != 0 ? -iq1Delta : iq1Delta
      for l in 0..<4 {
        let index = Int(u8(p, 2 + 4 * ib + l)) | ((Int(qh >> (3 * l)) & 7) << 8)
        let entry = GGMLTables.iq1sGrid[index]
        for j in 0..<8 {
          y[out + j] = dl * (Float(signedByte(entry, j)) + delta)
        }
        out += 8
      }
    }
  }

  private static func iq1M(
    _ p: UnsafeRawPointer, _ y: UnsafeMutableBufferPointer<Float>, _ start: Int
  ) {
    let sc = (0..<4).map { u16(p, 48 + 2 * $0) }
    let packed =
      (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
    let d = half(packed)

    var out = start
    for ib in 0..<(superBlock / 32) {
      let word = sc[ib / 2]
      let shift = UInt16(6 * (ib % 2))
      let dl1 = d * Float(2 * Int((word >> shift) & 0x7) + 1)
      let dl2 = d * Float(2 * Int((word >> (shift + 3)) & 0x7) + 1)

      let qs = 4 * ib
      let qh0 = Int(u8(p, 32 + 2 * ib))
      let qh1 = Int(u8(p, 32 + 2 * ib + 1))
      let index = [
        Int(u8(p, qs)) | ((qh0 << 8) & 0x700),
        Int(u8(p, qs + 1)) | ((qh0 << 4) & 0x700),
        Int(u8(p, qs + 2)) | ((qh1 << 8) & 0x700),
        Int(u8(p, qs + 3)) | ((qh1 << 4) & 0x700),
      ]
      let delta: [Float] = [
        qh0 & 0x08 != 0 ? -iq1Delta : iq1Delta,
        qh0 & 0x80 != 0 ? -iq1Delta : iq1Delta,
        qh1 & 0x08 != 0 ? -iq1Delta : iq1Delta,
        qh1 & 0x80 != 0 ? -iq1Delta : iq1Delta,
      ]
      for l in 0..<4 {
        let dl = l < 2 ? dl1 : dl2
        let entry = GGMLTables.iq1sGrid[index[l]]
        for j in 0..<8 {
          y[out + j] = dl * (Float(signedByte(entry, j)) + delta[l])
        }
        out += 8
      }
    }
  }
}
