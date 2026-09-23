// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// EXL3 trellis-coded projections, decoded inside the matvec rather than before it.

import Foundation
import MLX

public struct EXL3Tensor: @unchecked Sendable {
  public enum Codebook: Int, Sendable {
    case threeInst = 0
    case mcg = 1
    case mul1 = 2
  }

  public let trellis: MLXArray
  public let inputScales: MLXArray
  public let outputScales: MLXArray
  public let bits: Double
  public let codebook: Codebook

  public var inputDim: Int { trellis.dim(0) * 16 }
  public var outputDim: Int { trellis.dim(1) * 16 }
  public var wholeBits: Int { Int(bits) }
  public var halfBit: Bool { bits != bits.rounded(.down) }

  public static let supportedBits: Set<Double> = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8]

  public init(
    trellis: MLXArray, inputScales: MLXArray, outputScales: MLXArray, codebook: Codebook
  ) throws {
    guard trellis.ndim == 3, trellis.dtype == .int16 else {
      throw BonsaiError.shapeMismatch("EXL3 trellis must be int16 [k/16, n/16, words]")
    }
    let bits = Double(trellis.dim(2)) / 16
    guard Self.supportedBits.contains(bits) else {
      throw BonsaiError.unsupportedModel("EXL3 tile of \(trellis.dim(2)) words is not a rate")
    }
    guard bits.rounded(.down) == bits || codebook == .mul1 else {
      throw BonsaiError.unsupportedModel("half-integer EXL3 rate \(bits) needs the mul1 codebook")
    }
    let k = trellis.dim(0) * 16
    let n = trellis.dim(1) * 16
    guard k % 128 == 0, n % 128 == 0 else {
      throw BonsaiError.shapeMismatch("EXL3 \(k)x\(n) is not a whole number of 128 blocks")
    }
    guard inputScales.size == k, outputScales.size == n else {
      throw BonsaiError.shapeMismatch(
        "EXL3 scales are \(inputScales.size)/\(outputScales.size), expected \(k)/\(n)")
    }
    self.trellis = trellis
    self.inputScales = inputScales.reshaped([k])
    self.outputScales = outputScales.reshaped([n])
    self.bits = bits
    self.codebook = codebook
  }

  public init?(store: WeightStore, key: String) throws {
    guard let trellis = store.optional(key + ".trellis") else { return nil }
    let codebook: Codebook =
      store.has(key + ".mul1") ? .mul1 : store.has(key + ".mcg") ? .mcg : .threeInst
    try self.init(
      trellis: trellis,
      inputScales: try Self.scales(store, key, unpacked: "suh", packed: "su"),
      outputScales: try Self.scales(store, key, unpacked: "svh", packed: "sv"),
      codebook: codebook)
  }

  private static func scales(
    _ store: WeightStore, _ key: String, unpacked: String, packed: String
  ) throws -> MLXArray {
    if let scales = store.optional("\(key).\(unpacked)") { return scales }
    let words = try store("\(key).\(packed)").view(dtype: .uint16).asType(.int32)
    let bit = (words.expandedDimensions(axis: -1) >> MLXArray(Int32(0)..<Int32(16))) & 1
    return (1 - 2 * bit.reshaped([-1])).asType(.float16)
  }
}

public enum EXL3Kernels {
  public static let matvecRows = 1...8

  public static func apply(_ x: MLXArray, _ t: EXL3Tensor) -> MLXArray {
    let shape = x.shape
    let rows = x.size / t.inputDim
    let xh = rotate(x.reshaped([rows, t.inputDim]) * t.inputScales.asType(x.dtype))
      .asType(x.dtype)
    var yh: MLXArray
    if matvecRows.contains(rows), let y = matvec(xh, t) {
      yh = y
    } else if rows <= 2 * matvecRows.upperBound,
      let lo = matvec(xh[0..<8], t), let hi = matvec(xh[8...], t)
    {
      yh = concatenated([lo, hi], axis: 0)
    } else if let w = dequantize(t, dtype: x.dtype) {
      yh = matmul(xh, w).asType(.float32)
    } else {
      return MLXArray.zeros(Array(shape.dropLast()) + [t.outputDim], dtype: x.dtype)
    }
    let y = rotate(yh) * t.outputScales.asType(.float32)
    return y.asType(x.dtype).reshaped(Array(shape.dropLast()) + [t.outputDim])
  }

  static func rotate(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    return hadamardTransform(
      x.asType(.float32).reshaped([-1, 128]), scale: 1 / Float(128).squareRoot()
    ).reshaped(shape)
  }

  private static func template(_ t: EXL3Tensor) -> [(String, any KernelTemplateArg)] {
    [
      ("KA", t.wholeBits), ("HALF", t.halfBit ? 1 : 0), ("CB", t.codebook.rawValue),
      ("K", t.inputDim), ("N", t.outputDim),
    ]
  }

  public static func dequantize(_ t: EXL3Tensor, dtype: DType = .float16) -> MLXArray? {
    #if canImport(Metal)
      guard let dequantizeKernel else { return nil }
      let tiles = t.trellis.dim(0) * t.trellis.dim(1)
      return dequantizeKernel(
        [t.trellis],
        template: template(t) + [("OT", dtype)],
        grid: (tiles * 32, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[t.inputDim, t.outputDim]],
        outputDTypes: [dtype])[0]
    #else
      return nil
    #endif
  }

  public static func matvec(_ x: MLXArray, _ t: EXL3Tensor) -> MLXArray? {
    #if canImport(Metal)
      guard let matvecKernel, x.ndim == 2, x.dim(1) == t.inputDim else { return nil }
      let m = x.dim(0)
      guard matvecRows.contains(m) else { return nil }
      return matvecKernel(
        [x, t.trellis],
        template: template(t) + [("IT", x.dtype), ("vecs", m)],
        grid: (t.trellis.dim(1) * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[m, t.outputDim]],
        outputDTypes: [.float32])[0]
    #else
      return nil
    #endif
  }

  #if canImport(Metal)
    static let header = """
      inline float exl3_decode(uint s, int cb) {
        if (cb == 2) {
          uint x = s * 0x83DCD12Du;
          uint p = (x & 0x00ff00ffu) + ((x >> 8) & 0x00ff00ffu);
          half h = as_type<half>(ushort((p & 0xffffu) + (p >> 16) + 0x6400u));
          return float(fma(h, as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931))));
        }
        uint x = cb == 1 ? s * 0xCBAC1FEDu : s * 89226354u + 64248484u;
        x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
        return float(as_type<half>(ushort(x)) + as_type<half>(ushort(x >> 16)));
      }

      inline uint exl3_end(uint p, uint ka, bool half_k) {
        return half_k ? (p >> 1) * (2 * ka + 1) + ka + (p & 1) * (ka + 1) : (p + 1) * ka;
      }

      template <uint KA, bool HK>
      inline uint exl3_rel(uint i) {
        return HK ? (i >> 1) * (2 * KA + 1) + (i & 1) * (KA + 1) + 16 : i * KA + 16;
      }

      template <uint KA, bool HK, uint RING, uint WORDS, int CB>
      inline void exl3_decode8(const device uint* tile, uint lane, thread float* v) {
        uint lo = exl3_end(lane * 8, KA, HK) + RING - 16;
        uint i0 = lo / 32;
        uint sh = lo % 32;
        uint j0 = i0 >= WORDS ? i0 - WORDS : i0;
        uint j1 = j0 + 1 >= WORDS ? j0 + 1 - WORDS : j0 + 1;
        uint j2 = j1 + 1 >= WORDS ? j1 + 1 - WORDS : j1 + 1;
        ulong w0 = tile[j0], w1 = tile[j1], w2 = tile[j2];
        ulong a = ((w0 << 32) | w1) << sh | (sh ? (w2 >> (32 - sh)) : 0);
        ulong c = ((w1 << 32) | w2) << sh;
        #pragma unroll
        for (uint i = 0; i < 8; i++) {
          uint r = exl3_rel<KA, HK>(i);
          uint s = r <= 64 ? uint(a >> (64 - r)) : uint(c >> (96 - r));
          v[i] = exl3_decode(s & 0xffffu, CB);
        }
      }
      """

    static let common = """
        constexpr bool half_k = HALF != 0;
        constexpr uint ring = half_k ? 128 * (2 * KA + 1) : 256 * KA;
        constexpr uint words = ring / 32;
        constexpr uint tiles_n = N / 16;
        constexpr uint tiles_k = K / 16;
        uint lane = thread_index_in_simdgroup;
        uint r0 = (lane % 4) * 2;
      """

    private static let dequantizeKernel: MLXFast.MLXFastKernel? = MLXFast.metalKernel(
      name: "exl3_dequantize",
      inputNames: ["trellis"],
      outputNames: ["w"],
      source: common + """
          uint tile = thread_position_in_grid.x / 32;
          if (tile >= tiles_k * tiles_n) return;
          uint row = (tile / tiles_n) * 16;
          uint col = (tile % tiles_n) * 16 + lane / 4;
          float v[8];
          exl3_decode8<KA, half_k, ring, words, CB>(
              (const device uint*)trellis + tile * words, lane, v);
          uint rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
          for (uint i = 0; i < 8; i++) {
            w[(row + rows[i % 4]) * N + col + (i / 4) * 8] = static_cast<OT>(v[i]);
          }
        """,
      header: header)

    private static let matvecKernel: MLXFast.MLXFastKernel? = MLXFast.metalKernel(
      name: "exl3_matvec",
      inputNames: ["x", "trellis"],
      outputNames: ["y"],
      source: common + """
          constexpr int SG = 8;
          uint sg = simdgroup_index_in_threadgroup;
          uint tn = threadgroup_position_in_grid.x;
          float acc0[vecs];
          float acc1[vecs];
          for (int m = 0; m < vecs; m++) { acc0[m] = 0.0f; acc1[m] = 0.0f; }
          const device uint* column = (const device uint*)trellis + tn * words;
          for (uint tk = sg; tk < tiles_k; tk += SG) {
            float v[8];
            exl3_decode8<KA, half_k, ring, words, CB>(column + tk * tiles_n * words, lane, v);
            const device IT* xr = x + tk * 16 + r0;
            for (int m = 0; m < vecs; m++) {
              float x0 = xr[m * K], x1 = xr[m * K + 1], x2 = xr[m * K + 8], x3 = xr[m * K + 9];
              acc0[m] += x0 * v[0] + x1 * v[1] + x2 * v[2] + x3 * v[3];
              acc1[m] += x0 * v[4] + x1 * v[5] + x2 * v[6] + x3 * v[7];
            }
          }
          threadgroup float red[SG][vecs][16];
          for (int m = 0; m < vecs; m++) {
            float a = acc0[m], b = acc1[m];
            a += simd_shuffle_xor(a, 1);
            a += simd_shuffle_xor(a, 2);
            b += simd_shuffle_xor(b, 1);
            b += simd_shuffle_xor(b, 2);
            if (lane % 4 == 0) {
              red[sg][m][lane / 4] = a;
              red[sg][m][lane / 4 + 8] = b;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          uint t = thread_index_in_threadgroup;
          if (t < vecs * 16) {
            int m = t / 16;
            uint c = t % 16;
            float s = 0.0f;
            for (int g = 0; g < SG; g++) s += red[g][m][c];
            y[m * N + tn * 16 + c] = s;
          }
        """,
      header: header)
  #endif
}
