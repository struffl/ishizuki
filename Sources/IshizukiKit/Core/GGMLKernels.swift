// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Metal decoding of the GGML block formats, one thread per super-block.

import Foundation
import MLX

/// The codebooks as flat signed bytes, ready to hand to a kernel.
///
/// `ggml-common.h` packs each entry into a 64- or 32-bit word and casts it to a byte pointer;
/// unpacking once here means a kernel indexes `grid[width * entry + j]` rather than shifting.
/// `iq1s_grid` is the one read signed upstream, and the wrap that gives is the format.
enum GGMLGrids {
  static let iq2xxs = expand(GGMLTables.iq2xxs_grid, width: 8)
  static let iq2xs = expand(GGMLTables.iq2xs_grid, width: 8)
  static let iq2s = expand(GGMLTables.iq2s_grid, width: 8)
  static let iq1s = expand(GGMLTables.iq1s_grid, width: 8)
  static let iq3xxs = expand(GGMLTables.iq3xxs_grid, width: 4)
  static let iq3s = expand(GGMLTables.iq3s_grid, width: 4)

  private static func expand(_ entries: [UInt64], width: Int) -> [Int8] {
    var out = [Int8](repeating: 0, count: entries.count * width)
    for (i, entry) in entries.enumerated() {
      for j in 0..<width {
        out[width * i + j] = Int8(bitPattern: UInt8((entry >> (8 * UInt64(j))) & 0xff))
      }
    }
    return out
  }

  private static func expand(_ entries: [UInt32], width: Int) -> [Int8] {
    expand(entries.map(UInt64.init), width: width)
  }

  #if canImport(Metal)
    nonisolated(unsafe) static let buffers: [MLXArray] = [
      MLXArray(iq2xxs), MLXArray(iq2xs), MLXArray(iq2s), MLXArray(iq1s),
      MLXArray(iq3xxs), MLXArray(iq3s),
      MLXArray(GGMLTables.ksigns_iq2xs), MLXArray(GGMLTables.kvalues_iq4nl),
    ]
  #endif
}

/// Runs a tensor's GGML blocks on the GPU, either expanded or folded into a matvec.
///
/// A block is self-contained — its scales live in its own bytes — so nothing is shared between
/// threads and decoding is a plain map. All three kernels are built from one body per format,
/// and a body ends at `GGML_ACC8`: the dequantizer and the gather define it to store the eight
/// weights a lane holds, the matvec to multiply them into a running dot product and never write
/// the weight down. A format read three ways is a format that can be read three different ways,
/// so there is one.
///
/// Every body is written for the same division of labour. A simdgroup takes one block, lane `L`
/// takes the eight weights at offset `8L`, and the lanes of a simdgroup therefore read the
/// block's per-group bytes — and, in the matvec, the activations — as single contiguous runs.
public enum GGMLKernels {
  /// Blocks as raw bytes to a dense array of `shape`. Nil when Metal is unavailable or the
  /// type has no decoder, so callers fall back to the CPU reference.
  public static func dequantize(
    blocks: MLXArray, type: GGMLType, shape: [Int], dtype: DType = .bfloat16
  ) -> MLXArray? {
    #if canImport(Metal)
      guard let dequantKernel, GGMLDequant.supported.contains(type), type.isQuantized else {
        return nil
      }
      let count = shape.reduce(1, *)
      guard count % type.blockSize == 0 else { return nil }
      let blockCount = count / type.blockSize
      guard blocks.size >= blockCount * type.typeSize else { return nil }

      let threads = 256
      let work = blockCount * 32
      let groups = (work + threads - 1) / threads
      let outputs = dequantKernel(
        [blocks] + GGMLGrids.buffers + [blockCount],
        template: [
          ("OT", dtype),
          ("qtype", Int(type.rawValue)),
          ("block_bytes", type.typeSize),
          ("block_elems", type.blockSize),
          ("grid_bytes", gridBytes(type)),
        ],
        grid: (groups * threads, 1, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [[count]],
        outputDTypes: [dtype])
      return outputs[0].reshaped(shape)
    #else
      return nil
    #endif
  }

  /// The rows named by `ids`, decoded into `[ids.count, inputDim]`.
  ///
  /// An embedding table at 1.3 billion entries cannot be expanded to be indexed, and it does
  /// not have to be: a row's blocks are contiguous and self-contained, so only the rows asked
  /// for are ever touched.
  public static func gather(
    ids: MLXArray, blocks: MLXArray, type: GGMLType, inputDim: Int, dtype: DType = .bfloat16
  ) -> MLXArray? {
    #if canImport(Metal)
      guard let gatherKernel, GGMLDequant.supported.contains(type), type.isQuantized else {
        return nil
      }
      guard inputDim % type.blockSize == 0 else { return nil }
      let count = ids.size
      guard count > 0 else { return nil }
      let perRow = inputDim / type.blockSize

      let threads = 256
      let work = count * perRow
      let groups = (work * 32 + threads - 1) / threads
      let outputs = gatherKernel(
        [ids.asType(.int32).reshaped([count]), blocks] + GGMLGrids.buffers + [inputDim, work],
        template: [
          ("OT", dtype),
          ("qtype", Int(type.rawValue)),
          ("block_bytes", type.typeSize),
          ("block_elems", type.blockSize),
          ("grid_bytes", gridBytes(type)),
        ],
        grid: (groups * threads, 1, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [[count, inputDim]],
        outputDTypes: [dtype])
      return outputs[0]
    #else
      return nil
    #endif
  }

  /// How many rows of `x` one matvec launch will carry.
  ///
  /// A lane holds one accumulator per row, so a wide launch is a wide live set; past about forty
  /// rows the registers spill and the kernel falls off a cliff. Expanding the weight and calling
  /// a real matmul overtakes the fused path somewhere in the high twenties, which leaves this
  /// well inside both limits.
  public static let matvecBatch = 1...16

  /// Bytes of the codebook a type indexes into, or zero for one that decodes arithmetically.
  /// The matvec stages this much into threadgroup memory before any row starts.
  static func gridBytes(_ type: GGMLType) -> Int {
    switch type {
    case .iq2_xxs: 2048
    case .iq2_xs: 4096
    case .iq2_s: 8192
    case .iq1_s, .iq1_m: 16384
    case .iq3_xxs: 1024
    case .iq3_s: 2048
    default: 0
    }
  }

  /// `x [M, K]` against the blocks of `w [N, K]`, without ever materialising `w`.
  public static func matvec(
    _ x: MLXArray, blocks: MLXArray, type: GGMLType, outputDim: Int
  ) -> MLXArray? {
    #if canImport(Metal)
      guard let matvecKernel, GGMLDequant.supported.contains(type), type.isQuantized else {
        return nil
      }
      guard x.ndim == 2 else { return nil }
      let m = x.dim(0)
      let k = x.dim(1)
      guard matvecBatch.contains(m), k % type.blockSize == 0, outputDim > 0 else { return nil }
      let rowBytes = k / type.blockSize * type.typeSize
      guard blocks.size >= outputDim * rowBytes else { return nil }

      let lanes = 32
      let threads = 256
      let groups = (outputDim * lanes + threads - 1) / threads
      let outputs = matvecKernel(
        [x, blocks] + GGMLGrids.buffers + [k, outputDim],
        template: [
          ("IT", x.dtype),
          ("qtype", Int(type.rawValue)),
          ("block_bytes", type.typeSize),
          ("block_elems", type.blockSize),
          ("vecs", m),
          ("grid_bytes", gridBytes(type)),
        ],
        grid: (groups * threads, 1, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [[m, outputDim]],
        outputDTypes: [x.dtype])
      return outputs[0]
    #else
      return nil
    #endif
  }

  #if canImport(Metal)
    private static let dequantKernel: MLXFast.MLXFastKernel? = {
      MLXFast.metalKernel(
        name: "ggml_dequantize",
        inputNames: [
          "blocks", "g_iq2xxs", "g_iq2xs", "g_iq2s", "g_iq1s", "g_iq3xxs", "g_iq3s",
          "ksigns", "kvalues", "n_blocks",
        ],
        outputNames: ["y"],
        source: macros + blockMacros + storeEach + blockTables + dequantProlog + blockChain)
    }()

    private static let gatherKernel: MLXFast.MLXFastKernel? = {
      MLXFast.metalKernel(
        name: "ggml_gather",
        inputNames: [
          "ids", "w", "g_iq2xxs", "g_iq2xs", "g_iq2s", "g_iq1s", "g_iq3xxs", "g_iq3s",
          "ksigns", "kvalues", "K", "n_work",
        ],
        outputNames: ["y"],
        source: macros + blockMacros + storeEach + blockTables + gatherProlog + blockChain)
    }()

    private static let matvecKernel: MLXFast.MLXFastKernel? = {
      MLXFast.metalKernel(
        name: "ggml_matvec",
        inputNames: [
          "x", "w", "g_iq2xxs", "g_iq2xs", "g_iq2s", "g_iq1s", "g_iq3xxs", "g_iq3s",
          "ksigns", "kvalues", "K", "N",
        ],
        outputNames: ["y"],
        source: macros + blockMacros + accumulateEach + blockTables + matvecProlog + blockChain
          + matvecEpilog)
    }()

    private static let macros = """
          #define GGML_HALF(p) ((float)as_type<half>(*(device const ushort *)(p)))
          #define GGML_U32(p) ((uint)(p)[0] | ((uint)(p)[1] << 8) \\
                              | ((uint)(p)[2] << 16) | ((uint)(p)[3] << 24))

      """

    /// A simdgroup per block, and the eight outputs a lane holds written as two vectors.
    private static let dequantProlog = """
          const uint gid = thread_position_in_grid.x;
          const uint bi = gid / 32;
          const int lane = (int)(gid % 32);
          if (bi >= (uint)n_blocks) { return; }

          device const uchar *b = blocks + (ulong)bi * block_bytes;
          device vec<OT, 4> *yp =
              (device vec<OT, 4> *)(y + (ulong)bi * block_elems + 8 * lane);

      """

    /// The same, over `(row, block)` pairs rather than a tensor's blocks in order.
    private static let gatherProlog = """
          const uint gid = thread_position_in_grid.x;
          const uint unit = gid / 32;
          const int lane = (int)(gid % 32);
          if (unit >= (uint)n_work) { return; }

          const int nblk = K / block_elems;
          const int slot = (int)unit / nblk;
          const int blk = (int)unit % nblk;
          const int row = ids[slot];

          device const uchar *b = w + (ulong)row * nblk * block_bytes
                                + (ulong)blk * block_bytes;
          device vec<OT, 4> *yp =
              (device vec<OT, 4> *)(y + (ulong)slot * K + blk * block_elems + 8 * lane);

      """

    /// The matvec's half of the pair: a group's weights meet the activations they line up with,
    /// and the group's scale comes out of the sum rather than being applied to each weight. The
    /// `M` form is for the formats that subtract a block minimum, which comes out the same way.
    private static let accumulateEach = """
          #define GGML_ACC8(db, q0, q1) { \\
              for (int _m = 0; _m < vecs; ++_m) { \\
                  device const vec<IT, 4> *_xv = (device const vec<IT, 4> *)(xrow + _m * K); \\
                  const float4 _p = float4(_xv[0]) * (q0) + float4(_xv[1]) * (q1); \\
                  acc[_m] += (db) * (_p.x + _p.y + _p.z + _p.w); } }

          #define GGML_ACC8M(dl, ml, q0, q1) { \\
              for (int _m = 0; _m < vecs; ++_m) { \\
                  device const vec<IT, 4> *_xv = (device const vec<IT, 4> *)(xrow + _m * K); \\
                  const float4 _x0 = float4(_xv[0]); \\
                  const float4 _x1 = float4(_xv[1]); \\
                  const float4 _p = _x0 * (q0) + _x1 * (q1); \\
                  const float4 _t = _x0 + _x1; \\
                  acc[_m] += (dl) * (_p.x + _p.y + _p.z + _p.w) \\
                           - (ml) * (_t.x + _t.y + _t.z + _t.w); } }

      """

    /// The other half: the same eight weights written down instead, scale and minimum applied
    /// to each the way the reference decoder applies them, so the bytes stored are ggml's own.
    private static let storeEach = """
          #define GGML_ACC8(db, q0, q1) { \\
              yp[0] = vec<OT, 4>((db) * (q0)); \\
              yp[1] = vec<OT, 4>((db) * (q1)); }

          #define GGML_ACC8M(dl, ml, q0, q1) { \\
              yp[0] = vec<OT, 4>((dl) * (q0) - (ml)); \\
              yp[1] = vec<OT, 4>((dl) * (q1) - (ml)); }

      """

    /// The iq grids are a few kilobytes and every weight in the block reads one of them, so the
    /// threadgroup copies its own before any row starts. `qtype` is a compile-time constant, so
    /// only the grid this specialization actually uses is copied — or none at all.
    ///
    /// An eight-byte grid is copied as two four-byte planes rather than whole entries. Threadgroup
    /// memory banks by word, so entries laid end to end put every lookup on an even bank and half
    /// the banks go unused; split, the two halves of an entry are read at word `i` and `nent + i`
    /// and the thirty-two lanes of a simdgroup spread over all thirty-two banks.
    private static let blockTables = """
          threadgroup int8_t tg_grid[grid_bytes > 0 ? grid_bytes : 1];
          threadgroup uchar tg_signs[128];
          threadgroup float tg_kv[16];
          if (grid_bytes > 0) {
              device const int8_t *src = (qtype == 16) ? g_iq2xxs
                                       : (qtype == 17) ? g_iq2xs
                                       : (qtype == 22) ? g_iq2s
                                       : (qtype == 18) ? g_iq3xxs
                                       : (qtype == 21) ? g_iq3s
                                       : g_iq1s;
              if (qtype == 18 || qtype == 21) {
                  for (uint i = thread_position_in_threadgroup.x; i < (uint)grid_bytes; i += 256u) {
                      tg_grid[i] = src[i];
                  }
              } else {
                  device const uint *s32 = (device const uint *)src;
                  threadgroup uint *d32 = (threadgroup uint *)tg_grid;
                  const uint nent = (uint)grid_bytes / 8u;
                  for (uint i = thread_position_in_threadgroup.x; i < nent; i += 256u) {
                      d32[i] = s32[2 * i];
                      d32[nent + i] = s32[2 * i + 1];
                  }
              }
          }
          if (qtype == 16 || qtype == 17 || qtype == 18) {
              for (uint i = thread_position_in_threadgroup.x; i < 128u; i += 256u) {
                  tg_signs[i] = ksigns[i];
              }
          }
          if (qtype == 23 && thread_position_in_threadgroup.x < 16u) {
              tg_kv[thread_position_in_threadgroup.x] =
                  (float)kvalues[thread_position_in_threadgroup.x];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

      """

    /// What every format's body is written against.
    ///
    /// A lane holds eight weights, so a group is two vector loads and one scale. A sign bit is
    /// applied by flipping the float's own sign bit, which is what multiplying by ±1 does and
    /// costs no multiply.
    private static let blockMacros = """
          #define GGML_FLIP(g, sb, sh) as_type<float4>(as_type<uint4>(g) \\
              ^ (((uint4((sb)) >> uint4((sh), (sh) + 1, (sh) + 2, (sh) + 3)) & 1u) << 31))

          #define GGML_W8(sb, g0, g1) \\
              const float4 w0 = GGML_FLIP(g0, sb, 0); \\
              const float4 w1 = GGML_FLIP(g1, sb, 4);

          #define GGML_TG4(i) float4(*(threadgroup const char4 *)(tg_grid + 4 * (i)))
          #define GGML_G8(i) \\
              const float4 g0 = GGML_TG4(i); \\
              const float4 g1 = GGML_TG4((grid_bytes / 8) + (i));

          #define GGML_U8X2(p) uchar4(*(device const uchar2 *)(p), \\
                                      *(device const uchar2 *)((p) + 2))

      """

    /// A simdgroup takes one block at a time, and lane `L` takes the eight weights at offset
    /// `8L` within it.
    ///
    /// The alternative — a lane per block, blocks strided by the simd width — reads the
    /// activations at thirty-two places five hundred bytes apart, which is thirty-two memory
    /// transactions for what fits in four. Here the lanes of a simdgroup ask for `x` as one
    /// contiguous five-hundred-and-twelve-byte run, and for the block's own bytes the same way:
    /// the index and sign bytes a format keeps per group are read at `b + something + lane`.
    /// It also ends the lanes that sat idle when a row held fewer blocks than the simd width,
    /// since every lane now works on every block.
    private static let matvecProlog = """
          const uint gid = thread_position_in_grid.x;
          const uint row = gid / 32;
          const int lane = (int)(gid % 32);
          if (row >= (uint)N) { return; }

          const int nblk = K / block_elems;
          const ulong row_bytes = (ulong)nblk * block_bytes;

          float acc[vecs];
          for (int _m = 0; _m < vecs; ++_m) { acc[_m] = 0.0f; }

          for (int blk = 0; blk < nblk; ++blk) {
              device const uchar *b = w + (ulong)row * row_bytes + (ulong)blk * block_bytes;
              device const IT *xrow = x + blk * block_elems + 8 * lane;

      """

    /// Q2_K: sixteen weights to a scale and a minimum, so a lane is half a group.
    private static let blockQ2K = """
          {
              const int g = lane / 2;
              const int sub = lane % 2;
              const uchar sc = b[g];
              const float dl = GGML_HALF(b + 80) * (float)(sc & 0xF);
              const float ml = GGML_HALF(b + 82) * (float)(sc >> 4);
              const int shift = 2 * ((g % 8) / 2);
              device const uchar4 *q =
                  (device const uchar4 *)(b + (g < 8 ? 16 : 48) + 16 * (g % 2) + 8 * sub);
              const float4 q0 = float4((q[0] >> shift) & uchar4(3));
              const float4 q1 = float4((q[1] >> shift) & uchar4(3));
              GGML_ACC8M(dl, ml, q0, q1);
          }
      """

    /// Q4_K: thirty-two weights to a scale, the low nibbles of a run and then the high ones,
    /// with the scale and minimum packed six bits apiece.
    private static let blockQ4K = """
          {
              const int j = lane / 8;
              const int hi = (lane % 8) >= 4 ? 1 : 0;
              const int i = 2 * j + hi;
              device const uchar *sc = b + 4;
              uchar s, m;
              if (i < 4) { s = sc[i] & 63; m = sc[i + 4] & 63; }
              else { s = (sc[i + 4] & 0xF) | ((sc[i - 4] >> 6) << 4);
                     m = (sc[i + 4] >> 4) | ((sc[i] >> 6) << 4); }
              const float dl = GGML_HALF(b + 0) * (float)s;
              const float ml = GGML_HALF(b + 2) * (float)m;
              device const uchar4 *q = (device const uchar4 *)(b + 16 + 32 * j + 8 * (lane % 4));
              const float4 q0 = hi ? float4(q[0] >> 4) : float4(q[0] & uchar4(0xF));
              const float4 q1 = hi ? float4(q[1] >> 4) : float4(q[1] & uchar4(0xF));
              GGML_ACC8M(dl, ml, q0, q1);
          }
      """

    /// Q6_K: six bits split between a nibble and two high bits, sixteen weights to a signed
    /// scale. Its blocks are two hundred and ten bytes, so the quants are read two at a time —
    /// an odd block starts halfway through a word and a wider load would be misaligned.
    private static let blockQ6K = """
          {
              const int n = lane / 16;
              const int quarter = (lane % 16) / 4;
              const int l = 8 * (lane % 4);
              const float dl = GGML_HALF(b + 208)
                  * (float)((device const char *)b)[192 + 8 * n + 2 * quarter + l / 16];
              device const uchar *ql = b + 64 * n + 32 * (quarter % 2) + l;
              device const uchar *qh = b + 128 + 32 * n + l;
              const int hshift = 2 * quarter;
              const uchar4 l0 = GGML_U8X2(ql), l1 = GGML_U8X2(ql + 4);
              const uchar4 h0 = GGML_U8X2(qh), h1 = GGML_U8X2(qh + 4);
              const uchar4 nib0 = quarter < 2 ? (l0 & uchar4(0xF)) : (l0 >> 4);
              const uchar4 nib1 = quarter < 2 ? (l1 & uchar4(0xF)) : (l1 >> 4);
              const float4 q0 = float4(nib0 | (((h0 >> hshift) & uchar4(3)) << 4)) - 32.0f;
              const float4 q1 = float4(nib1 | (((h1 >> hshift) & uchar4(3)) << 4)) - 32.0f;
              GGML_ACC8(dl, q0, q1);
          }
      """

    /// IQ4_XS: a nibble into the sixteen-entry non-linear codebook, which is small enough to
    /// sit in threadgroup memory as floats.
    private static let blockIQ4XS = """
          {
              const int ib = lane / 4;
              const int lo = (b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF;
              const int hi = ((*(device const ushort *)(b + 2) >> (2 * ib)) & 3) << 4;
              const float dl = GGML_HALF(b + 0) * (float)((lo | hi) - 32);
              device const uchar4 *q = (device const uchar4 *)(b + 8 + 16 * ib + 8 * (lane % 2));
              const bool upper = (lane % 4) >= 2;
              const uchar4 v0 = upper ? (q[0] >> 4) : (q[0] & uchar4(0xF));
              const uchar4 v1 = upper ? (q[1] >> 4) : (q[1] & uchar4(0xF));
              const float4 q0 = float4(tg_kv[v0.x], tg_kv[v0.y], tg_kv[v0.z], tg_kv[v0.w]);
              const float4 q1 = float4(tg_kv[v1.x], tg_kv[v1.y], tg_kv[v1.z], tg_kv[v1.w]);
              GGML_ACC8(dl, q0, q1);
          }
      """

    /// IQ2_XXS: a lane's eight weights are exactly one grid entry, its signs and its scale,
    /// all four packed into the pair of words the group shares.
    private static let blockIQ2XXS = """
          {
              const int ib = lane / 4, l = lane % 4;
              device const ushort *p = (device const ushort *)(b + 2 + 8 * ib);
              const uint a1 = (uint)p[0] | ((uint)p[1] << 16);
              const uint a2 = (uint)p[2] | ((uint)p[3] << 16);
              const float db = GGML_HALF(b + 0) * (0.5f + (float)(a2 >> 28)) * 0.25f;
              GGML_G8((int)((a1 >> (8 * l)) & 0xFF));
              const uchar sb = tg_signs[(a2 >> (7 * l)) & 127];
              GGML_W8(sb, g0, g1);
              GGML_ACC8(db, w0, w1);
          }
      """

    /// IQ2_XS: nine bits of grid index and seven of sign in one short, read at `b + 2 + 2 *
    /// lane` so the simdgroup takes the block's index table as one run.
    private static let blockIQ2XS = """
          {
              const int ib = lane / 4, l = lane % 4;
              const uchar sc = b[66 + ib];
              const float db = GGML_HALF(b + 0) * 0.25f
                  * (0.5f + (float)(l < 2 ? (sc & 0xF) : (sc >> 4)));
              const ushort q = *(device const ushort *)(b + 2 + 2 * lane);
              GGML_G8((int)(q & 511));
              const uchar sb = tg_signs[q >> 9];
              GGML_W8(sb, g0, g1);
              GGML_ACC8(db, w0, w1);
          }
      """

    /// IQ2_S: the grid index's tenth bit lives in a shared high byte, and the signs are the
    /// block's own rather than the shared table's.
    private static let blockIQ2S = """
          {
              const int ib = lane / 4, l = lane % 4;
              const uchar sc = b[74 + ib];
              const float db = GGML_HALF(b + 0) * 0.25f
                  * (0.5f + (float)(l < 2 ? (sc & 0xF) : (sc >> 4)));
              const int idx = (int)b[2 + lane] | (((int)b[66 + ib] << (8 - 2 * l)) & 0x300);
              GGML_G8(idx);
              const uchar sb = b[34 + lane];
              GGML_W8(sb, g0, g1);
              GGML_ACC8(db, w0, w1);
          }
      """

    /// IQ3_XXS: four weights to a grid entry, so a lane reads two of them — adjacent bytes,
    /// which is one short.
    private static let blockIQ3XXS = """
          {
              const int ib = lane / 4, l = lane % 4;
              const uint a = GGML_U32(b + 66 + 4 * ib);
              const float db = GGML_HALF(b + 0) * (0.5f + (float)(a >> 28)) * 0.5f;
              const uchar2 e = *(device const uchar2 *)(b + 2 + 2 * lane);
              const uchar sb = tg_signs[(a >> (7 * l)) & 127];
              GGML_W8(sb, GGML_TG4((int)e.x), GGML_TG4((int)e.y));
              GGML_ACC8(db, w0, w1);
          }
      """

    /// IQ3_S: the same pair of entries, with the ninth index bit out of a shared byte and the
    /// signs stored per group.
    private static let blockIQ3S = """
          {
              const int ib = lane / 4, l = lane % 4;
              const uchar sc = b[106 + ib / 2];
              const float db = GGML_HALF(b + 0)
                  * (float)(1 + 2 * (ib % 2 == 0 ? (sc & 0xF) : (sc >> 4)));
              const int qh = b[66 + ib];
              const uchar2 e = *(device const uchar2 *)(b + 2 + 2 * lane);
              const uchar sb = b[74 + lane];
              GGML_W8(sb,
                  GGML_TG4((int)e.x | ((qh << (8 - 2 * l)) & 256)),
                  GGML_TG4((int)e.y | ((qh << (7 - 2 * l)) & 256)));
              GGML_ACC8(db, w0, w1);
          }
      """

    /// IQ1_S: no sign bits at all. The grid is ternary and the block shifts it by a delta,
    /// which comes out of the sum the way a minimum does.
    private static let blockIQ1S = """
          {
              const int ib = lane / 4, l = lane % 4;
              const ushort qh = *(device const ushort *)(b + 34 + 2 * ib);
              const float dl = GGML_HALF(b + 0) * (float)(2 * ((qh >> 12) & 7) + 1);
              const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
              GGML_G8((int)b[2 + lane] | (((qh >> (3 * l)) & 7) << 8));
              const float4 w0 = g0 + delta;
              const float4 w1 = g1 + delta;
              GGML_ACC8(dl, w0, w1);
          }
      """

    /// IQ1_M: the same grid, with a delta and a scale per group of eight rather than per
    /// thirty-two, and the block scale reassembled from four nibbles.
    private static let blockIQ1M = """
          {
              const int ib = lane / 4, l = lane % 4;
              ushort sc[4];
              for (int i = 0; i < 4; ++i) { sc[i] = *(device const ushort *)(b + 48 + 2 * i); }
              const ushort packed = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
                                  | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
              const int shift = 6 * (ib % 2) + (l < 2 ? 0 : 3);
              const float dl = (float)as_type<half>(packed)
                  * (float)(2 * ((sc[ib / 2] >> shift) & 0x7) + 1);
              const int qh = b[32 + 2 * ib + (l < 2 ? 0 : 1)];
              const int idx = (int)b[lane] | ((qh << (l % 2 == 0 ? 8 : 4)) & 0x700);
              const float delta = (qh & (l % 2 == 0 ? 0x08 : 0x80)) ? -0.125f : 0.125f;
              GGML_G8(idx);
              const float4 w0 = g0 + delta;
              const float4 w1 = g1 + delta;
              GGML_ACC8(dl, w0, w1);
          }
      """

    /// One body per type, picked at compile time: `qtype` is a specialization constant, so a
    /// built kernel holds exactly one of these.
    private static let blockChain: String = {
      let bodies: [(GGMLType, String)] = [
        (.q2_K, blockQ2K), (.q4_K, blockQ4K), (.q6_K, blockQ6K), (.iq4_xs, blockIQ4XS),
        (.iq2_xxs, blockIQ2XXS), (.iq2_xs, blockIQ2XS), (.iq2_s, blockIQ2S),
        (.iq3_xxs, blockIQ3XXS), (.iq3_s, blockIQ3S), (.iq1_s, blockIQ1S),
        (.iq1_m, blockIQ1M),
      ]
      return bodies.enumerated().map { i, entry in
        (i == 0 ? "if" : "} else if") + " (qtype == \(entry.0.rawValue)) {" + entry.1
      }.joined() + "}"
    }()

    private static let matvecEpilog = """

          }

          for (int _m = 0; _m < vecs; ++_m) {
              const float total = simd_sum(acc[_m]);
              if (lane == 0) { y[_m * N + row] = (IT)total; }
          }
      """

  #endif
}
