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

/// Expands a tensor's GGML blocks on the GPU.
///
/// A block is self-contained — its scales live in its own bytes — so one thread owns one
/// super-block and no thread reads another's state. That makes the dequantizer a plain map,
/// and it is the same decode the fused matvec will inline rather than a second description
/// of the format.
public enum GGMLKernels {
  /// Blocks as raw bytes to a dense array of `shape`. Nil when Metal is unavailable or the
  /// type has no decoder, so callers fall back to the CPU reference.
  public static func dequantize(
    blocks: MLXArray, type: GGMLType, shape: [Int], dtype: DType = .bfloat16
  ) -> MLXArray? {
    #if canImport(Metal)
      guard let kernel, GGMLDequant.supported.contains(type), type.isQuantized else {
        return nil
      }
      let count = shape.reduce(1, *)
      guard count % type.blockSize == 0 else { return nil }
      let blockCount = count / type.blockSize
      guard blocks.size >= blockCount * type.typeSize else { return nil }

      let threads = 256
      let groups = (blockCount + threads - 1) / threads
      let outputs = kernel(
        [blocks] + GGMLGrids.buffers + [blockCount],
        template: [
          ("OT", dtype),
          ("qtype", Int(type.rawValue)),
          ("block_bytes", type.typeSize),
          ("block_elems", type.blockSize),
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

  #if canImport(Metal)
    private static let kernel: MLXFast.MLXFastKernel? = {
      MLXFast.metalKernel(
        name: "ggml_dequantize",
        inputNames: [
          "blocks", "g_iq2xxs", "g_iq2xs", "g_iq2s", "g_iq1s", "g_iq3xxs", "g_iq3s",
          "ksigns", "kvalues", "n_blocks",
        ],
        outputNames: ["y"],
        source: source)
    }()

    private static let source = """
          #define GGML_HALF(p) ((float)as_type<half>(*(device const ushort *)(p)))
          #define GGML_SIGN(s, j) (((s) & (1 << (j))) ? -1.0f : 1.0f)
          #define GGML_U32(p) ((uint)(p)[0] | ((uint)(p)[1] << 8) \
                              | ((uint)(p)[2] << 16) | ((uint)(p)[3] << 24))

          uint bi = thread_position_in_grid.x;
          if (bi >= (uint)n_blocks) { return; }

          device const uchar *b = blocks + (ulong)bi * block_bytes;
          const int base = bi * block_elems;
          int o = 0;

          if (qtype == 10) {
              const float d = GGML_HALF(b + 80);
              const float dmin = GGML_HALF(b + 82);
              int sidx = 0;
              for (int n = 0; n < 256; n += 128) {
                  const int q0 = 16 + n / 4;
                  int shift = 0;
                  for (int j = 0; j < 4; ++j) {
                      for (int lane = 0; lane < 2; ++lane) {
                          const uchar sc = b[sidx++];
                          const float dl = d * (float)(sc & 0xF);
                          const float ml = dmin * (float)(sc >> 4);
                          for (int l = 0; l < 16; ++l) {
                              const uchar q = b[q0 + 16 * lane + l];
                              y[base + o++] = (OT)(dl * (float)((q >> shift) & 3) - ml);
                          }
                      }
                      shift += 2;
                  }
              }
          } else if (qtype == 12) {
              const float d = GGML_HALF(b + 0);
              const float dmin = GGML_HALF(b + 2);
              device const uchar *sc = b + 4;
              for (int j = 0; j < 4; ++j) {
                  uchar s1, m1, s2, m2;
                  const int i1 = 2 * j, i2 = 2 * j + 1;
                  if (i1 < 4) { s1 = sc[i1] & 63; m1 = sc[i1 + 4] & 63; }
                  else { s1 = (sc[i1 + 4] & 0xF) | ((sc[i1 - 4] >> 6) << 4);
                         m1 = (sc[i1 + 4] >> 4) | ((sc[i1] >> 6) << 4); }
                  if (i2 < 4) { s2 = sc[i2] & 63; m2 = sc[i2 + 4] & 63; }
                  else { s2 = (sc[i2 + 4] & 0xF) | ((sc[i2 - 4] >> 6) << 4);
                         m2 = (sc[i2 + 4] >> 4) | ((sc[i2] >> 6) << 4); }
                  const float d1 = d * (float)s1, mm1 = dmin * (float)m1;
                  const float d2 = d * (float)s2, mm2 = dmin * (float)m2;
                  device const uchar *q = b + 16 + 32 * j;
                  for (int l = 0; l < 32; ++l) {
                      y[base + o + l]      = (OT)(d1 * (float)(q[l] & 0xF) - mm1);
                      y[base + o + 32 + l] = (OT)(d2 * (float)(q[l] >> 4) - mm2);
                  }
                  o += 64;
              }
          } else if (qtype == 23) {
              const float d = GGML_HALF(b + 0);
              const ushort sh = *(device const ushort *)(b + 2);
              for (int ib = 0; ib < 8; ++ib) {
                  const int lo = (b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF;
                  const int hi = ((sh >> (2 * ib)) & 3) << 4;
                  const float dl = d * (float)((lo | hi) - 32);
                  device const uchar *q = b + 8 + 16 * ib;
                  for (int j = 0; j < 16; ++j) {
                      y[base + o + j]      = (OT)(dl * (float)kvalues[q[j] & 0xF]);
                      y[base + o + 16 + j] = (OT)(dl * (float)kvalues[q[j] >> 4]);
                  }
                  o += 32;
              }
          } else if (qtype == 16) {
              const float d = GGML_HALF(b + 0);
              for (int ib = 0; ib < 8; ++ib) {
                  const uint a1 = GGML_U32(b + 2 + 8 * ib);
                  const uint a2 = GGML_U32(b + 6 + 8 * ib);
                  const float db = d * (0.5f + (float)(a2 >> 28)) * 0.25f;
                  for (int l = 0; l < 4; ++l) {
                      device const int8_t *g = g_iq2xxs + 8 * (int)((a1 >> (8 * l)) & 0xFF);
                      const uchar s = ksigns[(a2 >> (7 * l)) & 127];
                      for (int j = 0; j < 8; ++j) {
                          y[base + o + j] = (OT)(db * (float)g[j] * GGML_SIGN(s, j));
                      }
                      o += 8;
                  }
              }
          } else if (qtype == 17) {
              const float d = GGML_HALF(b + 0);
              for (int ib = 0; ib < 8; ++ib) {
                  const uchar sc = b[66 + ib];
                  const float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
                  const float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
                  for (int l = 0; l < 4; ++l) {
                      const ushort q = *(device const ushort *)(b + 2 + 2 * (4 * ib + l));
                      device const int8_t *g = g_iq2xs + 8 * (int)(q & 511);
                      const uchar s = ksigns[q >> 9];
                      const float db = (l < 2) ? db0 : db1;
                      for (int j = 0; j < 8; ++j) {
                          y[base + o + j] = (OT)(db * (float)g[j] * GGML_SIGN(s, j));
                      }
                      o += 8;
                  }
              }
          } else if (qtype == 22) {
              const float d = GGML_HALF(b + 0);
              for (int ib = 0; ib < 8; ++ib) {
                  const uchar sc = b[74 + ib];
                  const float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
                  const float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
                  const int qh = b[66 + ib];
                  for (int l = 0; l < 4; ++l) {
                      const int idx = (int)b[2 + 4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300);
                      device const int8_t *g = g_iq2s + 8 * idx;
                      const uchar s = b[34 + 4 * ib + l];
                      const float db = (l < 2) ? db0 : db1;
                      for (int j = 0; j < 8; ++j) {
                          y[base + o + j] = (OT)(db * (float)g[j] * GGML_SIGN(s, j));
                      }
                      o += 8;
                  }
              }
          } else if (qtype == 18) {
              const float d = GGML_HALF(b + 0);
              for (int ib = 0; ib < 8; ++ib) {
                  const uint a = GGML_U32(b + 66 + 4 * ib);
                  const float db = d * (0.5f + (float)(a >> 28)) * 0.5f;
                  for (int l = 0; l < 4; ++l) {
                      const uchar s = ksigns[(a >> (7 * l)) & 127];
                      device const int8_t *g1 = g_iq3xxs + 4 * (int)b[2 + 8 * ib + 2 * l];
                      device const int8_t *g2 = g_iq3xxs + 4 * (int)b[2 + 8 * ib + 2 * l + 1];
                      for (int j = 0; j < 4; ++j) {
                          y[base + o + j]     = (OT)(db * (float)g1[j] * GGML_SIGN(s, j));
                          y[base + o + 4 + j] = (OT)(db * (float)g2[j] * GGML_SIGN(s, j + 4));
                      }
                      o += 8;
                  }
              }
          } else if (qtype == 21) {
              const float d = GGML_HALF(b + 0);
              for (int pair = 0; pair < 8; pair += 2) {
                  const uchar sc = b[106 + pair / 2];
                  const float db0 = d * (float)(1 + 2 * (sc & 0xF));
                  const float db1 = d * (float)(1 + 2 * (sc >> 4));
                  for (int grp = 0; grp < 2; ++grp) {
                      const int qh = b[66 + pair + grp];
                      device const uchar *q = b + 2 + 8 * (pair + grp);
                      device const uchar *sg = b + 74 + 4 * (pair + grp);
                      const float db = (grp == 0) ? db0 : db1;
                      for (int l = 0; l < 4; ++l) {
                          device const int8_t *g1 = g_iq3s + 4 * ((int)q[2 * l] | ((qh << (8 - 2 * l)) & 256));
                          device const int8_t *g2 = g_iq3s + 4 * ((int)q[2 * l + 1] | ((qh << (7 - 2 * l)) & 256));
                          const uchar s = sg[l];
                          for (int j = 0; j < 4; ++j) {
                              y[base + o + j]     = (OT)(db * (float)g1[j] * GGML_SIGN(s, j));
                              y[base + o + 4 + j] = (OT)(db * (float)g2[j] * GGML_SIGN(s, j + 4));
                          }
                          o += 8;
                      }
                  }
              }
          } else if (qtype == 19) {
              const float d = GGML_HALF(b + 0);
              for (int ib = 0; ib < 8; ++ib) {
                  const ushort qh = *(device const ushort *)(b + 34 + 2 * ib);
                  const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
                  const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
                  for (int l = 0; l < 4; ++l) {
                      const int idx = (int)b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
                      device const int8_t *g = g_iq1s + 8 * idx;
                      for (int j = 0; j < 8; ++j) {
                          y[base + o + j] = (OT)(dl * ((float)g[j] + delta));
                      }
                      o += 8;
                  }
              }
          } else if (qtype == 29) {
              ushort sc[4];
              for (int i = 0; i < 4; ++i) { sc[i] = *(device const ushort *)(b + 48 + 2 * i); }
              const ushort packed = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0)
                                  | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
              const float d = (float)as_type<half>(packed);
              for (int ib = 0; ib < 8; ++ib) {
                  const ushort w = sc[ib / 2];
                  const int shift = 6 * (ib % 2);
                  const float dl1 = d * (float)(2 * ((w >> shift) & 0x7) + 1);
                  const float dl2 = d * (float)(2 * ((w >> (shift + 3)) & 0x7) + 1);
                  const int qh0 = b[32 + 2 * ib];
                  const int qh1 = b[32 + 2 * ib + 1];
                  int idx[4];
                  float dt[4];
                  idx[0] = (int)b[4 * ib + 0] | ((qh0 << 8) & 0x700);
                  idx[1] = (int)b[4 * ib + 1] | ((qh0 << 4) & 0x700);
                  idx[2] = (int)b[4 * ib + 2] | ((qh1 << 8) & 0x700);
                  idx[3] = (int)b[4 * ib + 3] | ((qh1 << 4) & 0x700);
                  dt[0] = (qh0 & 0x08) ? -0.125f : 0.125f;
                  dt[1] = (qh0 & 0x80) ? -0.125f : 0.125f;
                  dt[2] = (qh1 & 0x08) ? -0.125f : 0.125f;
                  dt[3] = (qh1 & 0x80) ? -0.125f : 0.125f;
                  for (int l = 0; l < 4; ++l) {
                      device const int8_t *g = g_iq1s + 8 * idx[l];
                      const float dl = (l < 2) ? dl1 : dl2;
                      for (int j = 0; j < 8; ++j) {
                          y[base + o + j] = (OT)(dl * ((float)g[j] + dt[l]));
                      }
                      o += 8;
                  }
              }
          }
      """
  #endif
}
