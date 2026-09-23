// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A few-row affine matmul for the speculative verify: each weight is decoded once, straight into
// a simdgroup matrix fragment, and multiplied against up to eight rows at the cost of about one.

import Foundation
import MLX
import MLXFast

public enum VerifyMatmul {
  public static let supportedRows = 5...8

  static let rowBlocks = 4
  static let simdgroups = 2
  static let unroll = 2

  public static func apply(
    _ x: MLXArray, _ w: MLXArray, scales: MLXArray, biases: MLXArray,
    groupSize: Int, bits: Int
  ) -> MLXArray? {
    #if canImport(Metal)
      guard x.ndim == 2 else { return nil }
      let m = x.dim(0)
      let k = x.dim(1)
      let n = w.dim(0)
      let rowsPerGroup = 8 * rowBlocks * simdgroups
      guard supportedRows.contains(m), (2...8).contains(bits) else { return nil }
      guard k % (64 * unroll) == 0, groupSize % (16 * unroll) == 0, k % groupSize == 0 else {
        return nil
      }
      guard n % rowsPerGroup == 0, w.dim(1) * 32 == k * bits else { return nil }

      return kernel(
        [x, w, scales, biases],
        template: [
          ("M", m), ("K", k), ("N", n), ("BITS", bits), ("GS", groupSize),
          ("RB", rowBlocks), ("SGS", simdgroups), ("U", unroll),
        ],
        grid: ((n / rowsPerGroup) * 32 * simdgroups, 1, 1),
        threadGroup: (32 * simdgroups, 1, 1),
        outputShapes: [[m, n]],
        outputDTypes: [x.dtype])[0]
    #else
      return nil
    #endif
  }

  #if canImport(Metal)
    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
      name: "ishizuki_verify_qmm",
      inputNames: ["x", "w", "scales", "biases"],
      outputNames: ["y"],
      source: """
            constexpr uint CH = 16 * U;
            constexpr uint TK = 64 * U;
            constexpr uint row_bytes = K * BITS / 8;
            constexpr uint groups_per_row = K / GS;
            constexpr uint mask = (1u << BITS) - 1u;
            constexpr uint NW = (CH * BITS + 31) / 32;

            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint qid = lane / 4;
            uint fm = (qid & 4) + ((lane / 2) % 4);
            uint fn = (qid & 2) * 2 + (lane % 2) * 2;
            uint row_base = (threadgroup_position_in_grid.x * SGS + sg) * 8 * RB;

            simdgroup_float8x8 acc[RB];
            UNROLL for (uint b = 0; b < RB; ++b)
                acc[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

            device const uchar *wb = (device const uchar *)w;
            uint kl = (fn / 2) * CH;
            uint kb = (fm / 2) * CH + (fm % 2);
            auto x0 = x + fn * K;
            auto x1 = x + (fn + 1) * K;
            bool m0 = fn < M;
            bool m1 = fn + 1 < M;

            for (uint k0 = 0; k0 < K; k0 += TK) {
                uint words[RB][NW + 1];
                float sc[RB], bi[RB];
                UNROLL for (uint b = 0; b < RB; ++b) {
                    uint row = row_base + b * 8 + fm;
                    uint k = k0 + kl;
                    device const uint *p =
                        (device const uint *)(wb + row * row_bytes + (k * BITS) / 8);
                    UNROLL for (uint j = 0; j < NW; ++j) words[b][j] = p[j];
                    words[b][NW] = 0;
                    uint g = row * groups_per_row + k / GS;
                    sc[b] = float(scales[g]);
                    bi[b] = float(biases[g]);
                }
                UNROLL for (uint j = 0; j < 8 * U; ++j) {
                    simdgroup_float8x8 bx;
                    uint k = k0 + kb + 2 * j;
                    bx.thread_elements()[0] = m0 ? float(x0[k]) : 0.0f;
                    bx.thread_elements()[1] = m1 ? float(x1[k]) : 0.0f;
                    UNROLL for (uint b = 0; b < RB; ++b) {
                        simdgroup_float8x8 a;
                        uint p0 = (2 * j) * BITS;
                        uint p1 = p0 + BITS;
                        uint v0 = words[b][p0 >> 5] >> (p0 & 31u);
                        if ((p0 & 31u) + BITS > 32) v0 |= words[b][(p0 >> 5) + 1] << (32 - (p0 & 31u));
                        uint v1 = words[b][p1 >> 5] >> (p1 & 31u);
                        if ((p1 & 31u) + BITS > 32) v1 |= words[b][(p1 >> 5) + 1] << (32 - (p1 & 31u));
                        a.thread_elements()[0] = float(v0 & mask) * sc[b] + bi[b];
                        a.thread_elements()[1] = float(v1 & mask) * sc[b] + bi[b];
                        simdgroup_multiply_accumulate(acc[b], a, bx, acc[b]);
                    }
                }
            }

            UNROLL for (uint b = 0; b < RB; ++b) {
                uint row = row_base + b * 8 + fm;
                if (m0) y[fn * N + row] = acc[b].thread_elements()[0];
                if (m1) y[(fn + 1) * N + row] = acc[b].thread_elements()[1];
            }
        """,
      header: """
        #include <metal_simdgroup_matrix>
        #define UNROLL _Pragma("clang loop unroll(full)")

        """)
  #endif
}
