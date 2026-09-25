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

  nonisolated(unsafe) static var rowBlocks = 4
  nonisolated(unsafe) static var simdgroups = 2
  static let unroll = 2
  /// Simdgroups a product should keep in flight. A narrow output has too few blocks of rows
  /// to go round the GPU's cores, so its reduction is split and the slices added after.
  nonisolated(unsafe) static var occupancy = 2048

  private static let warmedLock = NSLock()
  nonisolated(unsafe) private static var warmedShapes = Set<[Int]>()

  /// Compiles a weight shape's kernel for each activation dtype at load, so the first drafted
  /// round does not wait on a Metal library build.
  public static func warm(
    _ w: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int
  ) {
    #if canImport(Metal)
      guard BonsaiRuntime.useVerifyMatmul else { return }
      let k = scales.dim(1) * groupSize
      let key = [k, w.dim(0), w.dim(1), bits, groupSize, scales.dtype.size]
      warmedLock.lock()
      let fresh = warmedShapes.insert(key).inserted
      warmedLock.unlock()
      guard fresh else { return }
      let weights = MLXArray.zeros(w.shape, dtype: w.dtype)
      let groups = MLXArray.zeros(scales.shape, dtype: scales.dtype)
      let offsets = MLXArray.zeros(biases.shape, dtype: biases.dtype)
      let outputs = [DType.float16, .float32, .bfloat16].compactMap {
        apply(
          MLXArray.zeros([supportedRows.upperBound, k], dtype: $0), weights, scales: groups,
          biases: offsets, groupSize: groupSize, bits: bits)
      }
      eval(outputs)
    #endif
  }

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

      if [2, 4, 8].contains(bits), groupSize % (64 * unroll) == 0 {
        let groups = k / groupSize
        let blocks = (n / rowsPerGroup) * simdgroups
        let split =
          (1...groups).first { groups % $0 == 0 && blocks * $0 >= occupancy } ?? groups
        let y = codes(
          [x, w, scales, biases, m],
          template: [
            ("K", k), ("N", n), ("BITS", bits), ("GS", groupSize),
            ("RB", rowBlocks), ("SGS", simdgroups), ("U", unroll), ("SPLIT", split),
          ],
          grid: ((n / rowsPerGroup) * 32 * simdgroups, split, 1),
          threadGroup: (32 * simdgroups, 1, 1),
          outputShapes: [split > 1 ? [split, m, n] : [m, n]],
          outputDTypes: [split > 1 ? .float32 : x.dtype])[0]
        return split > 1 ? y.sum(axis: 0).asType(x.dtype) : y
      }

      return kernel(
        [x, w, scales, biases, m],
        template: [
          ("K", k), ("N", n), ("BITS", bits), ("GS", groupSize),
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
    /// The same product with the weights left as their codes. Two codes come out of a word
    /// with one mask, sixteen bits apart, and become exact halves by setting the exponent of
    /// 1024 above them and taking 1024 away; the fragments multiply codes by activations, and
    /// each group's scale and bias are applied once, to its partial sums, rather than to every
    /// weight. The bias needs each group's activation sum, which the lanes that loaded those
    /// activations add up among themselves. Only widths whose codes pack a word evenly take it.
    private static let codes: MLXFast.MLXFastKernel = MLXFast.metalKernel(
      name: "ishizuki_verify_qmm_codes",
      inputNames: ["x", "w", "scales", "biases", "M"],
      outputNames: ["y"],
      source: """
            constexpr uint CH = 16 * U;
            constexpr uint TK = 64 * U;
            constexpr uint row_bytes = K * BITS / 8;
            constexpr uint G = K / GS;
            constexpr uint C = 32 / BITS;
            constexpr uint D = 16 / BITS;
            constexpr uint RW = CH / C;
            constexpr uint pair = ((1u << BITS) - 1u) * 0x00010001u;

            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            uint qid = lane / 4;
            uint fm = (qid & 4) + ((lane / 2) % 4);
            uint fn = (qid & 2) * 2 + (lane % 2) * 2;
            uint row_base = (threadgroup_position_in_grid.x * SGS + sg) * 8 * RB;

            simdgroup_float8x8 acc[RB];
            simdgroup_float8x8 part[RB];
            UNROLL for (uint b = 0; b < RB; ++b) {
                acc[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                part[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }

            device const uchar *wb = (device const uchar *)w;
            uint kl = (fn / 2) * CH;
            uint kb = (fm / 2) * CH + (fm % 2) * D;
            bool m0 = fn < uint(M);
            bool m1 = fn + 1 < uint(M);
            auto x0 = x + min(fn, uint(M) - 1) * K;
            auto x1 = x + min(fn + 1, uint(M) - 1) * K;
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            constexpr uint KS = K / SPLIT;
            uint split = threadgroup_position_in_grid.y;

            for (uint k0 = split * KS; k0 < (split + 1) * KS; k0 += TK) {
                uint words[RB][RW];
                UNROLL for (uint b = 0; b < RB; ++b) {
                    uint row = row_base + b * 8 + fm;
                    device const uint *p =
                        (device const uint *)(wb + row * row_bytes + ((k0 + kl) * BITS) / 8);
                    UNROLL for (uint j = 0; j < RW; ++j) words[b][j] = p[j];
                }
                UNROLL for (uint j = 0; j < CH / 2; ++j) {
                    uint k = k0 + kb + (j % D) + C * (j / D);
                    float v0 = float(x0[k]);
                    float v1 = float(x1[k]);
                    sum0 += v0;
                    sum1 += v1;
                    simdgroup_float8x8 bx;
                    bx.thread_elements()[0] = v0;
                    bx.thread_elements()[1] = v1;
                    UNROLL for (uint b = 0; b < RB; ++b) {
                        uint bitsv = (words[b][j / D] >> (BITS * (j % D))) & pair;
                        half2 q = as_type<half2>(bitsv | 0x64006400u) - half2(1024.0h);
                        simdgroup_half8x8 a;
                        a.thread_elements()[0] = q.x;
                        a.thread_elements()[1] = q.y;
                        simdgroup_multiply_accumulate(part[b], a, bx, part[b]);
                    }
                }
                if ((k0 + TK) % GS == 0) {
                    uint g = (k0 + TK) / GS - 1;
                    float s0 = sum0 + simd_shuffle_xor(sum0, 2);
                    s0 += simd_shuffle_xor(s0, 4);
                    s0 += simd_shuffle_xor(s0, 16);
                    float s1 = sum1 + simd_shuffle_xor(sum1, 2);
                    s1 += simd_shuffle_xor(s1, 4);
                    s1 += simd_shuffle_xor(s1, 16);
                    sum0 = 0.0f;
                    sum1 = 0.0f;
                    UNROLL for (uint b = 0; b < RB; ++b) {
                        uint gi = (row_base + b * 8 + fm) * G + g;
                        float sc = float(scales[gi]);
                        float bi = float(biases[gi]);
                        acc[b].thread_elements()[0] += sc * part[b].thread_elements()[0] + bi * s0;
                        acc[b].thread_elements()[1] += sc * part[b].thread_elements()[1] + bi * s1;
                        part[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    }
                }
            }

            UNROLL for (uint b = 0; b < RB; ++b) {
                uint row = row_base + b * 8 + fm;
                if (m0) y[(split * uint(M) + fn) * N + row] = acc[b].thread_elements()[0];
                if (m1) y[(split * uint(M) + fn + 1) * N + row] = acc[b].thread_elements()[1];
            }
        """,
      header: """
        #include <metal_simdgroup_matrix>
        #define UNROLL _Pragma("clang loop unroll(full)")

        """)

    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
      name: "ishizuki_verify_qmm",
      inputNames: ["x", "w", "scales", "biases", "M"],
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
            bool m0 = fn < uint(M);
            bool m1 = fn + 1 < uint(M);
            auto x0 = x + min(fn, uint(M) - 1) * K;
            auto x1 = x + min(fn + 1, uint(M) - 1) * K;

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
                    bx.thread_elements()[0] = float(x0[k]);
                    bx.thread_elements()[1] = float(x1[k]);
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
