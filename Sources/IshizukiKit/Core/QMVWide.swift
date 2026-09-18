// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public enum QMVWide {
  public static let supportedBatch = 2...5

  static let kLanes = 8
  static let simdgroupsPerThreadgroup = 4
  static let threadsPerThreadgroup = 32 * simdgroupsPerThreadgroup
  static let rowsPerThreadgroup = (32 / kLanes) * simdgroupsPerThreadgroup

  public static func apply(
    _ x: MLXArray, _ w: MLXArray, scales: MLXArray, biases: MLXArray,
    groupSize: Int, bits: Int
  ) -> MLXArray? {
    #if canImport(Metal)
      guard let kernel else { return nil }
      guard x.ndim == 2 else { return nil }

      let m = x.dim(0)
      let k = x.dim(1)
      let n = w.dim(0)
      guard supportedBatch.contains(m) else { return nil }
      guard k % groupSize == 0, groupSize % 8 == 0, bits == 2 else { return nil }
      guard k / groupSize > 0 else { return nil }

      let vectorBlocks = (m + m - 1) / m
      let rowBlocks = (n + rowsPerThreadgroup - 1) / rowsPerThreadgroup

      let outputs = kernel(
        [x, w, scales, biases, k, n, m],
        template: [
          ("IT", x.dtype),
          ("group_size", groupSize),
          ("bits", bits),
          ("vecs_per_tg", m),
          ("k_lanes", kLanes),
        ],
        grid: (vectorBlocks * threadsPerThreadgroup, rowBlocks, 1),
        threadGroup: (threadsPerThreadgroup, 1, 1),
        outputShapes: [[m, n]],
        outputDTypes: [x.dtype])
      return outputs[0]
    #else
      return nil
    #endif
  }

  #if canImport(Metal)
    private static let kernel: MLXFast.MLXFastKernel? = {
      let source = """
            constexpr int num_simdgroups = 4;
            constexpr int results_per_simdgroup = 32 / k_lanes;
            constexpr int sub = 8;

            const short k_lane = thread_index_in_simdgroup % k_lanes;
            const short sg_row = thread_index_in_simdgroup / k_lanes;

            const int out_row =
                threadgroup_position_in_grid.y * (results_per_simdgroup * num_simdgroups) +
                results_per_simdgroup * simdgroup_index_in_threadgroup + sg_row;
            const int vec0 = threadgroup_position_in_grid.x * vecs_per_tg;

            // Clamp rather than branch: out-of-range rows compute a harmless duplicate
            // and are dropped at the store.
            const int row = min(out_row, N - 1);

            const int row_bytes = K * bits / 8;
            const int groups = K / group_size;

            const device uint8_t* wrow = (const device uint8_t*)w + row * row_bytes;
            const device IT* srow = scales + row * groups;
            const device IT* brow = biases + row * groups;

            const device IT* xv[vecs_per_tg];
            for (int v = 0; v < vecs_per_tg; v++) {
                xv[v] = x + min(vec0 + v, M - 1) * K;
            }

            float result[vecs_per_tg];
            for (int v = 0; v < vecs_per_tg; v++) { result[v] = 0.0f; }

            // Each lane walks a strided subset of this row's groups.
            for (int g = k_lane; g < groups; g += k_lanes) {
                float scale = static_cast<float>(srow[g]);
                float bias = static_cast<float>(brow[g]);
                float spb  = scale + bias;
                float lut2 = fma(2.0f, scale, bias);
                float lut3 = fma(3.0f, scale, bias);

                for (int sc = 0; sc < group_size / sub; sc++) {
                    const int k0 = g * group_size + sc * sub;
                    const device uint8_t* wc = wrow + k0 * bits / 8;

                    uint8_t wb0 = wc[0];
                    uint8_t wb1 = wc[1];
                    float w_dq[sub];

                    // Two selects per value: bit0 picks within a pair, bit1 picks the
                    // pair. Metal lowers select() to one instruction, and avoiding a
                    // 4-entry lookup array keeps this out of spillable memory.
                    uint8_t q0 = wb0 & 0x03;
                    uint8_t q1 = (wb0 >> 2) & 0x03;
                    uint8_t q2 = (wb0 >> 4) & 0x03;
                    uint8_t q3 = (wb0 >> 6) & 0x03;
                    uint8_t q4 = wb1 & 0x03;
                    uint8_t q5 = (wb1 >> 2) & 0x03;
                    uint8_t q6 = (wb1 >> 4) & 0x03;
                    uint8_t q7 = (wb1 >> 6) & 0x03;

                    w_dq[0] = select(select(bias, spb, bool(q0 & 1)), select(lut2, lut3, bool(q0 & 1)), bool(q0 & 2));
                    w_dq[1] = select(select(bias, spb, bool(q1 & 1)), select(lut2, lut3, bool(q1 & 1)), bool(q1 & 2));
                    w_dq[2] = select(select(bias, spb, bool(q2 & 1)), select(lut2, lut3, bool(q2 & 1)), bool(q2 & 2));
                    w_dq[3] = select(select(bias, spb, bool(q3 & 1)), select(lut2, lut3, bool(q3 & 1)), bool(q3 & 2));
                    w_dq[4] = select(select(bias, spb, bool(q4 & 1)), select(lut2, lut3, bool(q4 & 1)), bool(q4 & 2));
                    w_dq[5] = select(select(bias, spb, bool(q5 & 1)), select(lut2, lut3, bool(q5 & 1)), bool(q5 & 2));
                    w_dq[6] = select(select(bias, spb, bool(q6 & 1)), select(lut2, lut3, bool(q6 & 1)), bool(q6 & 2));
                    w_dq[7] = select(select(bias, spb, bool(q7 & 1)), select(lut2, lut3, bool(q7 & 1)), bool(q7 & 2));

                    // The whole point: this decoded chunk is reused by every vector.
                    for (int v = 0; v < vecs_per_tg; v++) {
                        const device IT* xc = xv[v] + k0;
                        float acc = 0.0f;
                        for (int i = 0; i < sub; i++) {
                            acc += static_cast<float>(xc[i]) * w_dq[i];
                        }
                        result[v] += acc;
                    }
                }
            }

            // Reduce over k_lanes only. simd_sum would fold together the several output
            // rows this simdgroup is responsible for.
            for (int v = 0; v < vecs_per_tg; v++) {
                if (k_lanes >= 32) { result[v] += simd_shuffle_down(result[v], 16); }
                if (k_lanes >= 16) { result[v] += simd_shuffle_down(result[v], 8); }
                if (k_lanes >= 8)  { result[v] += simd_shuffle_down(result[v], 4); }
                if (k_lanes >= 4)  { result[v] += simd_shuffle_down(result[v], 2); }
                if (k_lanes >= 2)  { result[v] += simd_shuffle_down(result[v], 1); }
            }

            if (k_lane == 0 && out_row < N) {
                for (int v = 0; v < vecs_per_tg; v++) {
                    if (vec0 + v < M) {
                        y[(vec0 + v) * N + out_row] = static_cast<IT>(result[v]);
                    }
                }
            }
        """
      return MLXFast.metalKernel(
        name: "bonsai_affine_qmv_wide",
        inputNames: ["x", "w", "scales", "biases", "K", "N", "M"],
        outputNames: ["y"],
        source: source)
    }()
  #endif
}
