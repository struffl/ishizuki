// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later

import Foundation
import MLX

public enum FusedHadamard {
  public static let supportedBlocks: Set<Int> = [512, 1024, 2048, 4096]

  public static func apply(_ x: MLXArray, block: Int, signs: MLXArray) -> MLXArray? {
    #if canImport(Metal)
      guard let kernel, supportedBlocks.contains(block) else { return nil }
      let shape = x.shape
      let width = shape[shape.count - 1]
      guard width % block == 0 else { return nil }

      let rows = x.size / width
      let blocksPerRow = width / block
      let totalBlocks = rows * blocksPerRow

      let threads = 256
      let outputs = kernel(
        [x.reshaped([rows, width]), signs, block, blocksPerRow],
        template: [
          ("IT", x.dtype),
          ("BLOCK", block),
          ("THREADS", threads),
        ],
        grid: (threads, totalBlocks, 1),
        threadGroup: (threads, 1, 1),
        outputShapes: [[rows, width]],
        outputDTypes: [x.dtype])
      return outputs[0].reshaped(shape)
    #else
      return nil
    #endif
  }

  #if canImport(Metal)
    private static let kernel: MLXFast.MLXFastKernel? = {
      let source = """
            threadgroup float tile[BLOCK];

            const uint block_id = threadgroup_position_in_grid.y;
            const uint row = block_id / uint(blocks_per_row);
            const uint block_in_row = block_id % uint(blocks_per_row);
            const uint width = uint(blocks_per_row) * BLOCK;
            const uint base = row * width + block_in_row * BLOCK;
            const uint sign_base = block_in_row * BLOCK;
            const uint tid = thread_position_in_threadgroup.x;

            // Load and apply the signs on the way in, so the float32 values only ever
            // exist in threadgroup memory.
            for (uint i = tid; i < BLOCK; i += THREADS) {
                tile[i] = static_cast<float>(x[base + i]) * signs[sign_base + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // In-place radix-2 butterflies: BLOCK/2 of them per stage, log2(BLOCK) stages.
            for (uint h = 1; h < BLOCK; h <<= 1) {
                for (uint b = tid; b < BLOCK / 2; b += THREADS) {
                    // Map a flat butterly index to its pair, skipping the stride each
                    // stage doubles.
                    uint group = b / h;
                    uint offset = b % h;
                    uint j = group * (h << 1) + offset;
                    float a = tile[j];
                    float c = tile[j + h];
                    tile[j] = a + c;
                    tile[j + h] = a - c;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const float scale = rsqrt(float(BLOCK));
            for (uint i = tid; i < BLOCK; i += THREADS) {
                y[base + i] = static_cast<IT>(tile[i] * scale);
            }
        """
      return MLXFast.metalKernel(
        name: "bonsai_fused_hadamard",
        inputNames: ["x", "signs", "block", "blocks_per_row"],
        outputNames: ["y"],
        source: source)
    }()
  #endif
}
