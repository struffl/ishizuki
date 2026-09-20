# Benchmarks

M1 Max 64 GB, release build, measured. Reproduce with `make bench` and the
`ishizuki *-bench` / `batch-check` commands (see [README](README.md#verify)).

| Prompt | Generated | Prefill | Decode |
|---|---|---|---|
| 512 tok | 1024 tok | 138.9 tok/s | 20.1 tok/s |
| 3 622 tok | 64 tok | 95.4 tok/s | 17.5 tok/s |
| 7 218 tok | 8 tok | 69.9 tok/s | — |

Decode 17.5–20.1 tok/s; speculative decoding 24.4 tok/s on repetitive text. Load ~1 s (mmap).
Decode roofline at 400 GB/s is ~46 tok/s.

Decode scales with memory bandwidth: ~5 tok/s on M4, ~11 on M4 Pro, ~15–22 on M1/M4 Max,
~30 on M3 Ultra.

## Contents

- [KV quantization](#kv-quantization)
- [Batching](#batching)
- [Speculative decoding](#speculative-decoding)
- [GGUF](#gguf)
- [Custom kernels](#custom-kernels)

## KV quantization

`--kv-bits 3.5` = 3-bit keys, 4-bit values. Over 1045 tokens, greedy, vs an fp16 reference:

| Bits | Cache | KB/token | Agreement |
|---|---|---|---|
| fp16 | 80.0 MB | 78.4 | reference |
| 8 | 39.9 MB | 39.1 | 100% |
| 4 | 24.9 MB | 24.4 | 100% |
| **3.5** | **23.0 MB** | **22.6** | **100%** |
| 3 | 21.1 MB | 20.7 | 100% |
| 2 | 17.4 MB | 17.1 | 9%, diverges at token 4 |

Slightly slower at short context (17.8 vs 19.2 tok/s); wins once the cache dominates bandwidth.
This is MLX affine quantization at TurboQuant's bit split, not TurboQuant's codec.

## Batching

Bit-exact to batch 8 (`batch-check`).

| Batch | Prefill | Aggregate decode |
|---|---|---|
| 2 | 2.28× | 1.15× |
| 4 | 3.82× | 1.40× |
| 8 | 4.52× | 1.40× |

Decode saturates at 1.4×. The serving layer generates one request at a time; concurrent serving
needs a batch scheduler, per-sequence caches, and ragged-length handling.

## Speculative decoding

```
plain greedy      : 19.93 tok/s
speculative (n=4) : 24.37 tok/s   acceptance 93.3%   1.22×
lossless: matches greedy exactly over 71 tokens
```

`NgramDrafter` replays repeated context. No second model, no extra memory. Pays off on
repetitive work; proposes nothing on free-form prose.

## GGUF

A GGUF is read in its own blocks — nothing is re-quantized and nothing is expanded to be
multiplied — so the speed of a GGUF is the speed of the kernel that decodes it.

`Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp.gguf`, 8.16 GiB over eleven block types, 1832-token prompt:

| | Prefill | Decode |
|---|---|---|
| before | 62.5 tok/s | 3.5 tok/s |
| after | **92.9 tok/s** | **8.2 tok/s** |

The dequantizer's output is bit-identical to what it produced before, on every block type. The
matvec sums a row in a different order, so it is not. Against the exact product of that same
decoded weight, both the old kernel and the new one land near 3e-9 of the sum of the term
magnitudes, neither consistently nearer than the other — six orders below the ~4e-3 that bf16
activations already cost. Greedy generation tracks the old kernel's for a hundred-odd tokens
and then parts at a near-tie, which is what a reordered sum does.

Per projection, best of four runs of `ishizuki ggml-bench` (GB/s of block bytes read; MLX's
affine 4-bit matvec on the same shapes measured 292 and 288 GB/s in both runs, so the two runs
saw the same machine):

| Type | Shape | Before | After | |
|---|---|---|---|---|
| IQ1_S | 17408 × 5120 | 11.2 | 61.7 | 5.51× |
| IQ1_M | 248320 × 5120 | 14.8 | 72.5 | 4.90× |
| IQ2_S | 5120 × 6144 | 15.2 | 70.0 | 4.61× |
| IQ3_S | 6144 × 5120 | 26.1 | 85.3 | 3.27× |
| IQ2_XS | 5120 × 17408 | 33.9 | 75.6 | 2.23× |
| Q4_K | 1024 × 5120 | 34.0 | 49.7 | 1.46× |
| IQ3_XXS | 1024 × 5120 | 25.3 | 36.2 | 1.43× |
| Q2_K | 5120 × 17408 | 71.6 | 94.5 | 1.32× |
| IQ2_XXS | 5120 × 17408 | 54.3 | 69.6 | 1.28× |

What changed is who reads what. A simdgroup used to take thirty-two blocks at once, a lane
each, which meant thirty-two reads of `x` five hundred bytes apart for what fits in four — and
idle lanes whenever a row held fewer blocks than the simd width. It now takes one block at a
time with lane `L` holding the eight weights at offset `8L`, so the lanes of a simdgroup ask
for `x`, and for the block's own index and sign bytes, as single contiguous runs. Every format
is read as vectors rather than one weight at a time, and the dequantizer shares those bodies
rather than keeping a second, scalar copy of each one: it is 1.5–2.6× faster for the same
bytes, which is where the prefill number comes from.

Fused decode covers batch 1–16, measured against expanding the weight and calling a real
matmul, which does not overtake it until the high twenties.

## Custom kernels

Both are bit-exact and both are **off by default** — measured slower than MLX's own paths on
MLX 0.31.1 (what mlx-swift 0.31.6 vendors). Kept for other MLX versions.

| Kernel | Flag | Result |
|---|---|---|
| `qmv_wide` (batch 2–5) | `--qmv-wide` | 0.58–1.16× vs `quantizedMM` |
| Fused Hadamard | `--fused-hadamard` | 18.8–19.0 vs 19.8 tok/s |

Verify with `ishizuki kernel-check`.
