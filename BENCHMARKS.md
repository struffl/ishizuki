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

## Custom kernels

Both are bit-exact and both are **off by default** — measured slower than MLX's own paths on
MLX 0.31.1 (what mlx-swift 0.31.6 vendors). Kept for other MLX versions.

| Kernel | Flag | Result |
|---|---|---|
| `qmv_wide` (batch 2–5) | `--qmv-wide` | 0.58–1.16× vs `quantizedMM` |
| Fused Hadamard | `--fused-hadamard` | 18.8–19.0 vs 19.8 tok/s |

Verify with `ishizuki kernel-check`.
