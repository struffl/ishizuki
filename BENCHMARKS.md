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
| before | 61.6 tok/s | 3.4 tok/s |
| after | **92.1 tok/s** | **9.9 tok/s** |

The dequantizer's output is bit-identical to what it produced before, on every block type. The
matvec sums a row in a different order, so it is not. Against the exact product of that same
decoded weight, both the old kernel and the new one land near 3e-9 of the sum of the term
magnitudes, neither consistently nearer than the other — six orders below the ~4e-3 that bf16
activations already cost. Greedy generation tracks the old kernel's for a hundred-odd tokens
and then parts at a near-tie, which is what a reordered sum does.

Per projection, best of five alternating runs of `ishizuki ggml-bench` (GB/s of block bytes
read; MLX's affine 4-bit matvec on the same shapes came out within 1% of itself across the two
builds, so they saw the same machine):

| Type | Shape | Before | After | |
|---|---|---|---|---|
| IQ1_S | 17408 × 5120 | 10.5 | 58.5 | 5.57× |
| IQ2_S | 5120 × 6144 | 14.3 | 78.8 | 5.51× |
| IQ1_M | 248320 × 5120 | 13.7 | 68.4 | 4.99× |
| IQ3_S | 6144 × 5120 | 24.4 | 95.8 | 3.93× |
| IQ2_XS | 5120 × 17408 | 32.1 | 86.8 | 2.70× |
| IQ2_XXS | 5120 × 17408 | 52.2 | 82.6 | 1.58× |
| IQ3_XXS | 1024 × 5120 | 25.7 | 39.0 | 1.52× |
| Q4_K | 1024 × 5120 | 34.7 | 50.9 | 1.47× |
| Q2_K | 5120 × 17408 | 68.3 | 90.4 | 1.32× |

What changed is who reads what. A simdgroup used to take thirty-two blocks at once, a lane
each, which meant thirty-two reads of `x` five hundred bytes apart for what fits in four — and
idle lanes whenever a row held fewer blocks than the simd width. It now takes one block at a
time with lane `L` holding the eight weights at offset `8L`, so the lanes of a simdgroup ask
for `x`, and for the block's own index and sign bytes, as single contiguous runs. Every format
is read as vectors rather than one weight at a time, and the dequantizer shares those bodies
rather than keeping a second, scalar copy of each one: it is 1.5–2.6× faster for the same
bytes, which is where the prefill number comes from.

The formats that carry sign bits then gave up another 1.12–1.22×, which took finding out what
the kernel is actually short of. It is not arithmetic throughput in general — thirty-two added
integer operations per group of eight are free for Q2_K and Q4_K — but the i-quants have no
slack at all: the same thirty-two cost them a third of their throughput. Deriving a vector's
four sign masks from a sign byte is sixteen of those operations, so the masks are read from a
table instead. It only needs a nibble's worth at a time, which makes it sixteen entries of
four: two hundred and fifty-six bytes, small enough that staging it costs nothing. An
eight-kilobyte table indexed by the whole byte was measurably worse.

What did not work, all measured: prefetching a block's bytes an iteration ahead (0.90×, the
compiler schedules better without the loop-carried registers), unrolling the block loop by two
or four (1.00×), several partial sums to break the accumulator chain (0.91–1.00×), threadgroup
sizes from 128 to 1024 (flat), grids packed two bits per weight (a net loss — every grid holds
only three distinct values, but unpacking costs more than the memory saves), grids expanded to
floats to skip a conversion (0.89–1.02×), one sixteen-byte activation load in place of two
eight-byte ones (0.91–0.97×), and `simd_shuffle` as a register-file gather for IQ4_XS's
codebook (0.34× — a divergent lane index is emulated).

What remains is the codebook lookup itself. A threadgroup read whose address comes from a
device load costs about 1.25× against one whose address is known early, and nothing above
recovers it; the way out would be to transcode the blocks at load time so the index is already
the codes, which buys perhaps 1.25× for 10–40% more memory and would stop the bytes being
ggml's own.

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

## Pipelined decode

The unconstrained decode loop queues the next forward, and the pick after it, before reading
the token it is about to emit, so the GPU is never idle while the host samples, detokenizes and
streams. The serial loop waited on the GPU twice a token, once for the step and once for the
pick. A constrained decode still runs serially: it needs the allowed set on the host each step.

| Greedy, 96 tokens | tok/s |
|---|---|
| serial | 14.35 |
| pipelined | 22.48 |

Same tokens, and the cache is left exactly where the serial loop leaves it. Bonsai 2-bit, M1 Max,
2026-09-23; `BonsaiRuntime.pipelineDecode` turns it off.

## Drafting in a served turn

`Generator` can draft inside a served turn: prompt lookup when the reply repeats the context,
the pack's MTP head otherwise, verified a block at a time, exact for greedy and for sampling
(a rejected draft is replaced from the residual distribution). It is **off by default**
(`BonsaiRuntime.speculativeDecode`) because on the IQ2_XS GGUF it does not pay yet:

| Qwen3.8-27B IQ2_XS GGUF, served turn | drafting off | drafting on | tokens / round |
|---|---|---|---|
| prose, greedy | 11.05 | 13.21 | 1.85 (MTP) |
| prose, temp 0.7 min-p 0.05 | 11.41 | 10.04 | 1.64 |
| code edit, greedy | 10.92 | 15.28 | 3.71 (lookup) |
| code edit, temp 0.7 min-p 0.05 | 10.79 | 15.29 | 3.71 |

A rejected draft is not replayed on its own: the tokens it kept ride at the front of the next
round's block, since the token after them is already known from the verify. Greedy output is
token for token what plain decoding gives. Sampled prose still loses: acceptance falls to about
55%, so more rounds carry, and a three- or four-row forward on the matvec costs 145–175 ms.
M1 Max, 2026-09-23.

### Few-row GGUF multiply

The GGUF matvec decodes a block once but pays every row its own loads, multiplies and
reduction, so eight rows cost three to four times one. `GGMLKernels.matmulFew` runs the same
per-format block bodies into simdgroup matrix fragments instead. Cost of eight rows against
one, on the file's largest tensor of each format:

| Format | matvec | few-row |
|---|---|---|
| Q2_K | 3.34× | 1.50× |
| Q4_K | 2.47× | 1.83× |
| IQ1_S | 2.82× | 0.94× |
| IQ1_M | 2.85× | 1.04× |
| IQ2_XXS | 3.49× | 1.38× |
| IQ2_XS | 3.53× | 1.36× |
| IQ2_S | 3.04× | 1.35× |
| IQ3_XXS | 3.60× | 1.54× |
| IQ3_S | 2.75× | 1.30× |
| IQ4_XS (lm head) | 4.36× | 1.79× |

It serves five to sixteen rows (eight at a time); fewer stay on the matvec, which ties it at
two and is close at four.

## Speculative verify

MLX's affine matmul costs close to one full weight read per row until it switches to its tiled
path past eight rows: on the 2-bit Bonsai pack an 8-token forward took 279 ms against 52 ms for
one, so a longer draft bought nothing. `VerifyMatmul` decodes each weight once, straight into a
simdgroup matrix fragment, and applies it to up to eight rows. It is on by default for 5–8 rows,
where it wins; below that MLX's own path is faster.

| Verify width | MLX | `VerifyMatmul` |
|---|---|---|
| 6 tokens | 281 ms | 207 ms |
| 8 tokens | 279 ms | 134 ms |

| n-gram drafts (98% accepted) | MLX verify | `VerifyMatmul` |
|---|---|---|
| 4 per round | 17.6 tok/s | 18.2 tok/s |
| 7 per round | 15.8 tok/s | **29.5 tok/s** |

Lossless: every run matches greedy token for token. M1 Max, 2026-09-23. An M1 has no GPU
matrix units, so eight rows cost about twice one at best; chips with neural accelerators in the
GPU should close more of that.
