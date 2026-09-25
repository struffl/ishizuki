![](/assets/ishizuki.jpg)

# Ishizuki (石付き)

*Ishizuki* is the bonsai style where the tree is grown over rock, its roots gripping the stone.

Native macOS inference engine *originally* for **Ternary Bonsai 2 27B** on [MLX Swift](https://github.com/ml-explore/mlx-swift),
now supports GGUF for most Qwen architectures, Gemma 4 architectures, Qwen next-gen MoEs, and more!
2-bit Hadamard-rotated weights, vision, tool calling, OpenAI + Anthropic APIs. No Python.

Uses minimal memory and offers the fastest possible speeds on Apple Silicon.

Requires Apple Silicon and macOS 26.

## Contents

- [What's in it](#whats-in-it)
- [Install](#install)
- [Agents](#agents) — Hermes, Claude Code, Pi
- [Serve](#serve)
- [Companion](#companion) — the iPhone app
- [Tools](#tools) — quantize, measure, prune
- [Benchmarks](#benchmarks)
- [Context](#context)
  - [macOS wired ceiling](#macos-wired-ceiling)
- [Politeness](#politeness)
- [Sampling](#sampling)
- [Dependencies](#dependencies)
- [Layout](#layout)
- [Credits](#credits)
- [License](#license)

## What's in it

One Swift engine, no Python, written against Metal and MLX Swift from the tensor layout up.

**Faster than the usual tools, on the same Mac.** Qwen3.8-27B on one idle M1 Max, greedy,
tok/s on math / code prompts ([how it was measured](BENCHMARKS.md#engines-side-by-side)):

| Weights | llama.cpp | mlx-lm | ishizuki |
|---|---|---|---|
| Ternary Bonsai 2 27B, 2-bit | — | 20.9 / 19.7 | **45.0 / 45.0** |
| GGUF IQ2_XS | 12.8 / 13.0 | — | **23.9 / 24.7** |
| EXL3 2.0 bpw | — | — | **28.7 / 35.4** |
| OrcaSAQ2 27B | — | — | **28.9 / 33.3** |

**Reads what people actually ship.**
- GGUF in eleven quantization formats, decoded inside the matmul and never expanded.
- EXL3 trellis codes, the only way to run them on a Mac.
- OrcaSAQ2, JANG repacks with their Hadamard rotation, oQ4e, and plain MLX affine packs.
- Architectures: Qwen3.8's hybrid gated-delta-net and attention stack, Gemma 4, Qwen3.8-Flash-Next
  (`qwen4_exp`) up to the 125B-A6B, and DeepSeek-V4.1-Flash straight from its 48 release shards,
  with the routed experts streamed off the SSD.

**Speculative decoding that pays on Apple Silicon.**
- z-lab's DFlash 2 block drafter, ported to Swift, doubles decode on code and math.
- Prompt lookup drafting runs inside the pipelined decode loop, so a step with nothing to draft
  costs what a plain one does. It is on by default and makes a served code edit 16% faster.
- Pack MTP heads draft greedy turns.

**Kernels written for the verify.** MLX's quantized matmul costs about one decode step per row,
which is what keeps speculative decoding from paying on a Mac. Ishizuki's verify kernels decode
each weight once, straight into a simdgroup matrix fragment, and multiply it against every
drafted row, so eight rows cost about two and a half steps. There is one for MLX affine codes and
one for GGUF blocks, and the affine one takes a projection only from the row count where it
beats MLX, which the weight's size sets.

The affine kernel was also tested on hardware it was never written on. In the MLX Fast Bonsai 2
speed contest, whose runs are timed on an M5, a port of it was entered on the leading entry's own
tree and draft depth, and the kernel was the only difference between the two. Decode went from
18.9 to 16.1 ms a token, 15% faster than MLX's own kernels, and the drafts were accepted as they
had been. That took first place for a time, at 1.655× the reference engine.

**Details that took measuring.**
- macOS only skips the page cache when a read's file offset and its buffer share a 16 KB phase.
  Streamed expert reads go through page-aligned scratch, so a long prompt no longer squeezes
  other apps into compressed memory.
- MLX caches a custom kernel by name, and the name ignores input dtypes. Every kernel here puts
  its activation dtype in its template, so an fp16 model and its fp32 drafter never share a build.
- Streamed experts are locked in memory, because an idle slot the system compressed cost more to
  hit than the disk read it replaced.

Plus a menu bar app with OpenAI and Anthropic APIs for coding agents, a paired iPhone companion,
and pack tools for quantizing, measuring and streaming experts.

## Install

Ishizuki is a menu bar app. The bonsai in the status bar drops down what is loaded, the rate,
and the memory held; the window behind it carries the full dashboard, the model library, the
tools and the settings.

From source needs Xcode 26+, [`just`](https://github.com/casey/just) and
[`xcodegen`](https://github.com/yonaskolb/XcodeGen):

```bash
brew install just xcodegen
just app-run         # build and launch
just app-store       # archive and export a Mac App Store package
```

### Models

The app is sandboxed, so it reads its own container and nothing else until you say otherwise.
Packs land in `~/Library/Containers/studio.ishizuki.app/…/Ishizuki/models` when the app fetches
them, and the Models tab will take any HuggingFace repo by name — `org/model`, or
`org/model:file.gguf` to lift one quantization out of a repo carrying several.

Packs you already have stay where they are: point the Models tab at the folder holding them and
it reads them in place, nothing copied.

JANG repacks load as they are, including `dealignai/Bonsai-2-27B-CRACK-Ternary-JANG`. Ishizuki
reads the Hadamard rotation from `config.json` and the norm layout from `jang_config.json`, and
the quantized vision tower runs as it ships. A pack that declares a transform Ishizuki can't
apply is refused rather than loaded as plain affine.

`deepseek-ai/DeepSeek-V4.1-Flash` loads from its 48 release shards as they ship: the fp8 and
fp4 weights are multiplied where they lie, the routed experts are read into slots as tokens
route to them, and the two 100 GB n-gram tables are read a row at a time. About 10 GB stays in
memory, plus the slots, which are sized to the machine — 40 a layer, 30 GB, on a 64 GB Mac.
Everything else is disk, so the disk sets the pace: off a USB hard drive, an 11-token prompt
took 8 minutes and each token after it 43 seconds. Its DSpark draft head only runs when the
experts are held in memory. A prompt longer than 128 tokens runs the decoder over its last 128
only, the way DeepSeek serves the model; `BonsaiRuntime.deepseekBoundedReplay` turns that off.

## Agents

Point a coding agent at the local model. Start the server from the status bar; the dashboard
header carries the two base URLs with a button to copy each.

Claude Code needs only environment variables:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8128
export ANTHROPIC_AUTH_TOKEN=ishizuki
export ANTHROPIC_MODEL=ternary-bonsai-2-27b
export CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144
```

Hermes needs a named provider in `~/.hermes/config.yaml`, then
`hermes --provider ishizuki -m ternary-bonsai-2-27b`:

```yaml
providers:
  ishizuki:
    name: Ishizuki
    base_url: http://127.0.0.1:8128/v1
    model: ternary-bonsai-2-27b
    default_model: ternary-bonsai-2-27b
    key_env: ISHIZUKI_API_KEY
    api_mode: chat_completions
    context_length: 262144
```

Pi needs a provider in `~/.pi/agent/models.json`, then
`pi --provider ishizuki --model ternary-bonsai-2-27b`:

```json
{
  "providers": {
    "ishizuki": {
      "baseUrl": "http://127.0.0.1:8128/v1",
      "api": "openai-completions",
      "apiKey": "ishizuki",
      "models": [
        {
          "id": "ternary-bonsai-2-27b",
          "name": "ternary-bonsai-2-27b",
          "reasoning": false,
          "input": ["text", "image"],
          "contextWindow": 262144,
          "maxTokens": 32768,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
```


## Serve

One port, both API shapes. Streaming (SSE) on both.

```
POST /v1/chat/completions        OpenAI     — Hermes, Pi, agent frameworks
POST /v1/messages                Anthropic  — Claude Code
POST /v1/messages/count_tokens
GET  /v1/models,  GET /health
```

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8128/v1     # OpenAI clients
export ANTHROPIC_BASE_URL=http://127.0.0.1:8128     # Claude Code
```

The vision tower is read off disk on the first image request rather than at startup, so a
text-only session never pays for it. Settings can load it up front instead.

Settings holds the rest: the port, KV bits, how long an idle pool is held before it is freed
and the model unloaded, the scheduling politeness, the disk budget for prefixes, whether to
serve at launch and whether to open at login.

## Companion

An iPhone app that borrows this Mac: every conversation the window holds, the folders it shares,
and its shell. Turns run on the Mac's resident pack, so the phone is reading the same weights,
the same prefix cache and the same readout the desktop window is.

Switch it on in **Settings → Companion**, choose which folders to share, then **Pair a phone**
and point the camera at the square. The phone keeps the key in its keychain; nothing is typed
twice.

```
just phone-run     # the companion in the simulator
just phone-device  # a Release build for a real iPhone
```

How it is reached, in the order the phone tries:

| Address | Comes from | Works |
| --- | --- | --- |
| `100.x.y.z` | Tailscale | anywhere both ends are on the tailnet |
| `192.168.x.y` | the local router | the same network |
| `_ishizuki._tcp` | Bonjour | the same network, no address to type |

The pairing square carries all of them, best first, so a Mac that moves between a desk and a
tailnet is found again without being paired twice.

Every connection is TLS with a pre-shared key: the key is in the square, so completing a
handshake is itself the proof of pairing. Pairing is open for three minutes and hands the phone a
token of its own, which is what **Forget** revokes. A device that once knew the key still knows
it, so forgetting *every* phone rotates the key instead.

Without the Mac — asleep, or off the tailnet — the phone falls back to Apple's on-device model
under **Ask this iPhone**: no files, no shell, and nothing saved to the Mac.

### Reconnecting and cache

The iPhone keeps a disposable, protected cache of previously loaded conversations, shared
folder listings, model listings, and text previews (up to 32 MB per paired Mac). Saved content
appears before network refresh and stays readable during outages. Conversation streams
reconnect, stalled requests time out, and the chat list refreshes when returning to the app.
Forgetting a Mac clears its cache. Messages and shell commands are not automatically replayed
after a failed request, to avoid executing an action twice.

Dashboard cache memory is a snapshot published by the generation owner at checkpoints and
turn completion. It may lag during generation; drawing the dashboard never inspects mutable
MLX cache arrays from another thread.

## Tools

The Tools tab carries the work that is not serving:

- **Measure** — the kernel and batch checks, and the KV, context, prefill and speculative
  sweeps, run against the selected pack. Output streams into a console and can be cancelled.
- **Quantize** — build a mixed-width pack from a full-precision checkpoint. The plan names its
  destination and estimates its size before it starts, and it can measure real activations
  first rather than quantizing blind to them.
- **Stream Experts** — split a mixture-of-experts pack in two: the shared half stays in memory,
  the routed experts move to a file per sparse layer and are read a few at a time. The values
  are copied through untouched, so a streamed pack answers token for token like the pack it
  came from — slower, on a machine that could not otherwise hold it. How many experts a layer
  keeps resident is the slot budget under Settings › Model.
- **Prefix cache** — what is archived between runs, and getting rid of it.

## Benchmarks

M1 Max, release build, Ternary Bonsai 2 27B: plain decode 22.5 tok/s, 45 with DFlash on code and
math, prefill up to 139 tok/s. The engine comparison, DeepSeek off the SSD, KV quantization,
batching, speculative decoding and custom-kernel numbers are in [BENCHMARKS.md](BENCHMARKS.md).

## Context

Native 262 144, stretched from Settings (YaRN) up to about 1M.

Beyond the trained length is extrapolation and unmeasured. Only the 16 full-attention layers
carry position; the 48 recurrent layers have no maximum.

Per-token KV cost is 2 × 4 KV heads × 256 dims × 16 layers = 32768 values:

| Bits | Per token | 262K | 1M | + weights + recurrent |
|---|---|---|---|---|
| fp16 | 64.0 KB | 16.0 GB | 64.0 GB | 72.7 GB — does not fit |
| 4 | 18.0 KB | 4.5 GB | 18.0 GB | 26.7 GB |
| 3.5 | 16.0 KB | 4.0 GB | 16.0 GB | 24.7 GB |
| 3 | 14.0 KB | 3.5 GB | 14.0 GB | 22.7 GB |

### macOS wired ceiling

```bash
sysctl iogpu.wired_limit_mb     # 55296 here; 0 means ~75% of RAM
```

Weights 8.6 GB + recurrent 0.15 GB; the rest is KV budget.

| Ceiling | KV headroom | fp16 | 3.5-bit |
|---|---|---|---|
| 12 GB (16 GB Mac) | ~3 GB | ~49K | ~197K |
| 18 GB (24 GB Mac) | ~9 GB | ~147K | ~590K |
| 48 GB (64 GB Mac) | ~39 GB | ~639K | ~2.5M |

Ceilings, not recommendations. Exceeding it degrades sharply rather than erroring. The wired
reservation in Settings claims memory within the ceiling; it cannot raise it.

## Politeness

Default `normal`: user-initiated QoS and full-size prefill chunks. `adaptive` trades some
speed for sharing the machine: utility QoS, quarter-size prefill chunks (shorter GPU
submissions), and backoff under thermal pressure or Low Power Mode. On a resident pack that
costs ~6% prefill and ~0.5% decode; on a pack that streams its experts the quarter chunks cost
far more, because each chunk reads every expert it routes to (37 tok/s prefill rather than 66
on the 125B-A6B).

| Level | Prefill | Decode |
|---|---|---|
| `normal` (default) | 138.9 tok/s | 20.1 tok/s |
| `adaptive` | 130.3 tok/s | 20.0 tok/s |
| `background` | — | **>90× slower** |

## Sampling

Defaults: temperature 0.7, every truncation warper off. When enabled the chain runs
`top-k → top-p → min-p → temperature`, llama.cpp order, temperature last — so min-p's surviving
set does not dissolve as temperature rises. Pinned in `SamplerTests`.

## Dependencies

`mlx-swift`, `swift-jinja` (chat templates).

## Layout

```
Sources/IshizukiKit/
  Config/      pack config + validation, model fetch, paths
  Core/        Hadamard, packed layers, caches, KV quantization, custom kernels
  Text/        attention, gated delta net, MLP, RoPE + scaling
  Vision/      encoder, image processing, multimodal splicing
  Tokenizer/   byte-level BPE, chat template, tool-call parsing
  Generate/    model, sampler, generation loop, prefix cache
  Speculative/ drafters + verified decoding
  Serve/       HTTP, OpenAI + Anthropic APIs, residency, politeness
  Quantize/    checkpoint scan, bit allocation, pack writing
  Bench/       kernel, batch, KV, context, prefill, speculative harnesses
Sources/IshizukiLink/ the companion protocol: wire types, pairing ticket,
                      TLS-PSK transport, Bonjour discovery, remote shell host
App/Ishizuki/         the menu bar app: status item, dashboard, models,
                      tools, settings, companion listener
App/IshizukiPhone/    the iPhone companion: pairing, conversations, files,
                      shell, on-device fallback
App/Shared/           what both apps draw
```

## Credits

Ishizuki borrows freely. These are the people and projects its answers came from.

**DeepSeek-V4.1**

- [DeepSeek-AI](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) — the release's reference
  inference code, its prompt encoder and golden tests, and the technical report, where the
  encoder–decoder split, the shared compressed attention, DSpark and SWA bounded replay come
  from. Every DeepSeek layer here is checked against their `model.py`.
- [NumPy](https://numpy.org) — DeepSeek draws its n-gram hash multipliers with NumPy's
  `SeedSequence` and PCG64, so those are reimplemented exactly as NumPy has them, down to
  Daniel Lemire's bounded integers. PCG is Melissa O'Neill's.

**Speculative decoding and kernels**

- [z-lab](https://github.com/z-lab/dflash) — DFlash (Jian Chen, Yesheng Liang, Zhijian Liu,
  [arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) and the
  [Qwen3.8-27B DFlash 2 drafter](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2). The Swift
  drafter is a port of their MLX code, and a rejected block is rewound the way theirs does it,
  by running the delta rule again over the positions kept.
- [Inco AI](https://inco.ai/blog/dflash2/) — DFlash 2's convolution and candidate selector, and
  [Splash](https://github.com/incoai/splash), whose split-K decode keeps a narrow projection's
  reduction spread across the GPU.
- [paperniuk](https://github.com/paperniuk/splash/tree/apple7-m1-kernels) — the M1 port of
  Splash and
  [its write-up](https://www.reddit.com/r/oMLX/comments/1wovgy5/you_can_now_run_qwen3827b_on_a_2021_m1_max_at_39/):
  codes turned into exact halves with `| 0x6400` less 1024, half × float32 simdgroup products,
  each weight unpacked once for every row, and the reminder that an M1 has no bf16. The verify
  kernel's code path is built on those.
- [Underdog's Husky](https://husky.underdog.ai) and llm-in-c's Qwen3.8 C/Metal runtime — the
  prompt to measure what a few-row verify costs at all; llm-in-c's staged tile shape was one of
  the layouts tried. Liu Liu — on why a megakernel buys nothing on Metal, which has no grid-wide
  barrier.

**Models and formats**

- [PrismML](https://prismml.com) — Ternary Bonsai 2 27B, the Hadamard pack format, and the pack's
  bundled MLX runtime, the stock baseline the DFlash numbers are measured against.
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) — Qwen3.8 and its hybrid gated delta-net
  architecture; [transformers](https://github.com/huggingface/transformers)' `qwen4_exp`
  modeling code for Qwen3.8-Flash-Next.
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) — the GGUF block formats, read as they
  are, and the sampler order.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (turboderp) — the EXL3 format.
- [oMLX](https://github.com/jundot/omlx) (jundot) — the oQ4e format.
- [dealignai](https://huggingface.co/dealignai) — JANG repacks.
- TurboQuant (Zandieh et al.) — the KV cache's split of 3-bit keys and 4-bit values.
- [Hugging Face](https://huggingface.co) — `tokenizers` and `transformers`, which the tokenizer and
  chat-template tests compare against, and the Hub everything downloads from.
- [MLX](https://github.com/ml-explore/mlx) and [MLX Swift](https://github.com/ml-explore/mlx-swift)
  — the runtime underneath, and the quantized matmul the verify kernel is tested against;
  [mlx-lm](https://github.com/ml-explore/mlx-lm), the stock baseline.

## License

AGPL-3.0-or-later © 2026 Sarah Truffle. See [LICENSE](LICENSE).

---

Icon art from [StockCake](https://stockcake.com), public domain.
