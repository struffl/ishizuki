![](/assets/ishizuki.jpg)

# Ishizuki (石付き)

*Ishizuki* is the bonsai style where the tree is grown over rock, its roots gripping the stone.

Native macOS inference engine for **Ternary Bonsai 2 27B** on [MLX Swift](https://github.com/ml-explore/mlx-swift).
2-bit Hadamard-rotated weights, vision, tool calling, OpenAI + Anthropic APIs. No Python.

Uses minimal memory and offers the fastest possible speeds on Apple Silicon.

Requires Apple Silicon and macOS 26.

## Contents

- [Install](#install)
- [Agents](#agents) — Hermes, Claude Code, Pi
- [Serve](#serve)
- [Tools](#tools) — quantize, measure, prune
- [Benchmarks](#benchmarks)
- [Context](#context)
  - [macOS wired ceiling](#macos-wired-ceiling)
- [Politeness](#politeness)
- [Sampling](#sampling)
- [Dependencies](#dependencies)
- [Layout](#layout)
- [License](#license)

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

Requires Apple Silicon and macOS 26.

### Models

The app is sandboxed, so it reads its own container and nothing else until you say otherwise.
Packs land in `~/Library/Containers/studio.ishizuki.app/…/Ishizuki/models` when the app fetches
them, and the Models tab will take any HuggingFace repo by name — `org/model`, or
`org/model:file.gguf` to lift one quantization out of a repo carrying several.

Packs you already have stay where they are: point the Models tab at the folder holding them and
it reads them in place, nothing copied.

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

## Tools

The Tools tab carries the work that is not serving:

- **Measure** — the kernel and batch checks, and the KV, context, prefill and speculative
  sweeps, run against the selected pack. Output streams into a console and can be cancelled.
- **Quantize** — build a mixed-width pack from a full-precision checkpoint. The plan names its
  destination and estimates its size before it starts, and it can measure real activations
  first rather than quantizing blind to them.
- **Prefix cache** — what is archived between runs, and getting rid of it.

## Benchmarks

M1 Max, release build: decode 17.5-20.1 tok/s, prefill up to 138.9 tok/s. Full throughput,
KV quantization, batching, speculative decoding, and custom-kernel numbers are in
[BENCHMARKS.md](BENCHMARKS.md).

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

Default `adaptive`: utility QoS, quarter-size prefill chunks (shorter GPU
submissions), and backoff under thermal pressure or Low Power Mode. Costs ~6% prefill and ~0.5%
decode.

| Level | Prefill | Decode |
|---|---|---|
| `normal` | 138.9 tok/s | 20.1 tok/s |
| `adaptive` (default) | 130.3 tok/s | 20.0 tok/s |
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
App/Ishizuki/         the menu bar app: status item, dashboard, models,
                      tools, settings
```

## License

AGPL-3.0-or-later © 2026 Sarah Truffle. See [LICENSE](LICENSE).

---

Icon art from [StockCake](https://stockcake.com), public domain.
