![](/assets/ishizuki.jpg)

# Ishizuki (石付き)

*Ishizuki* is the bonsai style where the tree is grown over rock, its roots gripping the stone.

Native macOS inference engine for **Ternary Bonsai 2 27B** on [MLX Swift](https://github.com/ml-explore/mlx-swift).
2-bit Hadamard-rotated weights, vision, tool calling, OpenAI + Anthropic APIs. No Python.

Requires Apple Silicon and macOS 15+.

## Contents

- [Install](#install)
- [Agents](#agents) — Hermes, Claude Code, Pi
- [Run](#run)
- [Serve](#serve)
  - [launchd](#launchd)
- [Benchmarks](#benchmarks)
- [Context](#context)
  - [macOS wired ceiling](#macos-wired-ceiling)
- [Politeness](#politeness)
- [Memory](#memory)
- [Verify](#verify)
- [Sampling](#sampling)
- [Dependencies](#dependencies)
- [Layout](#layout)
- [License](#license)

## Install

Download `ishizuki-<version>.pkg` from [Releases](../../releases) and open it — signed and
notarized, and it installs to your home folder (`~/.local`), so there's **no admin password**.

**Uninstall is one command, also without a password** (the installer's last screen shows it too):

```bash
~/.local/libexec/ishizuki/uninstall.sh
```

Add `--purge` to also delete downloaded models.

From source needs Xcode 16+ and [`just`](https://github.com/casey/just) (`brew install just`):

```bash
just install         # rootless build + install to ~/.local
just package         # signed + notarized .pkg (identities from .env — see .env.example)
```

## Agents

Wire a coding agent to the local model. `launch` starts the server (or attaches to one
already running), configures the tool, and hands over the terminal.

**Hermes Agent**

```bash
ishizuki launch hermes
```

**Claude Code**

```bash
ishizuki launch claude
```

**Pi**

```bash
ishizuki launch pi
```

Pass arguments through after `--`, e.g. `ishizuki launch hermes -- chat -q "hello"`.
`ishizuki launch list` shows the tools, `--print-config` shows the changes without
applying them.

Configuring it yourself instead — Claude Code needs only environment variables:

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


## Run

```bash
ishizuki generate --prompt "Explain gated delta networks."
ishizuki generate --image photo.jpg --prompt "What is in this picture?"
ishizuki serve --kv-bits 3.5
```

`--model` defaults to `~/Library/Application Support/Ishizuki/models/Ternary-Bonsai-2-27B-mlx-2bit`.
Drop the MLX pack there, or point `--model` at one anywhere.
Run `ishizuki <command> --help` for the full option surface.

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

While it runs, `serve` draws a live dashboard: every in-flight request as its own row with
phase (queued, prefill, decode), a progress meter, its current tok/s and token counts, above
running session totals — prefill and decode rates, requests, tokens in/out, prefix-cache hit
rate, memory and thermal state. Requests waiting on the generation queue show as `queued`.
It needs a TTY; under launchd or a pipe it falls back to plain log lines, as does
`--disable-dashboard`.

Tool calling works in both shapes, including parallel calls and tool-result round trips. The
model emits an XML-ish `<tool_call>` form; the server converts it to `tool_calls` / `tool_use`
and strips reasoning and call syntax from streamed text. Thinking is off unless `--thinking`.

Consecutive requests reuse the previous cache when the prompt extends it (measured: 459 of 479
tokens reused on a follow-up turn). Reuse needs an exact prefix — the recurrent layers cannot be
rewound to a divergence point.

### launchd

```bash
ishizuki install-agent --kv-bits 3.5 --evict-timeout 900
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/studio.ishizuki.server.plist
```

Uses `--lazy-load`: the port listens but no weights are resident until a request arrives.

## Benchmarks

M1 Max, release build: decode 17.5-20.1 tok/s, prefill up to 138.9 tok/s. Full throughput,
KV quantization, batching, speculative decoding, and custom-kernel numbers are in
[BENCHMARKS.md](BENCHMARKS.md).

## Context

Native 262 144. Extend with `--context-scale` (YaRN by default; `--rope-scaling ntk|linear|none`):

```bash
ishizuki generate --context-scale 2 --prompt "..."     # ~512K
```

Beyond the trained length is extrapolation and unmeasured. Only the 16 full-attention layers
carry position; the 48 recurrent layers have no maximum.

Per-token KV cost is 2 × 4 KV heads × 256 dims × 16 layers = 32768 values:

| Bits | Per token | 262K | 1M | + weights + recurrent |
|---|---|---|---|---|
| fp16 | 64.0 KB | 16.0 GB | 64.0 GB | 72.7 GB — does not fit |
| 4 | 18.0 KB | 4.5 GB | 18.0 GB | 26.7 GB |
| 3.5 | 16.0 KB | 4.0 GB | 16.0 GB | 24.7 GB |
| 3 | 14.0 KB | 3.5 GB | 14.0 GB | 22.7 GB |

1M fits in ~25 GB at 3.5-bit. Decode there is ~15 tok/s at roofline, so single digits real.
One-shot 1M prefill is hours (O(L²) over 16 layers) — practical for context accumulated across a
session, not for ingesting 1M at once.

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

Ceilings, not recommendations. Exceeding it degrades sharply rather than erroring.
`--wire-gb` reserves within the ceiling; it cannot raise it.

## Politeness

Default `--politeness adaptive`: utility QoS, quarter-size prefill chunks (shorter GPU
submissions), and backoff under thermal pressure or Low Power Mode. Costs ~6% prefill and ~0.5%
decode.

| Level | Prefill | Decode |
|---|---|---|
| `normal` | 138.9 tok/s | 20.1 tok/s |
| `adaptive` (default) | 130.3 tok/s | 20.0 tok/s |
| `background` | — | **>90× slower** |

Metal exposes no public command-queue priority, so submission length is the only real control
over GPU disruption. `background` uses the Darwin background band, which throttles disk I/O —
fatal against memory-mapped weights. It exists for completeness; do not use it.

## Memory

```bash
ishizuki serve \
  --idle-timeout 120 \      # release caches
  --evict-timeout 900 \     # unload model
  --wire-gb 9 \             # keep resident while busy
  --cache-limit-gb 2 \      # cap MLX buffer pool
  --lazy-load
```

Reload after eviction is ~1 s; the weights are memory-mapped.

## Verify

```bash
just verify              # Hadamard + 2-bit path, full forward, custom kernels
just test                # sampler unit tests
ishizuki kv-bench        # KV quantization cost
ishizuki spec-bench      # speculative speedup + losslessness
ishizuki batch-check     # batch correctness + scaling
ishizuki context-bench   # scaling with context length
```

Current: Hadamard path bit-exact (max|Δ| 0.00000); full 64-layer forward vs the reference Python
stack correlates 0.999999 with matching argmax on prefill and decode.

## Sampling

Defaults: temperature 0.7, every truncation warper off. When enabled the chain runs
`top-k → top-p → min-p → temperature`, llama.cpp order, temperature last — so min-p's surviving
set does not dissolve as temperature rises. Pinned in `SamplerTests`.

## Dependencies

`mlx-swift`, `swift-argument-parser`, `swift-jinja` (chat templates).

Not `swift-transformers`: its `swift-huggingface` dependency imports AppKit, and against the
macOS 27 SDK `CAOpenGLLayer.h` still references the removed OpenGL types, so the build fails.
The byte-level BPE is implemented directly against `tokenizer.json`.

`Scripts/run-tests.sh` copies `mlx.metallib` beside the test binary and re-signs; MLX's loader
does not find it inside an `.xctest` bundle.

## Layout

```
Sources/IshizukiKit/
  Config/      pack config + validation
  Core/        Hadamard, packed layers, caches, KV quantization, custom kernels
  Text/        attention, gated delta net, MLP, RoPE + scaling
  Vision/      encoder, image processing, multimodal splicing
  Tokenizer/   byte-level BPE, chat template, tool-call parsing
  Generate/    model, sampler, generation loop, prefix cache
  Speculative/ drafters + verified decoding
  Serve/       HTTP, OpenAI + Anthropic APIs, residency, politeness
Sources/ishizuki/     generate serve install-agent verify verify-logits
                      kv-bench spec-bench batch-check context-bench kernel-check
```

## License

MIT © 2026 Sarah Truffle. See [LICENSE](LICENSE).

---

Icon art from [StockCake](https://stockcake.com), public domain.
