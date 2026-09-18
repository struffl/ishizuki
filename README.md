![](/assets/ishizuki.jpg)

# Ishizuki (石付き)

*Ishizuki* is the bonsai style where the tree is grown over rock, its roots gripping the stone.

Native macOS inference engine for **Ternary Bonsai 2 27B** on [MLX Swift](https://github.com/ml-explore/mlx-swift).
2-bit Hadamard-rotated weights, vision, tool calling, OpenAI + Anthropic APIs. No Python.

Uses minimal memory and offers the fastest possible speeds on Apple Silicon.

Requires Apple Silicon and macOS 15+.

## Contents

- [Install](#install)
- [Update](#update)
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

Or take the loose binary, `ishizuki-<version>-macos-arm64.tar.gz` — the executable and the
metallib it loads, signed and notarized. Put the two anywhere, together:

```bash
tar -xzf ishizuki-<version>-macos-arm64.tar.gz
./ishizuki-<version>-macos-arm64/ishizuki serve
```

From source needs Xcode 16+ and [`just`](https://github.com/casey/just) (`brew install just`):

```bash
just install         # rootless build + install to ~/.local
just package         # signed + notarized .pkg (identities from .env — see .env.example)
just tarball         # signed + notarized loose binary + metallib
just dist            # both
```

## Update

```bash
ishizuki update
```

Replaces the executable and the metallib beside it, in place, wherever ishizuki was
installed from — the `.pkg`, the tarball, or `just install`. Models, configuration files,
and the launchd agent are left alone, and a server that is already running keeps the code
it started with until you restart it.

Nothing is written until both checks pass: the download's sha256 matches the digest GitHub
publishes for that asset, and the new executable carries an intact Developer ID signature
from the same team as the binary it replaces. If either fails, or a file cannot be moved
into place, the old files are put back.

```bash
ishizuki update --check        # report what is available, change nothing
ishizuki update --tag v0.1.7   # install a specific release
ishizuki update --force        # reinstall the current version
ishizuki --version
```

`serve` and `launch` mention a newer release in their header, from a check refreshed in the
background once a day. Set `ISHIZUKI_NO_UPDATE_CHECK=1` to turn that off.

Updating writes to the install directory: a `~/.local` install needs no password, a
system-wide one needs `sudo`. The updater looks for a release asset named
`ishizuki-<tag>-macos-<arch>.tar.gz`, which is what `just tarball` produces.

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

`response_format` with a `json_schema` constrains decoding: the schema is expanded to the set of
documents it admits, held as a byte trie, and each step samples only from the tokens that stay
inside it, so the reply is schema-valid by construction rather than by retry. The schema's
language has to be finite — enums, bounded integer ranges, booleans, `additionalProperties:
false`. Anything unbounded is refused with a 400, as is GBNF `grammar`, so a client that probes
for a constraint mechanism falls through to the one that works.

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

## Memory

Nothing is sized from the machine up front. A cold server holds the weights, one prefix
cache of 8K tokens, and a 0.5 GB buffer pool. Each tier doubles when the work runs into it,
and stops at what the wired ceiling can still hold:

| Tier | Starts at | Doubles when | Stops at |
|---|---|---|---|
| context reserve | 8K tokens | a prompt asks for more | 262K, or what fits |
| prefix slots | 1 | a warm prefix is evicted for want of a slot | 8, or what fits |
| buffer pool | 0.5 GB | MLX saturates the pool twice running | the KV it recycles, 8 GB at most |

Growth is a high-water mark: a tier holds until the model is unloaded, then returns to the
floor. Context wins over slots — reserving more per conversation sheds slots rather than
overcommitting the ceiling. Each step is logged, and the dashboard carries the live tier,
the peak, and what is actually held.

Pin a tier and it stops moving:

```bash
ishizuki serve \
  --cache-slots 4 \         # pin prefix caches
  --cache-limit-gb 2 \      # pin MLX buffer pool
  --idle-timeout 120 \      # release the pool when idle
  --evict-timeout 900 \     # unload the model when idle
  --wire-gb 9 \             # keep resident while busy
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

## Sampling

Defaults: temperature 0.7, every truncation warper off. When enabled the chain runs
`top-k → top-p → min-p → temperature`, llama.cpp order, temperature last — so min-p's surviving
set does not dissolve as temperature rises. Pinned in `SamplerTests`.

## Dependencies

`mlx-swift`, `swift-argument-parser`, `swift-jinja` (chat templates).

## Layout

```
Sources/IshizukiKit/
  Config/      pack config + validation, model fetch, self-update
  Core/        Hadamard, packed layers, caches, KV quantization, custom kernels
  Text/        attention, gated delta net, MLP, RoPE + scaling
  Vision/      encoder, image processing, multimodal splicing
  Tokenizer/   byte-level BPE, chat template, tool-call parsing
  Generate/    model, sampler, generation loop, prefix cache
  Speculative/ drafters + verified decoding
  Serve/       HTTP, OpenAI + Anthropic APIs, residency, politeness
Sources/ishizuki/     generate serve launch pull update install-agent
                      verify verify-logits kv-bench spec-bench batch-check
                      context-bench kernel-check
```

## License

MIT © 2026 Sarah Truffle. See [LICENSE](LICENSE).

---

Icon art from [StockCake](https://stockcake.com), public domain.
