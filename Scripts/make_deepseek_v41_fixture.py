# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# A tiny DeepSeek-V4.1 checkpoint in the release's own formats, and what the reference makes of it.
#
#   uv venv --python 3.12 .venv-ref
#   uv pip install --python .venv-ref/bin/python torch safetensors numpy sympy pillow
#   .venv-ref/bin/python Scripts/make_deepseek_v41_fixture.py \
#       Tests/IshizukiKitTests/Fixtures/deepseek-v41

import json
import math
import os
import struct
import sys
import tempfile
import types
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
BASE = f"https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{REVISION}/inference/"
OUT = sys.argv[1]
SEED = int(os.environ.get("SEED", "26"))
PROMPT, STEPS = 20, 12

torch.set_default_dtype(torch.float32)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

folder = os.path.join(tempfile.gettempdir(), f"dsv41-reference-{REVISION[:12]}")
os.makedirs(folder, exist_ok=True)
for name in ["model.py", "engram.py", "vision.py", "image_processor.py"]:
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + name, path)
sys.path.insert(0, folder)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FAKE_QUANT = True


def pow2_ceil(r):
    bits = r.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    return ((e + 127) << 23).view(torch.float32)


def e2m1_round(v):
    mag = v.abs()
    code = torch.zeros_like(mag, dtype=torch.long)
    for t in [0.25, 1.25, 2.5, 5.0]:
        code += (mag > t).long()
    for t in [0.75, 1.75, 3.5]:
        code += (mag >= t).long()
    return torch.sign(v) * E2M1[code]


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
    assert inplace, "only the in-place round trip reaches this; the GEMMs run unquantized"
    if not FAKE_QUANT:
        return x
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // block_size, block_size)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax * torch.tensor(1.0 / 448.0, dtype=torch.float32))
    q = (blocks / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    x.copy_(q.reshape(shape).to(x.dtype))
    return x


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert inplace
    if not FAKE_QUANT:
        return x
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // block_size, block_size)
    amax = blocks.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6.0 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn).float()
    else:
        s = pow2_ceil(amax.clamp_min(6.0 * 2.0**-126) * torch.tensor(1.0 / 6.0, dtype=torch.float32))
    q = e2m1_round((blocks / s).clamp(-6.0, 6.0)) * s
    x.copy_(q.reshape(shape).to(x.dtype))
    return x


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    gathered = torch.gather(
        kv.float().unsqueeze(1).expand(b, m, kv.size(1), d), 2,
        idx.clamp(min=0).unsqueeze(-1).expand(b, m, idx.size(-1), d))
    logits = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered) * softmax_scale
    valid = (idx >= 0).unsqueeze(2)
    logits = torch.where(valid, logits, torch.tensor(float("-inf")))
    top = logits.amax(-1, keepdim=True).clamp_min(-1e30)
    w = torch.where(valid, torch.exp(logits - top), torch.tensor(0.0))
    denom = w.sum(-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, h, 1) - top)
    return (torch.einsum("bmhk,bmkd->bmhd", w, gathered) / denom).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def unavailable(*_, **__):
    raise RuntimeError("the fixture runs its GEMMs unquantized")


kernel = types.ModuleType("kernel")
kernel.act_quant = act_quant
kernel.fp4_act_quant = fp4_act_quant
kernel.sparse_attn = sparse_attn
kernel.hc_split_sinkhorn = hc_split_sinkhorn
kernel.fp8_gemm = kernel.fp4_gemm = unavailable
sys.modules["kernel"] = kernel

import engram as ref_engram  # noqa: E402

VOCAB = 512
keys = rng.integers(0, 360, VOCAB)
first = {}
TOKEN_MAP = [first.setdefault(int(k), len(first)) for k in keys]
COMPRESSED = len(first)
ref_engram.build_compressed_token_map = lambda tokenizer: (TOKEN_MAP, COMPRESSED)

import model as ref  # noqa: E402

layers = 10
config = dict(
    max_batch_size=1, max_seq_len=64, temperature=0.0, dtype="bf16", expert_dtype=None,
    vocab_size=VOCAB, dim=128, moe_inter_dim=64, n_layers=layers, n_mtp_layers=3, n_heads=4,
    n_routed_experts=8, n_shared_experts=1, n_activated_experts=2, score_func="sqrtsoftplus",
    route_scale=1.5, swiglu_limit=10.0, q_lora_rank=64, head_dim=64, rope_head_dim=16,
    norm_eps=1e-20, o_groups=2, o_lora_rank=32, window_size=8,
    compress_ratios=(0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0),
    kv_source_layers=(2, 4, 6), index_source_layers=(2, 4, 6, 8),
    compress_rope_theta=160000.0, original_seq_len=65536, rope_theta=10000.0, rope_factor=16,
    beta_fast=32, beta_slow=1, index_n_heads=8, index_head_dim=32, index_topk=4,
    candidate_source_layer=6, candidate_topk_blocks=3, candidate_block_size=4,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1, 4), engram_max_ngram_size=4, engram_vocab_size=101, engram_n_heads=2,
    engram_head_dim=32, engram_pad_id=2, engram_compressed_vocab_size=COMPRESSED,
    dspark_block_size=5, dspark_noise_token_id=500, dspark_target_layer_ids=(7, 8, 9),
    dspark_markov_rank=32, dspark_n_routed_experts=4, dspark_n_activated_experts=2,
    vision_n_layers=2, vision_dim=64, vision_n_heads=4, vision_inter_dim=96,
    vision_patch_size=14, vision_downsample_ratio=3, vision_max_n_token=64,
    vision_min_pixels=42 * 42, image_token_id=501,
)
layout = ref_engram.EngramLayout.from_args(types.SimpleNamespace(**config, engram_num_embeddings=()))
config["engram_num_embeddings"] = tuple(sum(p for order in per for p in order) for per in layout.primes)
args = ref.ModelArgs(**config)

margins = []


def margin(scores, k, what):
    finite = torch.where(torch.isfinite(scores), scores, torch.tensor(float("nan")))
    ordered = finite.sort(dim=-1, descending=True).values
    for row in ordered.reshape(-1, ordered.size(-1)):
        row = row[~torch.isnan(row)]
        if row.numel() > k:
            gap = (row[k - 1] - row[k]).item()
            margins.append((what, gap / max(row.abs().max().item(), 1e-12)))


original_select = ref.select_candidate_blocks


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    margin(scores, topk_blocks, "candidate blocks")
    return original_select(logits, compress_lens, topk_blocks, block_size)


ref.select_candidate_blocks = select_candidate_blocks


def indexer_forward(self, x, qr, latent, start_pos, offset):
    # The reference reads `shared_attn.index_k` through a pointer an owner only moves when it
    # publishes; on the decode steps where its group is still filling, an owner of ratio 2 would
    # score against the last publisher's keys instead of its own.
    if self.owns_k:
        ref.shared_attn.index_k = self.k_cache
    bsz, seqlen, _ = x.size()
    ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
    if self.owns_k and latent is not None:
        freqs = (
            self.freqs_cis[: seqlen - seqlen % ratio : ratio]
            if start_pos == 0
            else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        )
        k = self.k_norm(self.wk(latent))
        ref.apply_rotary_emb(k[..., -rd:], freqs)
        fp4_act_quant(k, ref.fp4_block_size, True)
        self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
        ref.shared_attn.index_k = self.k_cache
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
    ref.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
    fp4_act_quant(q, ref.fp4_block_size, True)
    index_k = ref.shared_attn.index_k[:bsz, : end_pos // ratio]
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
    index_score = torch.einsum("bshd,btd->bsht", q, index_k)
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if start_pos == 0:
        compress_lens = (torch.arange(1, seqlen + 1) // ratio).unsqueeze(-1)
        index_score.masked_fill_(torch.arange(seqlen // ratio) >= compress_lens, -torch.inf)
    else:
        compress_lens = end_pos // ratio
    if self.is_candidate_source:
        ref.shared_attn.candidates = ref.select_candidate_blocks(
            index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size)
    elif self.uses_candidates:
        index_score = index_score.masked_fill(~ref.shared_attn.candidates, -torch.inf)
    topk = min(self.index_topk, end_pos // ratio)
    margin(index_score, topk, f"index top-k, layer {self.layer_id}")
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).int()


original_indexer_init = ref.Indexer.__init__


def indexer_init(self, args, layer_id):
    original_indexer_init(self, args, layer_id)
    self.layer_id = layer_id


ref.Indexer.__init__ = indexer_init
ref.Indexer.forward = indexer_forward

original_gate = ref.Gate.forward


def gate_forward(self, x, image_mask=None):
    scores = F.softplus(ref.linear(x.float(), self.weight.float())).sqrt()
    bias = self.bias
    if image_mask is not None and self.bias_vl is not None:
        bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, self.bias)
    margin(scores + bias, self.topk, "routing")
    return original_gate(self, x, image_mask)


ref.Gate.forward = gate_forward
ref.linear = lambda x, weight, bias=None: F.linear(x.to(weight.dtype), weight)


def head_forward(self, x, full_logits=True):
    return F.linear(x.float(), self.weight)


ref.ParallelHead.forward = head_forward

model = ref.Transformer(args, tokenizer=None).float()


def fp8_blocks(w):
    out, inp = w.shape
    blocks = w.reshape(out // 32, 32, inp // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax * torch.tensor(1.0 / 448.0))
    codes = (blocks / s).clamp(-448, 448).to(torch.float8_e4m3fn)
    exact = (codes.float() * s).reshape(out, inp)
    scale = (torch.log2(s).round() + 127).to(torch.uint8).reshape(out // 32, inp // 32)
    return codes.reshape(out, inp).view(torch.uint8), scale, exact


def fp8_rows(w):
    rows, width = w.shape
    groups = w.reshape(rows, width // 32, 32)
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = pow2_ceil(amax * torch.tensor(1.0 / 448.0))
    codes = (groups / s).clamp(-448, 448).to(torch.float8_e4m3fn)
    exact = (codes.float() * s).reshape(rows, width)
    scale = (torch.log2(s).round() + 127).to(torch.uint8).reshape(rows, width // 32)
    return codes.reshape(rows, width).view(torch.uint8), scale, exact


def fp4_rows(w):
    out, inp = w.shape
    groups = w.reshape(out, inp // 32, 32)
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(6 * 2.0**-126)
    s = pow2_ceil(amax * torch.tensor(1.0 / 6.0))
    values = e2m1_round((groups / s).clamp(-6, 6))
    code = (E2M1.view(1, 1, 1, 8) == values.abs().unsqueeze(-1)).float().argmax(-1)
    code = code | ((values < 0).long() << 3)
    code = code.reshape(out, inp)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    exact = (values * s).reshape(out, inp)
    scale = (torch.log2(s).round() + 127).to(torch.uint8).reshape(out, inp // 32)
    return packed.view(torch.int8), scale, exact


FP8 = {"wq_a", "wq_b", "wkv", "wo_a", "wo_b", "main_proj"}
FP8_ALSO = {"shared_experts.w1", "shared_experts.w2", "shared_experts.w3", "engram.wkv"}

checkpoint = {}
reference_state = {}
params = dict(model.named_parameters())
for name, p in params.items():
    if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
        continue
    if ".engram.embed." in name:
        continue
    shape = tuple(p.shape)
    leaf = name.rsplit(".", 1)[0]
    kind = leaf.rsplit(".", 1)[-1]
    if name.endswith("attn_sink"):
        value = torch.randn(shape) * 0.5
    elif "hc_" in name and name.endswith("_fn"):
        value = torch.randn(shape) * 0.05
    elif "hc_" in name and name.endswith("_scale"):
        value = 0.5 + torch.rand(shape)
    elif "hc_" in name and name.endswith("_base"):
        value = torch.randn(shape) * 0.5
    elif name.endswith("gate.bias") or name.endswith("gate.bias_vl"):
        value = torch.randn(shape) * 0.1
    elif "norm" in kind or name.endswith("q_weight") or name.endswith("k_weight"):
        value = 1.0 + 0.1 * torch.randn(shape)
    elif p.ndim == 2:
        value = torch.randn(shape) / math.sqrt(shape[1])
    else:
        value = 0.1 * torch.randn(shape)

    quantize = (kind in FP8 and "indexer" not in name) or any(leaf.endswith(k) for k in FP8_ALSO)
    quantize = quantize or name.endswith("indexer.wq_b.weight")
    if ".experts." in name and ".shared_experts." not in name:
        packed, scale, exact = fp4_rows(value)
        checkpoint[name] = ("I8", packed)
        checkpoint[leaf + ".scale"] = ("F8_E8M0", scale)
        reference_state[name] = exact
    elif quantize and p.ndim == 2:
        codes, scale, exact = fp8_blocks(value)
        checkpoint[name] = ("F8_E4M3", codes)
        checkpoint[leaf + ".scale"] = ("F8_E8M0", scale)
        reference_state[name] = exact
    elif p.dtype == torch.float32 and ("hc_" in name or name.endswith(("attn_sink", "gate.bias", "gate.bias_vl"))):
        checkpoint[name] = ("F32", value)
        reference_state[name] = value
    else:
        rounded = value.to(torch.bfloat16)
        checkpoint[name] = ("BF16", rounded)
        reference_state[name] = rounded.float()

tables = {}
for layer, rows in zip(args.engram_layer_ids, args.engram_num_embeddings):
    codes, scale, exact = fp8_rows(torch.randn(rows, args.engram_head_dim))
    checkpoint[f"layers.{layer}.engram.embed.weight"] = ("F8_E4M3", codes)
    checkpoint[f"layers.{layer}.engram.embed.scale"] = ("F8_E8M0", scale)
    tables[layer] = exact

missing, unexpected = model.load_state_dict(reference_state, strict=False)
missing = [m for m in missing if ".engram.embed." not in m and not m.startswith("mtp.") or
           (m.startswith("mtp.") and m.split(".", 2)[-1] not in ("embed.weight", "head.weight"))]
missing = [m for m in missing if ".engram.embed." not in m]
assert not unexpected, unexpected
assert not missing, missing


def engram_lookup(module, layer):
    def forward(indices):
        return tables[layer][indices]
    module.forward = forward


for layer in args.engram_layer_ids:
    engram_lookup(model.layers[layer].engram.embed, layer)


def reset():
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if name == "score_state":
                buffer.fill_(-torch.inf)
            elif name.endswith("cache") or name == "kv_state":
                buffer.zero_()


tokens = torch.from_numpy(rng.integers(0, VOCAB, PROMPT + STEPS)).long().unsqueeze(0)
tokens[0, 5] = args.engram_pad_id


def run(fake_quant):
    global FAKE_QUANT
    FAKE_QUANT = fake_quant
    reset()
    stages = {}
    hooks = []

    def keep(key, pick=lambda out: out):
        def hook(module, inputs, out):
            stages.setdefault(key, pick(out).float().clone())
        return hook

    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(keep(f"stage_{i}", lambda out: out[0])))
        if layer.engram is not None:
            hooks.append(layer.engram.register_forward_hook(keep(f"engram_{i}")))
    hooks.append(model.layers[2].attn.register_forward_hook(keep("attn_2")))
    hooks.append(model.layers[6].attn.register_forward_hook(keep("attn_6")))
    hooks.append(model.layers[0].ffn.register_forward_hook(keep("ffn_0")))

    hashes = model.engram_hash(tokens[:, :PROMPT], 0, None)
    reset()
    _, logits, main_hidden = model(tokens[:, :PROMPT], 0)
    for h in hooks:
        h.remove()
    prefill = logits.float().clone()
    next_ids = logits[:, -1].argmax(-1)
    model.forward_spec(next_ids, main_hidden, 0)

    steps, drafts, confidences, draft_ids = [], [], [], []
    for t in range(PROMPT, PROMPT + STEPS):
        _, logits, main_hidden = model(tokens[:, t : t + 1], t)
        steps.append(logits[:, -1].float().clone())
        ids, draft_logits, confidence = model.forward_spec(logits[:, -1].argmax(-1), main_hidden, t)
        drafts.append(draft_logits.float().clone())
        confidences.append(confidence.float().clone())
        draft_ids.append(ids.int().clone())

    reset()
    _, whole, _ = model(tokens, 0)
    return {
        "prefill_logits": prefill,
        "decode_logits": torch.cat(steps, 0),
        "whole_logits": whole.float(),
        "dspark_logits": torch.cat(drafts, 0),
        "dspark_confidence": torch.cat(confidences, 0),
        "dspark_ids": torch.cat(draft_ids, 0),
        "hashes": hashes.int(),
        **stages,
    }


quantized = run(True)
continuous = run(False)

import image_processor as ref_images  # noqa: E402

VIT_H, VIT_W = 6, 9
LLM_H, LLM_W = -(-VIT_H // 3), -(-VIT_W // 3)
pixels = torch.rand(3, VIT_H * 14, VIT_W * 14) * 2 - 1
patches = pixels.reshape(3, VIT_H, 14, VIT_W, 14).permute(1, 3, 0, 2, 4).reshape(VIT_H * VIT_W, 3, 14, 14)
span_types = ref_images.image_token_types(LLM_H, LLM_W)
before, after = tokens[0, :6], tokens[0, 6:14]
vision_ids = torch.cat([before, torch.full((span_types.numel(),), args.image_token_id), after]).unsqueeze(0)
vision_types = torch.cat([torch.full((6,), ref_images.TEXT), span_types,
                          torch.full((after.numel(),), ref_images.TEXT)]).unsqueeze(0)
image = ref_images.ImageInput(6, patches, VIT_H, VIT_W, span_types)
FAKE_QUANT = True
reset()
with torch.inference_mode():
    _, vision_logits, _ = model(vision_ids, 0, images=[[image]], token_types=vision_types)
    features = model.encode_image(patches, VIT_H, VIT_W).float()
    vision_hashes = model.engram_hash(vision_ids, 0, vision_types < 0)
print("vision span", span_types.tolist(), "logits", tuple(vision_logits.shape))

for name, result in [("fake quant", quantized), ("continuous", continuous)]:
    drift = (result["whole_logits"][0, PROMPT:] - result["decode_logits"]).abs().max().item()
    scale = result["whole_logits"].abs().max().item()
    print(f"{name}: prefill+decode against one prefill {drift:.2e} of {scale:.2f}")
tight = sorted(margins, key=lambda m: m[1])[:5]
print("tightest selection margins:", [(w, f"{g:.2e}") for w, g in tight])
assert tight[0][1] > float(os.environ.get("MIN_MARGIN", "1e-4")), "a top-k is decided by too thin a margin; pick another SEED"


def save(path, tensors, metadata=None):
    header, blobs, offset = {}, [], 0
    if metadata:
        header["__metadata__"] = metadata
    for name in sorted(tensors):
        dtype, value = tensors[name]
        data = value.contiguous().view(torch.uint8).numpy().tobytes() if value.dtype in (
            torch.bfloat16,) else value.contiguous().numpy().tobytes()
        header[name] = {"dtype": dtype, "shape": list(value.shape),
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for blob in blobs:
            f.write(blob)


os.makedirs(OUT, exist_ok=True)
save(os.path.join(OUT, "model.safetensors"), checkpoint, {"format": "pt"})

text_config = {
    "model_type": "deepseek_v41_text", "vocab_size": VOCAB, "hidden_size": args.dim,
    "moe_intermediate_size": args.moe_inter_dim, "num_hidden_layers": layers,
    "num_attention_heads": args.n_heads, "num_key_value_heads": 1, "head_dim": args.head_dim,
    "qk_rope_head_dim": args.rope_head_dim, "q_lora_rank": args.q_lora_rank,
    "o_lora_rank": args.o_lora_rank, "o_groups": args.o_groups, "hidden_act": "silu",
    "swiglu_limit": args.swiglu_limit, "rms_norm_eps": args.norm_eps,
    "tie_word_embeddings": False, "max_position_embeddings": 1048576,
    "rope_theta": args.rope_theta,
    "rope_scaling": {"rope_type": "yarn", "factor": args.rope_factor, "beta_fast": args.beta_fast,
                     "beta_slow": args.beta_slow,
                     "original_max_position_embeddings": args.original_seq_len},
    "n_routed_experts": args.n_routed_experts, "n_shared_experts": 1,
    "num_experts_per_tok": args.n_activated_experts, "scoring_func": args.score_func,
    "topk_method": "noaux_tc", "norm_topk_prob": True,
    "routed_scaling_factor": args.route_scale, "sliding_window": args.window_size,
    "compress_ratios": list(args.compress_ratios), "compress_rope_theta": args.compress_rope_theta,
    "kv_source_layer_ids": list(args.kv_source_layers),
    "index_source_layer_ids": list(args.index_source_layers),
    "index_n_heads": args.index_n_heads, "index_head_dim": args.index_head_dim,
    "index_topk": args.index_topk, "candidate_source_layer_id": args.candidate_source_layer,
    "candidate_topk_blocks": args.candidate_topk_blocks,
    "candidate_block_size": args.candidate_block_size, "hc_mult": args.hc_mult,
    "hc_sinkhorn_iters": args.hc_sinkhorn_iters, "hc_eps": args.hc_eps,
    "engram_layer_ids": list(args.engram_layer_ids),
    "engram_num_embeddings": list(args.engram_num_embeddings),
    "engram_max_ngram_size": args.engram_max_ngram_size,
    "engram_vocab_size": args.engram_vocab_size, "engram_n_heads": args.engram_n_heads,
    "engram_head_dim": args.engram_head_dim, "engram_pad_token_id": args.engram_pad_id,
    "engram_compressed_vocab_size": COMPRESSED, "num_nextn_predict_layers": args.n_mtp_layers,
    "dspark_block_size": args.dspark_block_size,
    "dspark_noise_token_id": args.dspark_noise_token_id,
    "dspark_target_layer_ids": list(args.dspark_target_layer_ids),
    "dspark_markov_rank": args.dspark_markov_rank,
    "dspark_n_routed_experts": args.dspark_n_routed_experts,
    "dspark_num_experts_per_tok": args.dspark_n_activated_experts,
}
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump({
        "architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41",
        "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
        "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                                "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
                                "expert_dtype": "fp4"},
        "text_config": text_config,
        "image_token_id": args.image_token_id,
        "vision_config": {
            "model_type": "deepseek_v41_vision", "num_hidden_layers": args.vision_n_layers,
            "hidden_size": args.vision_dim, "num_attention_heads": args.vision_n_heads,
            "intermediate_size": args.vision_inter_dim, "patch_size": args.vision_patch_size,
            "rope_theta": args.vision_rope_theta, "downsample_ratio": args.vision_downsample_ratio,
            "max_image_tokens": args.vision_max_n_token, "min_pixels": args.vision_min_pixels,
            "max_wh_ratio": None,
        },
    }, f, indent=1)

golden = {"tokens": ("I32", tokens.int()),
          "token_map": ("I32", torch.tensor(TOKEN_MAP, dtype=torch.int32)),
          "multipliers": ("I64", model.engram_hash.multipliers.long()),
          "primes": ("I64", model.engram_hash.primes.long().reshape(len(args.engram_layer_ids), -1)),
          "offsets": ("I64", model.engram_hash.offsets.long())}
golden["vision_ids"] = ("I32", vision_ids.int())
golden["vision_types"] = ("I32", vision_types.int())
golden["vision_pixels"] = ("F32", pixels.contiguous())
golden["vision_features"] = ("F32", features.contiguous())
golden["vision_logits"] = ("F32", vision_logits.float().contiguous())
golden["vision_hashes"] = ("I32", vision_hashes.int().contiguous())
for label, result in [("", quantized), ("continuous_", continuous)]:
    for name, value in result.items():
        if name == "hashes" and label:
            continue
        if label and not name.endswith(("logits", "confidence", "ids")):
            continue
        kind = "I32" if value.dtype == torch.int32 else "F32"
        golden[label + name] = (kind, value.contiguous())
save(os.path.join(OUT, "reference.safetensors"), golden)

with open(os.path.join(OUT, "engram_token_map.json"), "w") as f:
    json.dump(TOKEN_MAP, f)


def byte_chars():
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars, extra = {}, 0
    for b in range(256):
        if b in keep:
            chars[b] = chr(b)
        else:
            chars[b] = chr(256 + extra)
            extra += 1
    return chars


vocab = {"<bos>": 0, "<eos>": 1}
for b, c in byte_chars().items():
    vocab[c] = 2 + b
for i in range(258, VOCAB):
    vocab[f"x{i}"] = i
with open(os.path.join(OUT, "tokenizer.json"), "w") as f:
    json.dump({
        "version": "1.0",
        "added_tokens": [{"id": 0, "content": "<bos>", "special": True},
                         {"id": 1, "content": "<eos>", "special": True}],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True},
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    }, f)
print("tensors", len(checkpoint), "bytes", os.path.getsize(os.path.join(OUT, "model.safetensors")))
