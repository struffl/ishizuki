# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# A tiny hyper-connected checkpoint, and what the reference makes of it.
#
# The fixture in Tests/IshizukiKitTests/Fixtures/qwen4-exp is what this writes. Regenerate it
# only to follow a change upstream, and expect the golden values to move when you do:
#
#   uv venv --python 3.12 .venv-ref
#   uv pip install --python .venv-ref/bin/python torch 'transformers>=4.57'
#   .venv-ref/bin/python Scripts/make_qwen4_exp_fixture.py \
#       Tests/IshizukiKitTests/Fixtures/qwen4-exp

import json, os, sys
import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextModel

OUT = sys.argv[1]
torch.manual_seed(20260922)

config = Qwen4ExpTextConfig(
    vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
    # The delta-net kernel carries a key head dimension of 32 per simd lane, so nothing here
    # goes below that however small the rest of the model is.
    num_attention_heads=2, num_key_value_heads=1, head_dim=32,
    full_attention_interval=4,
    linear_num_key_heads=2, linear_num_value_heads=4,
    linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
    # Wide enough that every expert projection divides a group of 32, so the pack the
    # quantizer writes has its experts quantized rather than carried whole.
    num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
    shared_expert_intermediate_size=32, norm_topk_prob=True,
    hc_count=4, hc_lowrank=8,
    ple_layer_ids=[2], ngram_size=3, heads_per_ngram=2, ple_embed_dim=32,
    ple_conv_kernel_size=4, ngram_vocab_size_base=257,
    indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=32,
    indexer_budget=4096, indexer_compress_ratio=4,
    attn_output_gate=True, output_gate_type="sigmoid", partial_rotary_factor=0.25,
    eos_token_id=1, bos_token_id=1, max_position_embeddings=4096,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                     "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                     "mrope_section": [2, 1, 1]},
)
model = Qwen4ExpTextModel(config).eval()

# Every parameter random: a zero-initialised gate or convolution would hide a wiring mistake.
with torch.no_grad():
    for name, p in model.named_parameters():
        p.normal_(0.0, 0.05 if p.ndim > 1 else 0.2)
    # The shipped checkpoint gives every head the same number of rows rather than a prime of
    # its own, so the fixture does too: the sizes are data, not something to recompute.
    ple = model.layers[1].ple.ple_embedding
    heads, rows = ple.ngram_heads, 257
    ple.ngram_heads_vocab_sizes.copy_(torch.full((heads,), rows, dtype=torch.long))
    ple.ngram_heads_offsets.copy_(torch.arange(heads, dtype=torch.long) * rows)

head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
with torch.no_grad():
    head.weight.normal_(0.0, 0.05)

tokens = torch.tensor([[7, 13, 1, 42, 5, 31, 9, 60, 22, 3, 18, 44]])
with torch.no_grad():
    # Every layer's streams on the way through, so a mismatch can be pinned to the layer that
    # introduced it rather than read off the end.
    stages = {}
    handles = [
        layer.register_forward_hook(
            lambda m, args, out, i=i: stages.update(
                {f"stage_{i}": (out[0] if isinstance(out, tuple) else out).to(torch.float32)}))
        for i, layer in enumerate(model.layers)
    ]
    def probe(name):
        def hook(m, args, out):
            values = out if isinstance(out, tuple) else (out,)
            for j, value in enumerate(values):
                if torch.is_tensor(value):
                    stages[f"{name}_{j}"] = value.to(torch.float32)
        return hook

    handles += [
        model.layers[0].attn_hyper_connection.register_forward_hook(probe("hc0")),
        model.layers[0].linear_attn.register_forward_hook(probe("gdn0")),
        model.layers[0].mlp.register_forward_hook(probe("moe0")),
        model.layers[1].ple.register_forward_hook(probe("ple1")),
    ]
    hidden = model(input_ids=tokens, use_cache=False).last_hidden_state
    logits = head(hidden)
    for handle in handles:
        handle.remove()

os.makedirs(OUT, exist_ok=True)
state = {k: v for k, v in model.state_dict().items()}

# The shipped layout: the table travels beside the model, sharded, with the buffers that
# address it hoisted to the top level.
prefix = "layers.1.ple.ple_embedding."
table = state.pop(prefix + "ngram_embedding.weight")
for name in ["layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"]:
    state["ple_embedding." + name] = state.pop(prefix + name)
cut = table.shape[0] // 2 + 1
state["ngram_embedding.shard_0.weight"] = table[:cut].contiguous()
state["ngram_embedding.shard_1.weight"] = table[cut:].contiguous()

tensors = {"model." + k: v.contiguous() for k, v in state.items()}
tensors["lm_head.weight"] = head.weight.detach().contiguous()
save_file(tensors, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})

settings = config.to_dict()
settings["model_type"] = "qwen4_exp_text"
settings["architectures"] = ["Qwen4ExpForCausalLM"]
settings["tie_word_embeddings"] = False
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(settings, f, indent=1, sort_keys=True, default=str)

save_file(
    {"tokens": tokens.to(torch.int32), "hidden": hidden.to(torch.float32),
     "logits": logits.to(torch.float32), **stages},
    os.path.join(OUT, "reference.safetensors"))

# The same weights with a budget short of the context, so the indexer actually has to choose.
# At ratio 4 and budget 8 a query near the end sees two blocks of the three it could.
indexed = Qwen4ExpTextConfig(**{**config.to_dict(), "indexer_budget": 8})
probe = Qwen4ExpTextModel(indexed).eval()
probe.load_state_dict(model.state_dict())
with torch.no_grad():
    sparse = probe(input_ids=tokens, use_cache=False).last_hidden_state
save_file(
    {"hidden": sparse.to(torch.float32), "logits": head(sparse).to(torch.float32)},
    os.path.join(OUT, "reference-indexed.safetensors"))
settings_indexed = dict(settings)
settings_indexed["indexer_budget"] = 8
with open(os.path.join(OUT, "config-indexed.json"), "w") as f:
    json.dump(settings_indexed, f, indent=1, sort_keys=True, default=str)
print("indexed differs from dense:", (sparse - hidden).abs().max().item())
print("layers", config.num_hidden_layers, "tensors", len(tensors), "hidden", tuple(hidden.shape))
