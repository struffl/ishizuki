#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Cut every offloadable projection of the pack into INT8 Neural Engine slices.
#
#   uv run --with coremltools --with numpy Scripts/ane-export.py --fraction 0.54
#
# MLP gate/up on every layer and the gated delta net's token-local z on the recurrent ones. The
# recurrent qkv is deliberately left on the checkpoint-precision Metal path: its error feeds the
# delta rule's state and accumulates along the prompt.
#
# Resumable — slices already on disk are skipped.
import argparse
import importlib.util
import json
import os
import shutil
import struct
import sys
import time

import coremltools as ct
import numpy as np

MODELS = os.path.expanduser("~/Library/Application Support/ishizuki/models")


def load(path):
    spec = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        spec = json.loads(f.read(n))
    return spec, 8 + n


def read(path, base, header, name):
    meta = header[name]
    start, end = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + start)
        raw = f.read(end - start)
    dtype = {"U32": np.uint32, "F16": np.float16, "F32": np.float32}[meta["dtype"]]
    return np.frombuffer(raw, dtype=dtype).reshape(meta["shape"]).copy()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("ane_slice", os.path.join(here, "ane-slice.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=os.path.join(MODELS, "Ternary-Bonsai-2-27B-mlx-2bit"))
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--fraction", type=float, default=0.54)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many slices")
    args = ap.parse_args()

    safetensors = os.path.join(args.pack, "model.safetensors")
    header, base = load(safetensors)
    config = json.load(open(os.path.join(args.pack, "config.json")))
    text = config.get("text_config", config)
    layers = text["num_hidden_layers"]
    kinds = text.get("layer_types") or []

    prefix = "language_model." if any(k.startswith("language_model.") for k in header) else ""
    out_dir = os.path.join(args.pack, f"ane-{args.rows}")
    os.makedirs(out_dir, exist_ok=True)

    targets = []
    for layer in range(layers):
        stem = f"{prefix}model.layers.{layer}"
        targets.append(f"{stem}.mlp.gate_proj")
        targets.append(f"{stem}.mlp.up_proj")
        if layer < len(kinds) and kinds[layer] == "linear_attention":
            targets.append(f"{stem}.linear_attn.in_proj_z")

    done = 0
    started = time.time()
    for index, tensor in enumerate(targets):
        name = tensor[len(prefix):].replace("model.layers.", "") + ".mlmodelc"
        out = os.path.join(out_dir, name)
        if os.path.exists(out):
            continue
        packed = read(safetensors, base, header, tensor + ".weight")
        scales = read(safetensors, base, header, tensor + ".scales")
        biases = read(safetensors, base, header, tensor + ".biases")
        w = mod.dequantize(packed, scales, biases)
        del packed, scales, biases

        model, channels = mod.build_slice(w, args.rows, args.fraction)
        compiled = model.get_compiled_model_path()
        tmp = out + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        shutil.copytree(compiled, tmp)
        os.rename(tmp, out)
        del w, model

        done += 1
        rate = (time.time() - started) / done
        left = (len(targets) - index - 1) * rate
        print(
            f"[{index + 1}/{len(targets)}] {name}  {channels} channels  "
            f"{rate:.0f}s each, ~{left / 60:.0f} min left",
            flush=True,
        )
        if args.limit and done >= args.limit:
            break

    manifest = {
        "rows": args.rows,
        "fraction": args.fraction,
        "slices": sorted(
            f[: -len(".mlmodelc")]
            for f in os.listdir(out_dir)
            if f.endswith(".mlmodelc")
        ),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{len(manifest['slices'])} slices in {out_dir}")


if __name__ == "__main__":
    main()
