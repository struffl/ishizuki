#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Cut one projection of the 2-bit pack into an INT8 Neural Engine slice.
#
#   uv run --with coremltools --with numpy Scripts/ane-slice.py --rows 2048 --fraction 0.54
#
# The pack's rows are ternary, so per-output-channel INT8 holds them almost exactly; what costs
# accuracy is activation precision, not the weights. coremltools wraps an fp16 program in fp32
# interface casts, which puts a cast op back on the CPU and doubles the bytes crossing to the
# ANE, so the casts are stripped here and the interface is pinned to fp16.
#
# Time the result against Metal with `ishizuki ane-check --slice <out>`.
import argparse
import json
import os
import struct

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as mt

DEFAULT_PACK = os.path.expanduser(
    "~/Library/Application Support/ishizuki/models/Ternary-Bonsai-2-27B-mlx-2bit/model.safetensors"
)


def read_tensors(path, names):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        out = {}
        for name in names:
            meta = header[name]
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            dtype = {"U32": np.uint32, "F16": np.float16, "F32": np.float32}[meta["dtype"]]
            out[name] = np.frombuffer(raw, dtype=dtype).reshape(meta["shape"]).copy()
        return out


def dequantize(packed, scales, biases, group_size=128, bits=2):
    per_word = 32 // bits
    dout, words = packed.shape
    din = words * per_word
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    codes = (packed[:, :, None] >> shifts[None, None, :]) & np.uint32((1 << bits) - 1)
    codes = codes.reshape(dout, din, order="C").astype(np.float32)
    groups = din // group_size
    codes = codes.reshape(dout, groups, group_size)
    w = codes * scales.astype(np.float32)[:, :, None] + biases.astype(np.float32)[:, :, None]
    return w.reshape(dout, din)


def strip_io_casts(model):
    import coremltools.proto.FeatureTypes_pb2 as ft
    from coremltools.proto import MIL_pb2

    fp16 = MIL_pb2.DataType.FLOAT16
    spec = model.get_spec()
    fn = spec.mlProgram.functions["main"]
    block = fn.block_specializations[list(fn.block_specializations.keys())[0]]

    cast_at = [i for i, op in enumerate(block.operations) if op.type == "cast"]
    if not cast_at:
        return model
    casts = [block.operations[i] for i in cast_at]
    by_output = {op.outputs[0].name: op for op in casts}

    for op in block.operations:
        for arg in op.inputs.values():
            for binding in arg.arguments:
                source = by_output.get(binding.name)
                if source is not None:
                    binding.name = source.inputs["x"].arguments[0].name

    for cast in (op for op in casts if op.outputs[0].name in set(block.outputs)):
        inner = cast.inputs["x"].arguments[0].name
        for producer in block.operations:
            if producer.outputs and producer.outputs[0].name == inner:
                producer.outputs[0].name = cast.outputs[0].name
                producer.outputs[0].type.tensorType.dataType = fp16

    drop = set(cast_at) | {
        i
        for i, op in enumerate(block.operations)
        if op.type == "const" and op.outputs and op.outputs[0].name.endswith("_dtype_0")
    }
    kept = [op for i, op in enumerate(block.operations) if i not in drop]
    del block.operations[:]
    block.operations.extend(kept)

    for spec_input in fn.inputs:
        spec_input.type.tensorType.dataType = fp16
    for port in list(spec.description.input) + list(spec.description.output):
        port.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT16
    return ct.models.MLModel(spec, weights_dir=model.weights_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=DEFAULT_PACK)
    ap.add_argument("--tensor", default="language_model.model.layers.0.mlp.gate_proj")
    ap.add_argument("--rows", type=int, default=2048, help="prefill chunk the slice is built for")
    ap.add_argument("--fraction", type=float, default=0.54)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = [args.tensor + s for s in (".weight", ".scales", ".biases")]
    t = read_tensors(args.pack, names)
    w = dequantize(t[names[0]], t[names[1]], t[names[2]])
    dout, k = w.shape

    channels = max(64, int(round(dout * args.fraction)) // 64 * 64)
    sliced = w[:channels]
    bias = np.zeros((channels,), dtype=np.float16)
    rows = args.rows

    @mb.program(input_specs=[mb.TensorSpec(shape=(rows, k), dtype=mt.fp16)])
    def prog(x):
        return mb.linear(x=x, weight=sliced.astype(np.float16), bias=bias, name="out")

    model = ct.convert(
        prog,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
    )
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"
        )
    )
    model = strip_io_casts(ct.optimize.coreml.linear_quantize_weights(model, config=config))

    out = args.out or f"{args.tensor.rsplit('.', 1)[-1]}_{rows}_{channels}.mlpackage"
    model.save(out)

    x = (np.random.randn(rows, k) * 0.05).astype(np.float16)
    got = np.asarray(model.predict({"x": x})["out"], dtype=np.float32)
    ref = x.astype(np.float32) @ sliced.T.astype(np.float32)
    cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref)))
    print(f"{out}  {channels}/{dout} channels ({channels / dout * 100:.1f}%)  cosine {cos:.6f}")


if __name__ == "__main__":
    main()
