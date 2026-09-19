import sys, struct
import mlx.core as mx
from mlx_lm import load

path, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
model, tok = load(path)
ids = tok.encode(prompt, add_special_tokens=False)
lm = model.model if hasattr(model, "model") else model
lm = getattr(lm, "language_model", lm)
lm = getattr(lm, "model", lm)
print("layers:", len(lm.layers), file=sys.stderr)

h = lm.embed_tokens(mx.array([ids]))
mx.eval(h)
dumps = [("embed", h)]
mask = None
cache = [None] * len(lm.layers)
try:
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
except Exception as e:
    print("cache:", e, file=sys.stderr)

for i, layer in enumerate(lm.layers):
    h = layer(h, mask, cache[i])
    if i in (0, 1, 2, 3, 15, 31, 63):
        mx.eval(h)
        dumps.append((f"layer{i}", h))

with open(out, "wb") as f:
    f.write(b"HIDDEN01")
    f.write(struct.pack("<I", len(dumps)))
    f.write(struct.pack("<I", len(ids)))
    for t in ids: f.write(struct.pack("<i", t))
    for name, a in dumps:
        v = a.astype(mx.float32)[0, -1].tolist()
        n = name.encode()
        f.write(struct.pack("<I", len(n))); f.write(n)
        f.write(struct.pack("<I", len(v)))
        for x in v: f.write(struct.pack("<f", x))
print("wrote", len(dumps), "dumps", file=sys.stderr)
