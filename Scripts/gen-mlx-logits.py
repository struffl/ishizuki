import sys, json, struct
import mlx.core as mx
from mlx_lm import load

path = sys.argv[1]; prompt = sys.argv[2]; out = sys.argv[3]
model, tokenizer = load(path)
ids = tokenizer.encode(prompt, add_special_tokens=False)
logits = model(mx.array([ids]))[0, -1].astype(mx.float32)
mx.eval(logits)
v = logits.tolist()
with open(out, "wb") as f:
    f.write(b"LLAMALG1")
    f.write(struct.pack("<II", len(ids), len(v)))
    for t in ids: f.write(struct.pack("<i", t))
    for x in v: f.write(struct.pack("<f", x))
print("wrote", len(ids), "tokens and", len(v), "logits", file=sys.stderr)
