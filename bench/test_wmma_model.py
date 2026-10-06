"""Full-model check of EXL3_GEMV_WMMA: logits of an 8-token forward (the DFlash2 verify shape) with and without it.
usage: test_wmma_model.py"""
import os, sys, time
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch
from common import MODEL_DIR
from exllamav3 import Config, Model, Cache, Tokenizer

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
model.load()
tokenizer = Tokenizer.from_config(config)
text = "The printing press, invented by Johannes Gutenberg around 1440, transformed European society by"
ids = tokenizer.encode(text)
for rows in (8, 4, 2):
    x = ids[:, :rows]
    out = {}
    for w in ("0", "1"):
        os.environ["EXL3_GEMV_WMMA"] = w
        out[w] = model.forward(x, {"attn_mode": "flash_attn_nc"}).float()
    a, b = out["0"][0], out["1"][0]
    pa, pb = torch.softmax(a, -1), torch.softmax(b, -1)
    kl = (pa * (torch.log(pa + 1e-12) - torch.log(pb + 1e-12))).sum(-1)
    top_same = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"rows {rows}: max |logit diff| {(a - b).abs().max().item():.4f} (logit scale {a.abs().max().item():.1f}), "
          f"max KL {kl.max().item():.2e}, top-1 agreement {top_same:.0%}")
os.environ.pop("EXL3_GEMV_WMMA", None)
torch.cuda.synchronize(); time.sleep(1); sys.stdout.flush(); os._exit(0)
