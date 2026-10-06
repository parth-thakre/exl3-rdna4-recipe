"""Where do the 16 GB go with 3.0 bpw + DFlash2 and a 64k Q4 KV cache? Allocated VRAM after each piece,
plus the biggest tensors and where the embedding table lives. Our numbers came from the 5.0 bpw draft;
DRAFT_DIR picks the draft (default: the 3.0 bpw requant). usage: vram_breakdown.py"""
import os, sys, time
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch
from exllamav3 import Config, Model, Cache
from exllamav3.cache import CacheLayer_quant
from common import MODEL_DIR, path

DRAFT_DIR = os.environ.get("DRAFT_DIR", path("models", "Qwen3.8-27B-DFlash2-EXL3-3.0bpw"))
GB = 1e9
def mem(): torch.cuda.synchronize(); return torch.cuda.memory_allocated() / GB
free0, total = torch.cuda.mem_get_info()
print(f"card: {total / GB:.2f} GB, free before anything {free0 / GB:.2f} GB")
steps = []
m0 = mem()
dcfg = Config.from_directory(DRAFT_DIR)
draft = Model.from_config(dcfg)
draft.load(); m1 = mem(); steps.append((f"DFlash2 draft weights ({os.path.basename(DRAFT_DIR)})", m1 - m0))
cfg = Config.from_directory(MODEL_DIR)
model = Model.from_config(cfg)
model.load(); m2 = mem(); steps.append(("main model weights (3.0 bpw)", m2 - m1))
cache = Cache(model, max_num_tokens = 65536, layer_type = CacheLayer_quant, k_bits = 4, v_bits = 4,
              max_batch_size = 1, max_history = draft.caps.get("default_draft_size", 4))
m3 = mem(); steps.append(("main KV cache 64k Q4 (+ recurrent state)", m3 - m2))
dcache = Cache(draft, max_num_tokens = 65536, layer_type = CacheLayer_quant, k_bits = 4, v_bits = 4)
m4 = mem(); steps.append(("draft KV cache 64k Q4", m4 - m3))
for name, gb in steps: print(f"  {gb:6.2f} GB  {name}")
print(f"  {m4:6.2f} GB  total allocated by torch")

# biggest individual weights of the main model
sizes = []
def walk(m):
    yield m
    for c in getattr(m, "modules", []) or []:
        yield from walk(c)
for mod in walk(model):
    for attr in ("embedding", "weight", "trellis"):
        t = getattr(mod, attr, None)
        if isinstance(t, torch.Tensor):
            sizes.append((t.numel() * t.element_size() / GB, mod.key + "." + attr, str(t.device), str(t.dtype)))
    inner = getattr(mod, "inner", None)
    if inner is not None and isinstance(getattr(inner, "trellis", None), torch.Tensor):
        t = inner.trellis
        sizes.append((t.numel() * t.element_size() / GB, mod.key + ".trellis", str(t.device), str(t.dtype)))
sizes.sort(reverse = True)
print("biggest tensors:")
seen = set()
for gb, key, dev, dt in sizes:
    if key in seen: continue
    seen.add(key)
    if len(seen) > 6: break
    print(f"  {gb:6.3f} GB  {dev:7s} {dt:15s} {key}")
emb = [s for s in sizes if "embed" in s[1]]
print("embedding:", emb[:1])
cats = {}
for gb, key, dev, dt in sizes:
    if dev.startswith("cuda"):
        c = "trellis (EXL3)" if key.endswith(".trellis") else ("embedding" if "embed" in key else "fp16 weights / norms")
        cats[c] = cats.get(c, 0) + gb
print("main model on GPU by kind:", {k: round(v, 2) for k, v in cats.items()})
torch.cuda.synchronize(); time.sleep(1); sys.stdout.flush(); os._exit(0)
