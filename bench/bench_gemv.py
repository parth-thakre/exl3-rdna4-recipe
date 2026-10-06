"""Achieved weight bandwidth of each EXL3 linear in Qwen3.8-27B 3.0 bpw on gfx1201, per shape.
rows=1 is plain decode, rows=8 is a DFlash2 verify step. Reports µs per call and GB/s of trellis read.
usage: bench_gemv.py [--rows 1 8] [--iters 200]"""
import argparse, os, sys, time
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch
from common import MODEL_DIR
from exllamav3 import Config, Model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default = MODEL_DIR)
ap.add_argument("--rows", type = int, nargs = "+", default = [1, 8])
ap.add_argument("--iters", type = int, default = 200)
args = ap.parse_args()
PEAK = 640.0   # GB/s, RX 9070 XT GDDR6

config = Config.from_directory(args.model)
model = Model.from_config(config)
model.load()

def walk(m):
    yield m
    for c in getattr(m, "modules", []) or []:
        yield from walk(c)

# One full-attention layer (3) and one DeltaNet layer (0) cover every distinct shape, plus the output head
want = ("layers.0.", "layers.3.", "lm_head")
linears = {}
for m in walk(model):
    inner = getattr(m, "inner", None)
    if inner is not None and type(inner).__name__ == "LinearEXL3" and any(w in m.key for w in want):
        linears[m.key] = inner

def time_call(fn, iters):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters   # µs

# Keep the input buffers small but not cache-resident across layers: the 64 MB L2 (Infinity Cache)
# would otherwise flatter small matrices that a real token reads once
flush = torch.empty(256 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
print(f"{'layer':52s} {'K':>3s} {'in':>6s} {'out':>6s} {'MB':>6s} " +
      " ".join(f"{'m=' + str(r) + ' µs':>9s} {'GB/s':>5s} {'%pk':>4s}" for r in args.rows))
tot = {r: [0.0, 0.0] for r in args.rows}
for key, lin in sorted(linears.items(), key = lambda kv: kv[0]):
    mb = lin.trellis.numel() * lin.trellis.element_size() / 1e6
    cells = []
    for r in args.rows:
        x = torch.randn(1, r, lin.in_features, dtype = torch.half, device = "cuda") * 0.1
        def fn():
            flush.add_(1)   # evict L2 between calls, like a real token streaming 10 GB
            lin.bc.run_alloc(x, lin.out_features, False)
        t_flush = time_call(lambda: flush.add_(1), args.iters)
        us = max(time_call(fn, args.iters) - t_flush, 1e-3)
        gbs = mb * 1e6 / (us * 1e-6) / 1e9
        cells.append(f"{us:9.1f} {gbs:5.0f} {gbs / PEAK:4.0%}")
        tot[r][0] += us; tot[r][1] += mb
    print(f"{key.replace('model.language_model.', ''):52s} {str(lin.K):>3s} {lin.in_features:6d} {lin.out_features:6d} {mb:6.1f} " + " ".join(cells))
torch.cuda.synchronize(); time.sleep(1); sys.stdout.flush(); os._exit(0)
