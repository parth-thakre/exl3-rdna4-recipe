"""EXL3_GEMV_WMMA=1 vs the fdot2 multi-row GEMV: max error relative to the output scale, and µs per call.
usage: test_wmma_gemv.py"""
import os, sys, time
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch
from common import MODEL_DIR
from exllamav3 import Config, Model

config = Config.from_directory(MODEL_DIR)
model = Model.from_config(config)
model.load()

def walk(m):
    yield m
    for c in getattr(m, "modules", []) or []:
        yield from walk(c)

SHAPES = ["layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.in_proj_z", "layers.0.linear_attn.out_proj",
          "layers.0.mlp.gate_proj", "layers.0.mlp.down_proj", "layers.3.self_attn.k_proj",
          "layers.3.self_attn.q_proj", "lm_head"]
lin = {}
for m in walk(model):
    inner = getattr(m, "inner", None)
    if inner is not None and type(inner).__name__ == "LinearEXL3":
        for s in SHAPES:
            if m.key.endswith(s):
                lin[s] = inner

flush = torch.empty(256 * 1024 * 1024, dtype = torch.uint8, device = "cuda")
def time_call(fn, iters = 150):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters
t_flush = time_call(lambda: flush.add_(1))

torch.manual_seed(0)
US = [int(u) for u in os.environ.get("TEST_U", "1,2,4").split(",")]
print(f"{'shape':34s} {'m':>2s} {'fdot2 µs':>9s} " + " ".join(f"{'U=' + str(u) + ' µs':>8s}" for u in US) + f" {'best':>6s} {'rel err':>8s} {'m=1 µs':>7s}")
worst = 0.0
for s in SHAPES:
    l = lin[s]
    os.environ.pop("EXL3_GEMV_WMMA", None)
    x1 = torch.randn(1, 1, l.in_features, dtype = torch.half, device = "cuda")
    t1 = time_call(lambda: (flush.add_(1), l.bc.run_alloc(x1, l.out_features, False))) - t_flush
    for rows in [int(r) for r in os.environ.get("TEST_ROWS", "2,4,8").split(",")]:
        x = torch.randn(1, rows, l.in_features, dtype = torch.half, device = "cuda")
        os.environ.pop("EXL3_GEMV_WMMA", None)
        ref = l.bc.run_alloc(x, l.out_features, True).float()
        t_ref = time_call(lambda: (flush.add_(1), l.bc.run_alloc(x, l.out_features, False))) - t_flush
        os.environ["EXL3_GEMV_WMMA"] = "1"
        ts, err = [], 0.0
        for u in US:
            os.environ["EXL3_GEMV_WMMA_U"] = str(u)
            out = l.bc.run_alloc(x, l.out_features, True).float()
            err = max(err, ((out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)).item())
            ts.append(time_call(lambda: (flush.add_(1), l.bc.run_alloc(x, l.out_features, False))) - t_flush)
        worst = max(worst, err)
        print(f"{s:34s} {rows:2d} {t_ref:9.1f} " + " ".join(f"{t:8.1f}" for t in ts) + f" {t_ref / min(ts):5.2f}x {err:8.1e} {t1:7.1f}")
os.environ.pop("EXL3_GEMV_WMMA", None); os.environ.pop("EXL3_GEMV_WMMA_U", None)
print(f"worst relative error {worst:.1e}")
torch.cuda.synchronize(); time.sleep(1); sys.stdout.flush(); os._exit(0)
