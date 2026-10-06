"""Split-decode paged attention on a synthetic Q4 cache (Qwen3.8: 24 q heads, 4 kv heads, head_dim 256).
Times one layer's call at a given depth and q_len, for several launch configs; checks each output against the
default config. Per-forward cost = 16x this (16 full-attention layers).
usage: bench_decode_attn.py [depth_k] [q_len]"""
import itertools, os, sys, time
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode

depth = int(float(sys.argv[1]) * 1024) if len(sys.argv) > 1 else 60 * 1024
q_len = int(sys.argv[2]) if len(sys.argv) > 2 else 8
QH, KVH, HD, BITS, PAGE = 24, 4, 256, 4, 256
pages = (depth + q_len + PAGE - 1) // PAGE + 1
dev = "cuda"
torch.manual_seed(0)
kc = torch.randint(-2**31, 2**31 - 1, (pages, PAGE, KVH * HD // 32 * BITS), dtype = torch.int32, device = dev)
vc = torch.randint(-2**31, 2**31 - 1, (pages, PAGE, KVH * HD // 32 * BITS), dtype = torch.int32, device = dev)
ks = (torch.rand(pages, PAGE, KVH * HD // 32, device = dev) * 0.5 + 0.1).half()
vs = (torch.rand(pages, PAGE, KVH * HD // 32, device = dev) * 0.5 + 0.1).half()
bt = torch.arange(pages, dtype = torch.int32, device = dev).view(1, pages)
seqlens = torch.tensor([depth], dtype = torch.int32, device = dev)
q = torch.randn(1, q_len, QH, HD, dtype = torch.half, device = dev)
kv_bytes = depth * (2 * KVH * HD * BITS // 8 + 2 * KVH * HD // 32 * 2)

FP16 = os.environ.get("BENCH_FP16") == "1"
if FP16:
    kc = torch.randn(pages, PAGE, KVH, HD, dtype = torch.half, device = dev) * 0.1
    vc = torch.randn(pages, PAGE, KVH, HD, dtype = torch.half, device = dev) * 0.1
    kv_bytes = depth * 2 * KVH * HD * 2

def run(warps, stages, block_n):
    if FP16:
        return paged_attn_triton_decode(q, None, None, kc, vc, bt, seqlens, causal = True, num_warps = warps,
                                        num_stages = stages, block_n = block_n)
    return paged_attn_triton_decode(q, None, None, kc, vc, bt, seqlens, causal = True, qc = (ks, vs, BITS, BITS),
                                    pre_appended_len = 0, n_kv_heads_override = KVH, num_warps = warps,
                                    num_stages = stages, block_n = block_n)

def timeit(fn, iters = 50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters

KEYS = ("EXL3_DEC_GROUP", "EXL3_DEC_BLOCK_H", "EXL3_DEC_SPLIT_MULT", "EXL3_DEC_Q4P", "EXL3_DEC_Q4W")
def setenv(env):
    for k in KEYS: os.environ.pop(k, None)
    os.environ.update(env)

setenv({}); ref = run(8, 1, 32).float().clone()
print(f"depth {depth}, q_len {q_len}, KV bytes per layer {kv_bytes / 1e6:.1f} MB (floor at 640 GB/s: {kv_bytes / 640e3:.0f} µs)")
print(f"{'config':44s} {'µs/layer':>9s} {'GB/s':>6s} {'max err':>8s}")
configs = []
GRID = os.environ.get("BENCH_GRID", "full")
for q4p, grp, mult, warps, stages, bn in itertools.product(("0", "1"), ("0", "1"), ("2", "4", "8"), (4, 8), (1, 2), (16, 32, 64)):
    if GRID == "q4p" and grp == "0" and not (q4p == "0" and mult == "2" and warps == 8 and stages == 1 and bn == 32): continue
    configs.append(({"EXL3_DEC_Q4W": q4p, "EXL3_DEC_Q4P": "0", "EXL3_DEC_GROUP": grp, "EXL3_DEC_SPLIT_MULT": mult}, warps, stages, bn))
rows = []
for env, warps, stages, bn in configs:
    setenv(env)
    try:
        out = run(warps, stages, bn).float()
        us = timeit(lambda: run(warps, stages, bn))
    except Exception as e:
        if os.environ.get("BENCH_DEBUG"): print(env, warps, stages, bn, str(e)[:300])
        continue
    err = (out - ref).abs().max().item()
    name = f"q4w={env['EXL3_DEC_Q4W']} group={env['EXL3_DEC_GROUP']} mult={env['EXL3_DEC_SPLIT_MULT']} w={warps} s={stages} bn={bn}"
    rows.append((us, name, err))
top = sorted(rows)[:8] + sorted(r for r in rows if r[1].startswith("q4w=1"))[:6]
for us, name, err in top + [r for r in rows if r[1] in ("q4w=0 group=0 mult=2 w=8 s=1 bn=32", "q4w=0 group=1 mult=4 w=4 s=1 bn=16")]:
    print(f"{name:44s} {us:9.1f} {kv_bytes / us / 1e3:6.0f} {err:8.1e}")
setenv({})
torch.cuda.synchronize(); time.sleep(1); sys.stdout.flush(); os._exit(0)
