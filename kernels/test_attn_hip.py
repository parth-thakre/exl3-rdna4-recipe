"""HIP Q4 split-decode attention (kernels/libattn_q4.so) vs the Triton path, on a synthetic 4-bit cache.
usage: kernels/test_attn_hip.py [depth_k] [q_len]   (setup/env.sh sourced, kernels/build.sh run first)"""
import ctypes, os, sys, time
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
import torch, triton
from exllamav3.modules.attention_fn import triton_paged as tp

here = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(here, "libattn_q4.so"))   # built by kernels/build.sh
lib.attn_q4_launch.restype = ctypes.c_int

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
bt = torch.randperm(pages, dtype = torch.int32, device = dev).view(1, pages)   # scrambled pages: exercises the table
seqlens = torch.tensor([depth], dtype = torch.int32, device = dev)
q = torch.randn(1, q_len, QH, HD, dtype = torch.half, device = dev)
kv_bytes = depth * (2 * KVH * HD * BITS // 8 + 2 * KVH * HD // 32 * 2)

os.environ.update({"EXL3_DEC_GROUP": "1", "EXL3_DEC_Q4W": "0"})
def triton_ref():
    return tp.paged_attn_triton_decode(q, None, None, kc, vc, bt, seqlens, causal = True, qc = (ks, vs, BITS, BITS),
                                       pre_appended_len = 0, n_kv_heads_override = KVH, num_warps = 4,
                                       num_stages = 1, block_n = 16)

group = QH // KVH
block_m = triton.next_power_of_2(q_len)
block_h = tp.decode_block_h(block_m, group)
block_rows = block_m * block_h
programs = KVH * triton.cdiv(group, block_h)
sm = torch.cuda.get_device_properties(0).multi_processor_count
mult = float(os.environ.get("HIP_SPLIT_MULT", "4"))
splits = max(2, min(int(mult * sm) // programs, triton.cdiv(depth, 16), 256))
split_len = triton.cdiv(triton.cdiv(depth + 0, splits), 32) * 32
partial_o = torch.empty(programs * splits * block_rows * HD, dtype = torch.float32, device = dev)
partial_ml = torch.empty(programs * splits * block_rows * 2, dtype = torch.float32, device = dev)
out = torch.empty_like(q)
h32 = tp._get_h32(q.device)
sinks = q
rows_sub, d_sub = tp.combine_subtiles(block_rows, HD)
stream = torch.cuda.current_stream().cuda_stream

def hip_run():
    rc = lib.attn_q4_launch(
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(kc.data_ptr()), ctypes.c_void_p(vc.data_ptr()),
        ctypes.c_void_p(bt.data_ptr()), ctypes.c_void_p(seqlens.data_ptr()),
        ctypes.c_void_p(partial_o.data_ptr()), ctypes.c_void_p(partial_ml.data_ptr()),
        ctypes.c_void_p(ks.data_ptr()), ctypes.c_void_p(vs.data_ptr()), ctypes.c_void_p(h32.data_ptr()),
        split_len, pages, splits, programs, q_len, 0, QH, KVH, PAGE, ctypes.c_float(1.0 / HD ** 0.5),
        block_m, block_h, 1, ctypes.c_void_p(stream))
    assert rc == 0, rc
    tp._paged_attn_decode_combine_kernel[(programs, (block_rows // rows_sub) * (HD // d_sub))](
        partial_o, partial_ml, out, h32, splits, sinks, BITS, False, q_len, QH, KVH, HD, HD, HD,
        block_m, block_h, block_rows, rows_sub, d_sub, num_warps = 4, num_stages = 1)
    return out

def timeit(fn, iters = 50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters

ref = triton_ref().float().clone()
got = hip_run().float().clone()
err = (got - ref).abs().max().item()
print(f"depth {depth} q_len {q_len}: splits {splits} x {split_len}, rows {block_rows}; "
      f"max err vs Triton {err:.2e} (output scale {ref.abs().max().item():.3f})")
t_tr = timeit(triton_ref)
os.environ["EXL3_DEC_Q4W"] = "1"
t_q4w = timeit(lambda: tp.paged_attn_triton_decode(q, None, None, kc, vc, bt, seqlens, causal = True,
               qc = (ks, vs, BITS, BITS), pre_appended_len = 0, n_kv_heads_override = KVH, num_warps = 4,
               num_stages = 2, block_n = 16))
t_hip = timeit(hip_run)
print(f"triton {t_tr:.1f} µs  q4w {t_q4w:.1f} µs  HIP {t_hip:.1f} µs ({kv_bytes / t_hip / 1e3:.0f} GB/s, floor {kv_bytes / 640e3:.0f} µs)")
torch.cuda.synchronize(); time.sleep(1); sys.stdout.flush(); os._exit(0)
