# Code reviews of the patches

Two review rounds by a second model (Sol, GPT-6.1 at high effort) on the exllamav3 changes. d6a4353 and 70ba2c2 are
local work commits, not upstream ones; file:line references point into those trees.

All six items of the first review are fixed in `patches/features/exllamav3-review-fixes.patch`. The five items of the
second review are fixed in the DeltaNet replay patch: abandoned caches hand their pending records back on eviction,
rewind validates every layer before consuming anything and marks failed commits, recording storage uses bounded
power-of-two row buckets, the test counts real captured-graph launches, and `batched_gdn_replay` checks head counts
before using them. Both are in the combined patch.

## Review of d6a4353 (RDNA4 patches), 2026-10-06

Static review plus CPU symbolic checks; no GPU runs. A second read-only WMMA audit agreed on 1-2.

1. Risk: the WMMA LDS exchanges (exl3_gemv_multirow_rdna.cu:654, 754, 763, 774) use bare
   `__builtin_amdgcn_wave_barrier()`, which LLVM marks IntrNoMem, so the compiler may reorder LDS
   accesses across it. Use `__syncwarp()` (fence/barrier/fence). No corruption observed.
2. Bug: `EXL3_GEMV_WMMA=1` selects the WMMA kernel on any arch (:1110, :1212); on gfx11 the body
   traps. Gate on runtime gfx12 + wave32.
3. Bug: eager decode (EXL3_BC_ATTN=0 or a BC decline) ignores EXL3_DEC_BLOCK_N/WARPS/STAGES
   (triton_paged.py:1458). Share the override resolution with BC.
4. Bug: EXL3_DEC_STAGES alone is ignored in bc_attn.py:79 (read only when WARPS is set).
5. Risk: EXL3_DEC_BLOCK_H=0 or a non-power-of-two value breaks the launch (triton_paged.py:1364). Validate.
6. Nit: inactive Q4W splits still do the query-selection dots before the inactive-split guard
   (triton_paged.py:1114 vs 1227).

Clean: row masking/clamp m1..8, mgemv indexing, LDS bounds (max 40,960 B), U/tail coverage over 1,152
K/warps/U combos, attention geometry q_len 1-16 x group 1-64, Q4W word indexing, graph re-instantiation.

## Review of 70ba2c2 (graph-captured GDN replay), 2026-10-06

The normal decode path is clean: reduction order, conv window, launch indexing over 64 jobs, graph param patching,
kernel args (2,064 / 1,048 B, under 4 KB). Off the normal path:
1. P2: a cache detached or deleted with unresolved records leaves GDNReplayBuffers.live > 0, so that shape stays eager forever.
2. P2: _pop_replay runs before validation and dispatch; a failure drops the only record, and a retry silently no-ops.
3. P2: per-layer recording buffers accumulate per (bsz, seqlen) shape: up to 4,896 rows/layer vs 128.
4. P3: the test counts configured buffers, not captured-graph launches.
5. P3: gdn.cu:2206 can hit `% 0` before validating head counts.
Sent back to the Opus author for fixes; a Sol re-review comes after.
