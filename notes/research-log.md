# Research log

The working notes this recipe came out of, lightly edited: machine-specific serving details are removed and paths
point at this repo's layout. Dates are 2026. Numbers are from one RX 9070 XT (16 GB) on Fedora 44. Names like
`wt-replay` were local work branches. All of this was on exllamav3 f1cf869 and TabbyAPI f07131c (with a since
superseded TabbyAPI patch); the work has since been ported to exllamav3 0662fac as the commits in `patches/features/`.
The WMMA GEMV, graph re-instantiation, GQA decode attention with Q4W, the review fixes and the DeltaNet replay are in
`patches/exllamav3-rdna4.patch`. The 16-row GEMV, adaptive draft length, tree fallback and Q4P experiments below are
not in this repo.

Some configuration labels in these notes are loose (for example, the "before/after attention patch" rows mix 64k and
80k servers, and the MTP row mixes 128k, 144k and 160k servers). The tables in the main README were rechecked against
the raw benchmark logs and give the exact configuration and prompt length of every number; where the two disagree,
the README is right.

## Recommended request settings

`"reasoning_effort": "medium"` plus this system prompt. On GPQA it stopped the thinking spirals (0 cut-offs vs 7)
and was 4.6x faster at equal or better accuracy:

> Think efficiently. Work through the problem once, carefully. If you notice yourself re-deriving the same result,
> second-guessing a conclusion you already checked, or cycling between options, stop and commit to the
> best-supported answer. A confident best answer beats an unfinished analysis.

The default `xhigh` effort tells it to "consider plausible alternatives", which is what makes it spiral.

## Measured on this card (2026-10-05)

| Setup | Decode | Notes |
|---|---|---|
| fast: DFlash2, all patches | **153.7 tok/s** code (greedy), ~58 prose | was 127 before the patches; 64k ctx, 15.9 GB peak |
| long: MTP, WMMA patch | 90 tok/s code | was 82; 128k ctx tested (161k fits but 16.28/16.3 GB) |
| no speculation | ~35-38 tok/s | |
| prefill | ~1,250 tok/s at 8-30k, ~900 at 129k, ~680 at 258k | |

GPQA Diamond, first 50 questions, 30k token cap, temperature 1.0:

| | correct | cut off |
|---|---|---|
| 3.0 bpw, `xhigh` (45 q) | 82.2% | 7 |
| **3.0 bpw, `medium` + anti-spiral** | **86.0%** | 0 |
| 2.5 bpw, `medium` + anti-spiral | 68.0% | 0 |

2.5 bpw lost 10 questions to 3.0's 1 on the same set (significant), so stay on 3.0. For long context use the MTP profile.
Published reference (ISTA, 198 q): BF16 89.9, UD-IQ3_S 89.9, UD-Q2_K_XL 86.9. Perplexity (llama.cpp method, no BOS):
EXL3 3.0 bpw 7.063, 2.5 bpw 7.248, UD-Q3_K_XL 7.068.

## Long context (2026-10-06)

Decode speed by depth (context_bench.py: a fresh document with a code buried mid-way, then ~450 tokens of Python;
every run recalled the code):

| Profile | 8k | 32k | 48k | 60-64k | 76k | 96k | 118-125k | 134k | 150-160k |
|---|---|---|---|---|---|---|---|---|---|
| fast before attention patch (7-token draft) | 128 | 109 | 96 | 89 | 80 | | | | |
| fast after (7-token draft) | 122 | 129 | 129 | 120 | 100 | | | | |
| fast, shipped (5-token draft, 80k) | 118 | | | | 92 | | | | |
| MTP 128k / 144k / 160k | 82 | 83 | | 75 | | 68 | 62 | 57 | 59 |

Prefill runs at ~1,250 tok/s up to 32k, ~1,035 at 80k, ~830 at 160k (about 3 min for a full 160k window).
What didn't pay off: `EXL3_QC_STAGING=0` (the load then fails), 2-bit draft KV (-8% speed), K4/V3 main KV (slower,
no headroom gained), a 4-token draft (-30% at 60k: fewer tokens per step means more full KV reads).
VRAM accounting (deep-dive): KV is 24,192 B/token incl. scales and the draft cache (not 16 KB); DeltaNet state is
0.155 GB + 0.152 GB per draft token; `autosplit_reserve` frees nothing resident.

### Update (later 2026-10-06)

- **DFlash2 draft requantized to 3.0 bpw** from incoai/Qwen3.8-27B-DFlash2 bf16 (`setup/requant_draft.sh`): 0.89 GB
  vs 1.4 GB on disk, same speed (154.3 vs 153.8 tok/s). The freed memory brings back the full 7-token draft and lets the
  fast profile go to 96k: 144 tok/s at 8k, 123 at 60k, 110 at 92k; 150 code / 60 prose through the API; 16.1 GB peak.
- **Q4W attention kernel** (`EXL3_DEC_Q4W=1`, generated into `triton_paged.py`): loads 4-bit words once and does 8 word-sliced dots.
  1.2x at q_len 8, 1.3x at q_len 4, even at q_len 1. Still LDS-shuffle bound: an fp16 cache runs the same kernel at
  582 GB/s (91% of peak) while the 4-bit one manages ~140 GB/s. The rest needs a hand-written HIP/WMMA kernel.
- Tried and dropped: a "Q4P" permuted-gather unpack (2.3x slower).
- TensorFold (research, SHA 609ca419): no working AMD path (PR144: NaN logits on gfx1201). Ideas worth porting: replaying
  only accepted DeltaNet updates instead of keeping a recurrent-state snapshot per draft token (~0.15 GB/token), and
  tree verification (limited here by the 8-row GEMV).

### TensorFold ports (2026-10-06)

- **DeltaNet accepted-input replay** (`wt-replay`, `EXL3_GDN_REPLAY=1`): the verify forward no longer commits
  recurrent state or keeps a snapshot per draft token. It records the small post-projection inputs, and on
  acceptance replays only the accepted prefix. Exact: the state check is bit-identical for every accepted length
  (372 comparisons, scalar and channelwise decay, bsz 1/2), and 400-token greedy output is identical. **-1.1 GB VRAM.**
  The first version ran DeltaNet eagerly: 85.4 vs 89.0 tok/s on the graph path (-4%). The graph-captured version
  (`wt-replay-bc`, Opus-written, Sol-reviewed) is **92.4 vs 89.3 tok/s (+3.4%)**, with the same -1.06 GB. It is faster
  because the verify no longer writes 8 state snapshots per layer per step. Greedy output is identical, the kernel
  checks are exact (712 comparisons), and all 48 layers take the graph path.
  - **Now the default** (`serve/run_tabby.sh`, `EXL3_GDN_REPLAY=1`). Through the server, graph-replay build:
    149 tok/s at 8k, 122 at 64k, **97.6 at 120k** (the old 96k default: 136 / 121 / 113 at 90k), 16.13 GB peak.
    Review round 2 (Sol) fixed abandoned-cache eviction, two-phase rewind with failure marking, bounded recording
    storage, and a test that counts real graph launches and checks interleaved caches with distinct inputs (exact).
    `patches/features/0004-Gated-DeltaNet-accepted-input-replay-for-speculative.patch`.
  - First (eager) version: DFlash2 at **128k** (was MTP-only past 96k). 132 tok/s at 8k, 112 at 64k, **96.5 at
    118k** (MTP: 62), recall OK at every depth, 16.07 GB peak, 911 tok/s prefill at 125k.
- **16-row WMMA GEMV** (`wt-tree`): rows 9-16 pass against reconstruct+hgemm. lm_head at 16 rows: 1.85 ms vs 7.5 ms.
  Not used yet: the DFlash2 block is 8.
- **Adaptive draft length** (`EXL3_ADAPTIVE_DRAFT=1`): neutral (121.3 vs 121.4 code, 59.2 vs 58.7 prose).
- **Sequential tree fallback** (`EXL3_DRAFT_TREE=1`): slower (118.7 code, 54.2 prose), so it isn't used. A real
  one-pass tree needs tree-masked attention and branched DeltaNet state.
- Every speculative mode, plain DFlash2 included, splits from non-speculative greedy at the same token (375 of 400).
  That is the multi-row verify's accumulation order, not the new code.
- Sol review of the RDNA4 patches: `notes/reviews.md`. Fixes (now folded into the commits in `patches/features/`):
  WMMA LDS fences (`__syncwarp`), gfx12-only WMMA gating, shared EXL3_DEC_* overrides for eager and graph attention,
  Q4W early exit for idle splits (q_len 1: 369 -> 350 µs/layer at 60k). Greedy output identical, speed unchanged.

## Patches (as they were against upstream f1cf869; now ported, see `patches/README.md`)

1. **`exllamav3-gfx12-wmma-multirow-gemv.patch`**: an RDNA4-native GEMV for 2-8 rows (draft-token verification),
   behind `EXL3_GEMV_WMMA=1` (on in `setup/env.sh`). The trellis is decoded once per tile with the tuned RDNA decoder, then
   staged through a per-wave LDS slot so one `v_wmma_f32_16x16x16_f16` takes all rows (layout probed on the card:
   `kernels/wmma_layout_probe.hip`). It brings an 8-row verify to ~1.16x a 1-row decode (was ~1.55x). Full-model check:
   top-1 agreement 100%, KL ~3e-5. It also includes the multi-row over-read fix, which turboderp independently
   committed upstream as cc50bbb.
2. **`exllamav3-rocm-graph-reinstantiate.patch`**: `hipGraphExecKernelNodeSetParams` exhausts a fixed HIP
   kernel-argument pool and segfaults after ~1.6M updates, about 1 h of serving (standalone repro:
   `kernels/graph_setparams_repro.hip`). `Graph::launch` now re-instantiates every 100k updates (`EXL3_GRAPH_REINST`), so HIP
   graphs can stay on (+6%) without the crash. Output is byte-identical even when it re-instantiates every 1,000 updates.
3. **`exllamav3-rdna4-decode-attention-gqa.patch`**: the split-decode attention gave each program 16 query rows, so a
   multi-token verify (q_len 8) split Qwen's 6-head GQA groups over 3 programs, each re-reading and re-dequantizing the
   same Q4 K/V. `EXL3_DEC_GROUP=1` takes the whole group per program, with tuned warps/stages/tile/split counts
   (`EXL3_DEC_*`, set in `setup/env.sh`). At 60k deep, one layer goes 1961 -> 605 µs (q_len 8), 1336 -> 450 (q_len 4) and
   550 -> 336 (q_len 1), with max error 2e-6 (`bench/bench_decode_attn.py`). The C++ graph path recomputed the program count
   with the old rule, which made the GPU fault; `configure_slot` now takes `block_h` from Python.
4. **`tabbyapi-allow-rdna3-rdna4.patch`**: let TabbyAPI f07131c accept gfx12. Not needed from TabbyAPI 2fd6cc7 on, so it
   isn't in this repo any more.

Build: `setup/build_exllamav3.sh`. When editing a .cu file, delete its generated `.hip` before rebuilding. Missing
Fedora header packages are unpacked into `deps/` (no sudo).

## Where the time goes (8-token verify forward)

GPU kernels ~32 ms (EXL3 GEMVs ~25 ms, DeltaNet recurrent 2 ms, small fp16 projections 2.6 ms), wall ~40 ms before
graphs were re-enabled. Per layer type at 1 row, EXL3 GEMVs reach 53-83% of the 640 GB/s bandwidth; the 2 MB k/v
projections only ~25%. Tools: `bench/bench_gemv.py`, `bench/test_wmma_gemv.py`, `bench/test_wmma_model.py`, plus profiling scripts and ISA
counting that aren't in this repo.

## Next steps (not started)

**More context in 16 GB.** VRAM in the fast profile: main weights 10.25 GB (lm_head alone 0.95 GB at 6 bpw), DFlash2
draft 1.67 GB, KV 64k Q4 ~1 GB (16 KB/token; superseded, the measured figure is 24,192 B/token, see above), state/scratch/reserve/runtime ~2.4 GB. The embedding table and vision
tower are already in system RAM. Ideas, best value first:
1. Re-quantize the DFlash2 draft to ~3 bpw: ~0.6 GB, about +40k context. Needs the bf16 draft (z-lab).
2. turboderp's `SC_3.00bpw_H4` branch (4-bit head, self-calibrated): ~0.3 GB. 13 GB download, then GPQA/PPL check.
3. Q3 KV cache: KV memory -25%. Check with `bench/needle_test.py`.
4. Smaller `chunk_size` / `autosplit_reserve` after measuring the real prefill peak: ~0.2-0.4 GB.
Together roughly 64k -> 140-160k context at full DFlash2 speed (estimate).

**More speed.**
- Launch/host overhead: ~1,250-1,450 kernels per token. Fuse qkv/z and gate/up launches, and the Hadamard into the GEMV
  (Argos1111/orcasaq2-rocm did this).
- 1-row GEMV efficiency on small shapes (k/v at ~25% of bandwidth).
- int8 WMMA for verify (buun-llama-cpp's approach).
- DFlash2 draft length tuning.

**Other open items.** Autostart on boot for TabbyAPI and Open WebUI; `medium` + anti-spiral as the Open WebUI
default; a second GPQA pass with a 64k cap; a KLD head-to-head vs UD-Q3_K_XL.
