# Research log

The working notes this recipe came out of, lightly edited: machine-specific serving details are removed and paths
point at this repo's layout. Dates are 2026. Numbers are from one RX 9070 XT (16 GB) on Fedora 44. Names like
`wt-replay` were local work branches. Most of this was on exllamav3 f1cf869 and TabbyAPI f07131c (with a since
superseded TabbyAPI patch); the work was then ported to exllamav3 0662fac (`patches/previous/`: the WMMA GEMV, graph
re-instantiation, GQA decode attention with Q4W, the review fixes, the DeltaNet replay and the batching series), and
on 2026-10-12 to 9cc3bd1, where upstream's own replay and graphs-off default replace ours, and on 2026-10-13 to
0f685aa, which has our DeltaNet scan change, and on 2026-10-27 to the v1.6.5 release (`patches/exllamav3-rdna4.patch`
and `patches/features/`); the last four sections cover those ports. The 16-row GEMV below later grew into the
batching series (16- and 32-row GEMV); see `notes/measurements.md`. The adaptive draft length, tree fallback and Q4P
experiments are not in this repo. The notes between the 16-row GEMV and the ports skip the batching work, which is
written up in `notes/measurements.md`.

Some configuration labels in these notes are loose (for example, the "before/after attention patch" rows mix 64k and
80k servers, and the MTP row mixes 128k, 144k and 160k servers). The tables in `notes/measurements.md` were rechecked against
the raw benchmark logs and give the exact configuration and prompt length of every number; where the two disagree,
`notes/measurements.md` is right.

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
   Later (2026-10-08) graphs went off entirely, following upstream; the +6% was the C++ path vs the Python one, and the C++
   path still runs without graphs (`notes/measurements.md`).
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
2. Done 2026-10-09: turboderp's `SC_3.00bpw_H4` branch (4-bit head, self-calibrated) is the default now. It saved
   244 MB, with a GPQA score within noise, slightly lower perplexity and about 3% slower greedy code
   (`notes/measurements.md`).
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

## Upstream dev of 2026-10-10 (tested, not adopted)

Our patch set rebased onto ExLlamaV3 `dev` 77ddae2, which now has upstream's own versions of two of our changes:
DeltaNet layers re-advance from staged inputs instead of keeping per-draft-token state history (3a8e2fb, the job our
replay patch did), and the split-decode attention packs (query, head) rows densely so a verify step reads K/V once per
32 rows (dd39a01). We kept the WMMA GEMV and batching series (cherry-picked cleanly) and dropped our replay and the
opt-in `EXL3_DRAFT_ROW_BUDGET` (it conflicted with upstream's new hybrid draft mode and was slower anyway).

- Upstream's attention layout alone is slow on gfx1201 at long context: decode 78.7 tok/s at 64k and 74.2 at 120k,
  against 127.7 and 109.0 on our build. With our `EXL3_DEC_*` tuning and Q4W kernel ported onto the dense layout
  (whole kv head per program up to 128 rows, the higher split target), one layer at 60k context takes 499 us for an
  8-token verify and 361 us for one token, against 1,553 and 557 for upstream's default and 503 / 360 on our build.
- Through the API, both builds on the SC H4 quant and the same config, two interleaved rounds, rebased vs ours: code
  144.9 vs 144.6 tok/s (averages of 6), prose 59.7 vs 60.3, 120k 102.5 vs 101.2 (two runs each), 3 at once
  (generator) 234 vs 237, perplexity 7.0487 vs 7.0488, VRAM after the 120k run 16,095 / 16,010 MB vs 16,173 /
  16,008. 64k is inconclusive: 102.0 / 127.4 (and 110.1 in an earlier run) vs 118.5 / 128.0. The greedy outputs
  diverged on 3 of 4 prompts, each time where the top two logits were 0.094, 0.094 and 0.172 apart.
- Upstream's fde1b37 passes `staging_pages` to `paged_attn_triton_decode`, which doesn't take it, so the eager Q4-cache
  decode raises TypeError (with `EXL3_BC_ATTN=0`, or whenever the graph-captured path declines). Found in review by
  GPT-6.1 Sol; fixed in our tree.

Short-context speed and quality come out even and 64k is unsettled, so the recipe stays on 0662fac until the rebased
build has had a long soak under load and more 64k runs.

## Upstream dev of 2026-10-12 (adopted; the base until 0f685aa)

The same port moved to `dev` 9cc3bd1 (v1.6.1 plus the next day's commits). Upstream fixed the `staging_pages` bug itself
(3a513ac, the same one-line change), so we dropped ours; everything else cherry-picked cleanly.

- A first A/B against the 0662fac build looked like a prose regression, and three interleaved rounds of three builds
  (0662fac, the 10 October port, this one) confirmed it: prose 64.4 vs 59.1 and 58.7 tok/s, code 146.7 vs 145.8 and
  145.5. Both ports had it, so it came with upstream's changes up to 10 October, not with the newer commits.
- Draft acceptance in the server logs ruled out sampling: on greedy code the port accepted more drafts (77% vs 73%)
  and was still slower, about 45 vs 42.5 ms per speculative step. There is no ROCm profiler on this machine, so we
  timed the phases of a step with synchronizations around each (`notes/measurements.md`, "Current base"): the whole
  difference was in the target model's verify forward, and in it the DeltaNet layers (11.9 -> 15.0 ms per step).
- Cause: upstream's re-advance (3a8e2fb) runs the scan in place on a scratch row, so every token reads the state twice
  and writes it once in memory; our replay had kept it in registers. Keeping each thread's slice in registers across
  the pass (one load, one store) brought it back to 12.1 ms with bit-identical output, and made upstream's replay
  faster too (0.84 -> 0.58 ms in the same per-module timing). A/B after that, three rounds: code 155.0 vs 146.3 tok/s, prose 62.6 vs 63.5 (noise),
  120k 106.6 vs 103.9. That made the port the default.
- Review (GPT-6.1 Sol) of the scan change found a pre-existing bug in upstream's Mamba2 replay launch (a v split with no
  matching kernel; blocks past the head's range updated the neighbouring heads' state), fixed in `0018`. Not tested on a
  Mamba2 model (we have none).
- The attention layout test failed 4 cases with our tuning on: `EXL3_DEC_STAGES=2` with the Q4W kernel needs 67.5 KB
  of LDS at the 128-row tile (16 q heads per kv head), over gfx1201's 64 KB. Qwen3.8 never builds that tile. Now it
  falls back to one stage (`0014`); Sol's review then found that the launch pick ignored the single-split variant's
  larger footprint (`0015`).
- Upstream's e2e `test_recurrent_spec_rewind.py` on Qwen3.5-9B (2.0 bpw) returned NaN for every case in 5 of 33 runs,
  with and without our scan change, and in none of the 16 runs after the last failure. Poisoning the KV cache and the
  recurrent pools with NaN didn't reproduce it. Update 2026-10-13: the same test fails intermittently with PyTorch's
  expandable segments on and not with them off (see "Upstream dev of 2026-10-13"), which makes them the likely cause.
- Soak and quality: GPQA Diamond (198, `medium` + anti-spiral, 3 at a time) through the live server on the current
  base, 92 minutes: 158/198, against 160 on the previous base (147 correct in both, 11 and 13 the other way), the same characters per
  second per request, VRAM flat after the first 10 minutes, no server errors.
- The MTP profiles work on the current base with upstream's replay on: `long` 92.3 / 84.0 / 66.0 tok/s at 8.1k /
  67.6k / 127.8k (peak 14.09 GB), `xl` 68.7 / 63.6 at 127.8k / 160.0k (peak 14.83 GB, was 16.24 of 16.3). Three slots
  fit in `long` (14.70 GB at 127.8k); concurrent MTP requests are untested, so both stay at one.
- PR #423 isn't ported: upstream rewrote `sample_from_state` for tensor parallelism, and the PR relied on our row
  pruning. It stays available on the previous base.

## Upstream dev of 2026-10-13 (adopted; the base until v1.6.5)

We filed the DeltaNet scan change and the Mamba2 replay launch fix as issues rather than PRs (#505, #506), with the
measurements and the patch linked. turboderp merged both into `dev` the same day (0f685aa), with the same kernel code;
only comments differ (upstream kept the old operand comments and has no note on the Mamba2 launch). So the port moved to 0f685aa and dropped `0016` to `0018`;
the other 15 commits cherry-picked cleanly.

- Upstream's other changes since 9cc3bd1 are the disk cache tier (write-behind during long prefills, an idle flush in
  steps), tensor-parallel DFlash2 top-k, TP fixes, the DSA top-k workspace and conversion wildcards. TabbyAPI doesn't
  configure the disk tier and TP isn't used; the compute kernels this setup runs are unchanged, though some shared
  Python code changed (generator and page-table bookkeeping, a refactor of the DFlash2 selector).
- Checks against the 9cc3bd1 build (`notes/measurements.md`, "Current base"): greedy output identical on all four test
  prompts, PPL 7.0487 on both, kernel tests pass (including upstream's new Mamba2 replay test), and the bs1/bs3
  near-tie at token 91 of one prompt is the same as before. The API A/B was even on code and prose; that night's
  long-context runs spread too widely (96-124 tok/s at 64k for one build) to say more than that.
- Soak and quality: GPQA Diamond (198, `medium` + anti-spiral, 3 at a time) through the live server on 0f685aa:
  164/198 (158 on 9cc3bd1, 160 on the previous base), within the run-to-run spread. No server errors, VRAM flat.
- Full GPQA at `xhigh` (60k cap, 2 at a time): 175/198 (88.4%) against 164 for `medium` on the same build. It got 18 of
  the 34 `medium` missed and missed 7 `medium` got; 6 answers hit the 60k cap, all wrong (`notes/measurements.md`).
- Container image (`container/`): a build stage runs the recipe's setup scripts on AMD's ROCm 7.2.4 Ubuntu image, the
  runtime stage is plain Ubuntu 24.04 with Python 3.12, gcc and the Python headers, no ROCm: the extension's ROCm
  libraries resolve to the PyTorch wheel's bundled copies, which carry the same sonames (`libamdhip64.so.7`,
  `libhipblas.so.3`, ...). 16.5 GB. Tested on Parth's card with the models mounted read-only: greedy output identical
  to the host build (4 x 400 tokens), 153 / 61 tok/s code / prose through the API (host: 156 / 61-67), and the registry
  gates passed (106k-token context recall included). `container/run.sh` picks Docker or Podman, downloads the models
  with the image's own `hf` and serves on 127.0.0.1; the draft comes from Hugging Face now
  (parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw, byte-identical to the requant we run), so nothing needs the GPU before
  serving. `triage.sh` collects the details and opens Claude Code, Codex or OpenCode (a free model) when something fails.
- `max_seq_len` equal to `cache_size` (both 131072) refuses the one request that fills the window: a 131,008-token
  prompt with `max_tokens: 64` failed with "Job requires 513 pages (only 512 available)" (ExLlamaV3 counts the cache in
  256-token pages and needs one more than the window). The configs now use `max_seq_len: 130816` with
  `cache_size: 131072`; the MTP profiles likewise. Found through the check in sybil-solutions/local-ai-images' build
  workflow, then reproduced on the live server.
- Other distros. The setup scripts assumed Fedora: ROCm under `/usr`, ROCm's clang in `/usr/lib64/rocm/llvm/bin`,
  headers fetched with `dnf`. `setup/rocm.sh` now finds ROCm through `hipconfig` (`/opt/rocm` on AMD's packages and
  Arch), and `setup/fetch_deps.sh` checks the headers and prints the missing packages for Fedora, Debian/Ubuntu or Arch
  (on Fedora it still fetches them without sudo). Tested the README install end to end in podman containers on the same
  GPU: Ubuntu 24.04 (`rocm/dev-ubuntu-24.04:7.2.4`, AMD's apt packages) and Arch (ROCm 7.2.4 from the Arch repos,
  Python 3.12 from uv). Both build the extension against ROCm 7.2.4, pass the 32-row GEMV test and give greedy output
  identical to the Fedora build (4 prompts x 400 tokens). Two fixes came out of it: the header check was missing
  `rocrand` (hipRAND's headers include it; Fedora's base ROCm packages have it, AMD's don't), and the Fedora unpack
  loop died partway with pipefail (`rpm2cpio | cpio`, SIGPIPE), which the old script had too.
- The recurrent kernel tests were flaky when the whole directory ran, on every build and distro we tried: 0 to 44 of
  884 failed per run, mostly guard-band checks around outputs overwritten with zeros, and once a `.sum()` that came back
  as an impossible count. The allocator reproducers and an A/B on the 0f685aa build strongly implicate PyTorch's
  expandable segments, which our `env.sh` had turned on. On gfx1201, memory
  mapped through HIP's virtual memory API can lose writes after a remap at the same address (ROCm/TheRock#8744,
  pytorch/pytorch#195202). A torch loop (empty the cache, allocate, fill, check) lost data in 65 of 300 rounds with them
  on and none off; a HIP-only loop (final version) lost data in 19-99% of rounds on ROCm 7.1.1 and 7.2.4, never with `hipMalloc`. A/B on
  the 0f685aa build: recurrent tests failed in 3 of 10 runs on, 0 of 10 off; upstream's spec-rewind e2e test (Qwen3.5-9B)
  failed 7 of its 8 cases in 3 of 45 runs on, none in 45 off (whether those failures were NaN wasn't checked; see the
  open NaN noted earlier). Off, the 106k-token prefill after concurrent and image load still passes at 3 slots. After
  switching, code measured 156.6 / 156.4 tok/s (about 152 before) and prose 60.7 / 67.2 (about 60). Expandable segments
  are now off in `setup/env.sh` and on the live server.


## TabbyAPI main of 2026-10-12 (adopted)

TabbyAPI 2fd6cc7 to a3fed43 (20 commits): a fix that names the assistant role in the first streamed delta (#501;
LangChain-style clients dropped streamed tool calls without it), upstream ROCm packaging and docs, a built-in chat page
at `/ui`, nested model folders, hybrid drafting and disk-cache options (the latter need ExLlamaV3 1.6.2). No new Python
dependencies, and nothing that touches the allocator setting (`env.sh` still sets it). Same ExLlamaV3 build and live
config (128k, 3 slots), 2026-10-13, A/B through the API: code 156.1 vs 155.1 tok/s, prose 60.4 vs 60.2, prefill 1,219 /
1,263 vs 1,242 / 1,262 tok/s at 7.7k / 29.4k, peak 16.25 vs 16.27 GB under concurrent and image load. The first
streamed delta of a tool call now carries `role: assistant` (it had none). After concurrent and image requests, the
registry's six gates pass on a3fed43: context recall at 106,084 tokens, 107.3 tok/s, peak 16,282 MiB, no OOM.

## Upstream v1.6.5 (adopted 2026-10-27: the current base)

v1.6.5 (2026-10-25) is 115 commits past 0f685aa. None of them carry our speed work, but it has a fix for a 1.6.4
tokenizer regression that sometimes broke prompt caching, less VRAM for prefill with a quantized cache, and lower-latency
drafting. 14 of our 15 commits cherry-picked cleanly; `0004` needed a hand edit because turboderp had made the same
buffer change to `BC_GatedMLP` (`patches/README.md`).

- With every patch switch at stock the port's greedy output is token-identical to plain v1.6.5, so the port adds nothing
  beyond the switchable kernels. With them on it differs from plain v1.6.5 at near-ties (1 of 4 prompts), as the
  0f685aa build did from plain 0f685aa (2 of 4).
- PPL moved from 7.0487 to 7.0579, the same on plain v1.6.5, so it comes with upstream.
- Speed against the 0f685aa build: code about 1% slower (154 vs 156 tok/s), decode at 8-56k equal or faster, prefill
  even, 0.15 GB less peak VRAM. The earlier A/B of the 0f685aa build against stock v1.6.4 (official ROCm wheel,
  gfx12-generic) had the patch 7% ahead on code and ~30% past 30k.
- Not run on this base yet: GPQA through the live server.

