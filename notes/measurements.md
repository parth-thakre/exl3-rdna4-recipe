# Measurements and details

The detail behind the main README: how each profile and patch performs, how it was measured, what didn't work, and
the known limitations. The scripts are in `bench/` (see `bench/README.md`); the working notes are in
`notes/research-log.md` and the code reviews in `notes/reviews.md`. Context sizes like "80k" mean the server's
`max_seq_len`/`cache_size` (80k = 81,920 tokens).

## Profiles

Decode speed is greedy code through the API (`bench/context_bench.py`), one request at a time, at the prompt
length shown. The first row is the current setup; the others were measured on the old base.

| Script | Draft | Context | DeltaNet replay | Peak VRAM | Decode |
|---|---|---|---|---|---|
| `serve/run_tabby.sh` (default), new base with batching, 4 slots | DFlash2 3.0 bpw, 7 tokens | 128k | on | 16.13 GB | 141 tok/s at 8.1k, 119 at 67.6k, 105.6 at 127.8k |
| `serve/run_tabby.sh` (default), old base, 1 slot | DFlash2 3.0 bpw, 7 tokens | 128k | on (`EXL3_GDN_REPLAY=1`) | 16.13 GB | 149.3 tok/s at 8.1k, 122.3 at 67.6k, 97.6 at 127.8k |
| `serve/run_tabby_long.sh` | built-in MTP head | 128k | off | not measured | 82.3 at 8.1k, 74.5 at 67.6k, 68.2 at 102.0k |
| `serve/run_tabby_xl.sh` | built-in MTP head | 160k | off | 16.24 of 16.3 GB | 62.1 at 125.6k, 59.2 at 160.0k |

The default is faster than the MTP profiles at every prompt length we measured. `long` is the conservative option: no
draft model, and replay off because we haven't tested replay with MTP. `xl` needs the whole card; anything else using
VRAM (a desktop session on the same GPU, a second model) will push it over.

Extra arguments go to TabbyAPI, e.g. `serve/run_tabby.sh --max-seq-len 65536 --cache-size 65536`.

## Recommended request settings

For hard questions, send `"reasoning_effort": "medium"` and this system prompt:

> Think efficiently. Work through the problem once, carefully. If you notice yourself re-deriving the same result,
> second-guessing a conclusion you already checked, or cycling between options, stop and commit to the
> best-supported answer. A confident best answer beats an unfinished analysis.

The default effort (`xhigh`) tells the model to "consider plausible alternatives", and on GPQA it then spiralled until
it hit the 30k token cap on 7 of 45 questions. With `medium` and this prompt it hit the cap on none and scored higher.
On the first 45 questions, which both runs answered, it took 4.6x less total time.

## Results

All numbers are from the one RX 9070 XT, measured on 5-6 October 2026 with the scripts in `bench/`. The batching
section and rows marked "new base" are on exllamav3 0662fac + our patches and TabbyAPI 2fd6cc7; the rest are on the
old base (exllamav3 f1cf869 + the single-request patches, TabbyAPI f07131c). Context sizes like "80k" mean the server's `max_seq_len`/`cache_size` (80k = 81,920
tokens).

### Decode speed by prompt length

`bench/context_bench.py` through TabbyAPI: a fresh filler document with a code buried mid-way, then ~450 tokens of
Python at temperature 0. Each cell is decode tok/s at the actual prompt length in tokens (8.1k = 8,136-8,137 tokens).
Every run recalled the code. Blank cells weren't measured.

| Setup (server config) | 8.1k | 33.2k | 50.4k | 62-68k | 80.5k | 95-102k | 125-128k | 142.8k | 160.0k |
|---|---|---|---|---|---|---|---|---|---|
| **New base, default, batching build, 4 slots (one request)** | 141 | | | 119 (67.6k) | | | 105.6 (127.8k) | | |
| Old base, default: DFlash2 3.0 bpw draft, replay (graph), 128k | 149.3 | | | 122.3 (67.6k) | | | 97.6 (127.8k) | | |
| DFlash2 3.0 bpw draft, no replay, 96k | 136.1 | | | 120.8 (63.3k) | | 112.6 (95.6k) | | | |
| DFlash2 3.0 bpw draft, first (eager) replay version, 128k | 132.3 | | | 111.6 (67.6k) | | | 96.5 (125.6k) | | |
| DFlash2 5.0 bpw draft, before the attention patch, 64k | 127.9 | 108.7 | 95.7 | 89.4 (62.2k) | | | | | |
| DFlash2 5.0 bpw draft, before the attention patch, 80k | 130.8 | | | | 80.3 | | | | |
| DFlash2 5.0 bpw draft, after the attention patch, 80k | 122.4 | 129.3 | 128.7 | 119.9 (63.3k) | 99.6 | | | | |
| MTP head, 128k | 82.3 | 83.4 | | 74.5 (67.6k) | | 68.2 (102.0k) | | | |
| MTP head, 144k | | | | | | | | 56.9 | |
| MTP head, 160k | | | | | | | 62.1 (125.6k) | | 59.2 |

### Batching (new base)

All through TabbyAPI 2fd6cc7 with the batching series: 128k context, Q4 cache, DFlash2 3.0 bpw draft with 7 draft
tokens, greedy 400-token code prompts, requests sent at the same time.

"Combined" is total tokens over a common window: from the first request's arrival to the last request's completion
in the server log, so it includes request handling and prefill. "Per-request mean rate" is each request's own
decode rate as TabbyAPI reports it; those rates cover different intervals, so they don't add up to the combined
figure.

| Setup | Per-request mean rate, tok/s | Combined, tok/s |
|---|---|---|
| 1 request alone | 131-132 | 131-132 |
| 2 at once | 118, 83 | 161 (800 tokens in 5.0 s) |
| 3 at once (two runs) | 90, 94, 68 / 90, 68, 95 | 194, 199 (1,200 tokens in 6.2 / 6.0 s) |
| 4 at once (three runs) | 88, 84, 69, 62 / 66, 83, 62, 89 / 62, 87, 84, 69 | 244, 240, 245 (1,600 tokens in 6.5-6.7 s) |
| Before the batching series (`rdna4-dev`), 2 / 4 at once | 36, 33 / 16, 18, 16, 20 | 63 / 61 (800 tokens in 12.7 s / 1,600 in 26.1 s) |
| Before the batching series, 2 at once with a 3-token draft | 58, 65 | 107 (800 tokens in 7.5 s) |
| `EXL3_DRAFT_ROW_BUDGET=16` / `24`, 4 at once | 52, 50, 54, 73 / 58, 50, 59, 55 | 196 / 194 (1,600 tokens in 8.2 s) |

The generator alone (`bench/bench_batch_sanity.py`, no server, common window): 188.0 tok/s at 2 at once (800 tokens
in 4.3 s) and 282.8 at 4 (1,600 tokens in 5.7 s), against 130.3 tok/s for the same prompts one at a time. The gap to
the API figures is request handling and prefill. Through the API, 4 at once is about 240-280 tok/s combined depending
on how much of that overhead a window includes.

Before the series, a batched verify of 2 x 8 or 4 x 8 rows fell off the 8-row WMMA GEMV onto a slow fallback, so
batching gave almost nothing. The 3-token-draft run (2 x 4 = 8 rows) is the diagnostic that showed it. The series
extends the WMMA GEMV to 16 and 32 rows and graph-captures the MLP up to 32 rows:

| Kernel | WMMA (new) | Fallback |
|---|---|---|
| down_proj, 32 rows | 151 µs | 796 µs |
| lm_head, 16 / 24 / 32 rows | 1.84 / 2.22 / 2.32 ms | 7.5 / 14.6 / 14.4 ms |

Correctness: relative error at most 7.4e-5 for 1-32 rows against the reference; bs1 greedy output identical to the
series' base over 4 x 400 tokens (`test_greedy_equal.py`); bs2 and bs4 batched outputs token-identical to bs1 runs over
400 tokens (`bench/bench_batch_sanity.py`). A 120k-token fill with 4 slots configured peaked at 16.13 GB, decoded at
141 / 119 / 105.6 tok/s at 8,136 / 67,604 / 127,779 prompt tokens, and recalled the buried code every time.

Memory: a test with 4 chats plus an image read 16.25 of 16.3 GB. Without expandable segments, a 106,294-token prefill
ran out of memory with 4 slots on a server that had already served concurrent chats and images: a 256 MB buffer
failed with 415 MB reserved but unallocated, which is fragmentation. With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(set in `setup/env.sh`) and the same prior load, a gate suite (0xSero's local-ai-registry: load, chat, reasoning,
tools, context with the 106,294-token prompt recalled, speed) passed at 3 slots (peak 16.17 GB) and at 4 slots (peak
16.28 of 16.3 GB). The config defaults to 3 slots (`max_batch_size: 3`) for the extra headroom.

`EXL3_DRAFT_ROW_BUDGET` (shorter drafts when many requests are active, to stay within 16 or 24 rows) is in the patch
but off: it was slower in every test (196 / 194 vs 244 tok/s at 4 at once).

### Decode speed through the API

`bench/bench_api.py`: 600 tokens, greedy code and sampled prose (temperature 0.7), short prompts.

| Server setup | Code, greedy | Prose, sampled |
|---|---|---|
| DFlash2 5.0 bpw draft, 64k, before the WMMA patch | 136.8 | 54.1 |
| DFlash2 5.0 bpw draft, with the WMMA patch | 145.4 | 58.4 |
| DFlash2 3.0 bpw draft, 7 tokens, 96k, with Q4W | 150.2 | 60.2 |
| No draft (2.5 bpw model, 256k) | 40.5 | 40.4 |

### Decode speed of the generator alone

`bench/smoke_test.py --long` loads the model directly (no server) and decodes 600 greedy tokens of code; the A/B rows
come from `test_gdn_replay.py` and a draft-strategy bench, as noted. These are the numbers behind the per-patch effects
below.

| Setup | tok/s |
|---|---|
| DFlash2 5.0 bpw draft, WMMA off / on | 127.1 / 145.1 |
| MTP head, WMMA off / on | 82.4 / 90.2 |
| DFlash2 5.0 bpw draft with WMMA, `EXL3_BC_GDN` (graph-captured DeltaNet decode) off / on | 145.4 / 153.7 |
| DFlash2 draft 5.0 / 3.0 bpw | 153.8 / 154.3 |
| No draft, 3.0 bpw | 38.1-38.4 |
| Replay off / on (`test_gdn_replay.py`: 379-token prompt, 400 greedy tokens, 3.0 bpw draft) | 89.3 / 92.4 |
| No draft (draft-strategy bench, 400 tokens) | 36.6 |

### Prefill

From the same `context_bench.py` and `bench_api.py` runs:

| Prompt tokens | Setup | Prefill |
|---|---|---|
| 7,741 / 29,351 | DFlash2 3.0 bpw draft, 96k (`bench_api.py`) | 1,246 / 1,261 tok/s |
| 8,137 / 67,604 | default profile | 1,337 / 1,102 tok/s |
| 80,508 | DFlash2 5.0 bpw draft, 80k | 1,034 tok/s |
| 127,779 | default profile | 904 tok/s |
| 160,026 | MTP head, 160k | 828 tok/s (193 s) |

### Accuracy: GPQA Diamond

First 50 questions of GPQA Diamond, thinking on, 30k token cap, temperature 1.0, top-p 0.95, top-k 20
(`bench/gpqa_eval.py`), one question at a time:

| Model and settings | Correct | Hit the token cap |
|---|---|---|
| 3.0 bpw, `xhigh` (45 questions) | 82.2% | 7 |
| **3.0 bpw, `medium` + anti-spiral prompt** | **86.0%** | 0 |
| 2.5 bpw, `medium` + anti-spiral prompt | 68.0% | 0 |

For reference, published results on all 198 questions (ISTA): BF16 89.9%, UD-IQ3_S 89.9%, UD-Q2_K_XL 86.9%. Fifty
questions is a small sample, so read our 86% as "in the same range", not as a ranking.

### 3.0 vs 2.5 bpw

| | 3.0 bpw | 2.5 bpw |
|---|---|---|
| Weights on disk | 13.8 GB | 12.3 GB |
| GPQA, 50 q, `medium` + anti-spiral | 86.0% | 68.0% |
| Perplexity (see below) | 7.063 | 7.248 |

On the same 50 questions, 10 were right at 3.0 bpw and wrong at 2.5, and 1 the other way round. That gap is too
big to be noise, so we stay on 3.0 bpw and get context from elsewhere.

### Perplexity

wikitext-2 raw, llama.cpp's method (512-token chunks, the second half of each scored, no BOS):

| Model | PPL |
|---|---|
| EXL3 3.0 bpw | 7.063 |
| EXL3 2.5 bpw | 7.248 |
| GGUF UD-Q3_K_XL (llama.cpp) | 7.068 |

### Vision

The model's vision encoder runs from system RAM (`vision_offload: true`), which costs about 20 ms per image. Keeping
it in VRAM doesn't fit at 128k. Images lower the DFlash2 draft's acceptance rate, so expect decode to be roughly
10-25% slower in chats with images.

## What the patches do

`patches/exllamav3-rdna4.patch` is the combined patch the build applies; `patches/features/` has the same work as 18
commits for reading, and `patches/README.md` explains both. The runtime switches are set in `setup/env.sh`.

**gfx12 WMMA multi-row GEMV** (`EXL3_GEMV_WMMA=1`). Speculative decoding verifies 2-8 draft tokens at once, and the
existing RDNA kernel for that ran at ~1.55x the cost of a 1-row decode. The new kernel decodes the trellis once per
tile with the tuned RDNA decoder, stages it through a per-wave LDS slot, and does all rows with one
`v_wmma_f32_16x16x16_f16`; the register layout was probed on the card (`kernels/wmma_layout_probe.hip`). An 8-row
verify now costs ~1.16x a 1-row decode. In the generator alone, DFlash2 went from 127.1 to 145.1 tok/s and MTP from
82.4 to 90.2; through the API, 136.8 to 145.4. Against the old kernel on the full model: top-1 agreement 100%,
KL ~3e-5. WMMA only runs on gfx1200/gfx1201.

**HIP graph re-instantiation** (`EXL3_GRAPH_REINST`, default 100k). `hipGraphExecKernelNodeSetParams` exhausts a fixed
kernel-argument pool in this HIP runtime and segfaults after ~1.6M updates, which is about an hour of serving
(`kernels/graph_setparams_repro.hip` reproduces it). `Graph::launch` now re-instantiates the graph every 100k updates,
so graph capture can stay on without the crash: in the generator, the graph-captured DeltaNet decode (`EXL3_BC_GDN`)
is 153.7 tok/s vs 145.4 without it. Output is byte-identical even when re-instantiating every 1,000 updates.

**Decode attention for GQA** (`EXL3_DEC_GROUP=1`, `EXL3_DEC_Q4W=1`, `EXL3_DEC_*` tuning). The split-decode attention
gave each program 16 query rows, so an 8-token verify split Qwen's 6-head GQA groups across 3 programs, each re-reading
and re-dequantizing the same Q4 K/V. Now each program takes a whole group, with warps, stages, tile and split counts
tuned for gfx1201. One layer at 60k depth: 1961 -> 605 µs at q_len 8, 1336 -> 450 at q_len 4, 550 -> 336 at q_len 1,
max error 2e-6 (`bench/bench_decode_attn.py`). Through the server, decode at ~63k tokens went from 89.4 tok/s (64k
config, before) to 119.9 (80k config, after), and at 80.5k from 80.3 to 99.6 (both 80k). The patch also fixes the C++
graph path, which recomputed the program count with the old rule and faulted the GPU.

The Q4W kernel, in the same patch, is a split-decode variant for 4-bit caches: it loads the packed words once and does
8 word-sliced dots instead of unpacking a whole tile. 1.2x the packed kernel at q_len 8 and 1.3x at q_len 4 (600 -> 510
µs per layer at 60k, q_len 8), and it helps at q_len 1 too. It is still bound by LDS shuffles: an fp16 cache runs the
same kernel at 582 GB/s (91% of peak) while the 4-bit cache gets ~140 GB/s.

**Review fixes**, folded into the commits above. A second model reviewed the patches (`notes/reviews.md`). Fixes: real
LDS fences around the WMMA exchanges (`__syncwarp()` instead of a bare wave barrier), WMMA gated to gfx1200/gfx1201 at
runtime, the `EXL3_DEC_*` overrides shared by the eager and graph attention paths and validated, an early exit for idle
Q4W splits (q_len 1 at 60k: 369 -> 350 µs per layer), and several robustness fixes in the replay code. Greedy output
and speed unchanged.

**DeltaNet accepted-input replay** (`EXL3_GDN_REPLAY=1`, on in the default profile). Qwen3.8's 48 Gated DeltaNet
layers keep a recurrent state, and speculative decoding normally stores a snapshot of it per draft token so it can
rewind (~0.15 GB per draft token). With replay, the verify pass reads the committed state without writing it and
records the small post-projection inputs; on acceptance it replays only the accepted prefix. The verify runs inside the
captured graph and the replay commit is two launches per step. In the `test_gdn_replay.py` A/B (379-token prompt, 400
greedy tokens), allocated VRAM went from 13.20 to 12.14 GB and decode from 89.3 to 92.4 tok/s, because the verify no
longer writes 8 state snapshots per layer per step. The state is bit-identical to the snapshot path for every accepted
length (712 kernel-level comparisons), and the greedy output is identical. That 1.06 GB is what lets the DFlash2
profile go from 96k to 128k. Run the test from the exllamav3 checkout (`bench/README.md`). The idea comes from
TensorFold.

**Batching series.** The gfx12 WMMA GEMV runs 16 and 32 rows (two M tiles), so a batched verify of up to 4 requests
x 8 rows stays on the fast kernel; the GatedMLP runs as a captured graph up to 32 rows, with graphs keyed by row count
and output dtype; LoRA adapters and per-projection controls take the unfused path; DFlash2 skips the head and selector
rows a short draft doesn't use. Effect: 4 requests at once go from 61 to about 244 tok/s combined through the API
(282.8 in the generator alone; see Batching above).
`EXL3_DRAFT_ROW_BUDGET` is included but off by default because it was slower.

**Optional, not recommended yet: PR #423, DFlash2 rejection sampling** (by Rafa,
[@rafatxf](https://github.com/rafatxf); off by default, `WITH_PR423=1 setup/build_exllamav3.sh` opts in). This is an
open upstream PR. For sampled requests, the draft path is sampled from the drafter's own distribution and accepted with
probability min(1, p/q), instead of only accepting draft tokens that match a sample from the target. Greedy decoding
doesn't use it. Our review of the PR's code (as rebased before the batching series) found these problems:

- A sampled request's speculative verify skips job-level token masks (`min_new_tokens`, banned-string continuations),
  so a stop token can end generation before `min_new_tokens`.
- `probs()` keeps exactly k tokens at a top-k tie, while the fused sampler keeps all tied tokens, so the two
  distributions can differ.
- The sampled DFlash2 path doesn't export `draft_conf`, which silently turns off confidence calibration.
- Extra device-to-host syncs.

On the GPU that earlier rebase sped up sampled prose from 59.3 to 62.6 tok/s with greedy output unchanged. We don't
recommend it until these are fixed upstream. The current patch is rebased onto the batching series with one
adaptation, compiled but not yet reviewed or run on the GPU. Details in `patches/README.md`.

**Draft requantization** (not a patch). The bf16 DFlash2 draft converted to EXL3 3.0 bpw (`setup/requant_draft.sh`)
is 0.89 GB instead of 1.4 GB for the 5.0 bpw quant, at the same speed (154.3 vs 153.8 tok/s in the generator). The
freed memory brought back the full 7-token draft and, before replay, took the DFlash2 profile from 80k to 96k.

## Where the memory and time go

VRAM with the 5.0 bpw draft and a 64k cache (before the requant and replay): main weights 10.25 GB (the lm_head alone
is 0.95 GB at 6 bpw), DFlash2 draft 1.67 GB, Q4 KV cache for 64k, and state, scratch, reserve and runtime ~2.4 GB.
The embedding table and the vision encoder stay in system RAM. The KV cache costs 24,192 bytes per token including
scales and the draft's cache; the DeltaNet state is 0.155 GB plus 0.152 GB per draft token without replay.

An 8-token verify forward is ~32 ms of GPU kernels: EXL3 GEMVs ~25 ms, the DeltaNet recurrence 2 ms, small fp16
projections 2.6 ms. At 1 row the EXL3 GEMVs reach 53-83% of the 640 GB/s memory bandwidth, except the small k/v
projections at ~25%. A token takes ~1,250-1,450 kernel launches.

## What didn't work

- `EXL3_QC_STAGING=0`: the model then fails to load.
- 2-bit KV cache for the draft: slower in our runs, and we didn't keep a clean A/B.
- K4/V3 KV cache for the main model (96k, 2-bit draft cache in both): 110.0 vs 118.1 tok/s at 8.1k and 53.2 vs 66.1
  at 97.7k against Q4, and no VRAM headroom gained.
- A 4-token draft to save memory (80k): 84.5 vs 119.9 tok/s at 63.3k against 7 tokens, because fewer tokens per step
  means more full KV reads.
- Lowering `autosplit_reserve`: it frees nothing that's actually resident.
- A "Q4P" permuted-gather unpack for the 4-bit attention: 2.3x slower.
- A hand-written gfx12 WMMA attention kernel (`kernels/`): 598 µs per layer at best vs 510 for Triton Q4W.
- Adaptive draft length, in a direct draft-strategy bench (400 tokens, greedy code / prose sampled at temperature
  0.8): neutral, 121.3 vs 121.4 tok/s code and 59.2 vs 58.7 prose.
- A sequential draft-tree fallback, same bench: slower, 118.7 code and 54.2 prose. A real one-pass tree needs
  tree-masked attention and branched DeltaNet state.
- `EXL3_DRAFT_ROW_BUDGET` (shorter drafts when many requests are active): 194-196 tok/s combined at 4 requests vs
  244 without it. It's in the patch, off by default.
- TensorFold itself has no working AMD path (its PR144 gives NaN logits on gfx1201). We ported the replay idea instead.
- 2.5 bpw to free memory: costs too much accuracy (above).

## Known limitations

- Most single-request tables were measured on the old base (exllamav3 f1cf869, TabbyAPI f07131c); the batching results
  and the "new base" rows are on the current one.
- Speculative decoding isn't bit-identical to plain greedy decoding. Every speculative mode, plain DFlash2 included,
  diverged from non-speculative greedy output at the same token (375 of 400) in our test. That comes from the
  multi-row verify's accumulation order, not from the replay code.
- Replay is only tested with the DFlash2 draft. The MTP profiles turn it off.
- The WMMA GEMV only runs on gfx1200/gfx1201. Everything was tested on one RX 9070 XT; the RX 9070 and RDNA3 cards are
  untested.
- Concurrency is `max_batch_size: 3` by default. 4 works and passed the same long-context gate, but peaks at 16.28 of
  16.3 GB (see Batching). Without expandable segments in the allocator, a long prompt after mixed concurrent use can
  run out of memory.
- The Q4W attention kernel is still far from memory bandwidth on 4-bit caches (~140 GB/s); that needs a hand-written
  HIP kernel that beats Triton, which we don't have.
- On gfx1201 with Fedora 44, the HSA exit handler can segfault at interpreter shutdown and leave the GPU at 100%. The
  direct test scripts tear down explicitly and call `os._exit` to avoid it.
- `setup/env.sh` sets `HIP_VISIBLE_DEVICES=0` because device 0 is the 9070 XT on the test machine (its iGPU is
  hidden that way). Check yours with `rocminfo`.
- A newer upstream will need the patches ported. `setup/regen_patches.sh` regenerates them from a ported branch, and
  `setup/versions.sh` holds the pinned commits.
- Nothing starts on boot.

## Repository layout

```
setup/     versions.sh (pinned commits and patch files), lib.sh, fetch_deps.sh, make_venv.sh, requirements*.txt,
           build_exllamav3.sh, install_tabby.sh, download_models.sh, requant_draft.sh, env.sh, regen_patches.sh
serve/     run_tabby.sh, run_tabby_long.sh, run_tabby_xl.sh, stop_tabby.sh, config.example.yml, open-webui.md
patches/   exllamav3-rdna4.patch, features/ (the same as 18 commits), optional/ (PR #423)
bench/     speed, context, batching, GPQA and kernel tests (bench/README.md)
kernels/   standalone HIP experiments: WMMA layout probe, HIP graph repro, hand-written attention
notes/     measurements.md (this file), research-log.md (the working notes), reviews.md (the code reviews)
```

The setup scripts create `.venv/`, `exllamav3/`, `tabbyAPI/`, `deps/`, `models/` and `logs/` in the repo root; all
are git-ignored.
