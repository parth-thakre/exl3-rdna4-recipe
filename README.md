# Qwen3.8-27B on a 16 GB RX 9070 XT: ExLlamaV3 (EXL3) + TabbyAPI

A recipe for running Qwen3.8-27B in EXL3 on an AMD RX 9070 XT or RX 9070 (gfx1201, RDNA4, 16 GB) under Linux, with
ExLlamaV3's ROCm path, a set of RDNA4 patches, speculative decoding with a DFlash2 draft model, and TabbyAPI as an
OpenAI-compatible server.

With everything below, the default profile serves a 128k context and decodes at 149 tok/s at 8k depth, 122 at 64k and
97.6 at 120k on greedy code (DFlash2 draft, 7 tokens per step). Without speculative decoding the same card does
35-38 tok/s. The rest of this README is how to reproduce that, what each patch does, and what didn't work.

Tested on one machine:

| | |
|---|---|
| GPU | RX 9070 XT, 16 GB (gfx1201). The RX 9070 is the same chip with the same memory but we haven't run it. |
| OS | Fedora 44, kernel 7.2 |
| ROCm | Fedora's system ROCm 7.1.1 packages to compile; PyTorch 2.13.0+rocm7.2 wheels (they ship their own ROCm libraries) to run |
| Python | 3.12.14, triton-rocm 3.7.1 |
| ExLlamaV3 | `dev` at f1cf869 + `patches/exllamav3-rdna4.patch` |
| TabbyAPI | f07131c + `patches/tabbyapi-allow-rdna3-rdna4.patch` |
| Model | `turboderp/Qwen3.8-27B-exl3`, 3.00bpw branch |
| Draft | DFlash2 draft requantized to EXL3 3.0 bpw from the bf16 release |

## Quick start

Prerequisites: Fedora 44 with its ROCm packages, `python3.12`, `git`, about 20 GB of disk for models and 15 GB for
the venv, and access to the GPU (`/dev/kfd` and `/dev/dri/renderD*`; on Fedora that means the `render` group). The
test machine had these ROCm packages installed: `rocm-hip-devel rocm-runtime-devel hipcc rocm-clang-devel rocm-llvm-devel rocm-comgr-devel
rocm-libc++-devel rocblas-devel hipblas-devel hipblas-common-devel rocminfo rocm-smi`.

Run everything from the repo root:

```bash
setup/fetch_deps.sh          # unpack missing -devel headers into deps/ (no sudo; or dnf install them, see the script)
setup/make_venv.sh           # .venv/ with torch 2.13.0+rocm7.2 and the tested package versions
setup/build_exllamav3.sh     # clone exllamav3 @ f1cf869, apply the patch, compile for gfx1201, pip install -e
setup/install_tabby.sh       # clone TabbyAPI @ f07131c, apply the patch, write tabbyAPI/config.yml
setup/download_models.sh     # main model (13.8 GB) + the bf16 DFlash2 draft (3.8 GB)
setup/requant_draft.sh       # bf16 draft -> models/Qwen3.8-27B-DFlash2-EXL3-3.0bpw (uses the GPU, a few minutes)

source setup/env.sh && python bench/smoke_test.py --draft models/Qwen3.8-27B-DFlash2-EXL3-3.0bpw   # optional check

serve/run_tabby.sh           # default profile, http://127.0.0.1:8096/v1
```

TabbyAPI creates `tabbyAPI/api_tokens.yml` with a random API key on first start. Then:

```bash
KEY=$(awk '/^api_key:/{print $2}' tabbyAPI/api_tokens.yml)
curl -s http://127.0.0.1:8096/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "x", "messages": [{"role": "user", "content": "Write a haiku about VRAM."}], "max_tokens": 400}'
```

If you'd rather not requantize, `setup/download_models.sh draft-mia` fetches Mia-AiLab's ready-made 5.0 bpw DFlash2
quant (1.4 GB). Set `draft_model_name: Qwen3.8-27B-DFlash2-EXL3-5.0bpw` in `tabbyAPI/config.yml` and lower
`max_seq_len`/`cache_size`: the bigger draft leaves less room for context (we ran it at 96k with an older setup).

The server listens on 127.0.0.1 only. To reach it over Tailscale, set `network.host` in `tabbyAPI/config.yml` to the
machine's Tailscale IP (`tailscale ip -4`) and keep auth on. `serve/open-webui.md` has the podman command for Open
WebUI.

## Profiles

| Script | Draft | Context | DeltaNet replay | Peak VRAM | Decode, greedy code |
|---|---|---|---|---|---|
| `serve/run_tabby.sh` (default) | DFlash2 3.0 bpw, 7 tokens | 128k | on (`EXL3_GDN_REPLAY=1`) | 16.13 GB | 149 tok/s at 8k, 122 at 64k, 97.6 at 120k |
| `serve/run_tabby_long.sh` | built-in MTP head | 128k | off | not measured | 82 at 8k, 75 at 60-64k, 62 at 118-125k |
| `serve/run_tabby_xl.sh` | built-in MTP head | 160k | off | 16.24 of 16.3 GB | 57 at 134k, 59 at 150-160k |

The default is faster than the MTP profiles at every depth we measured. `long` is the conservative option: no draft
model, and replay off because we haven't tested replay with MTP. `xl` needs the whole card; anything else using VRAM
(a desktop session on the same GPU, a second model) will push it over.

Extra arguments go to TabbyAPI, e.g. `serve/run_tabby.sh --max-seq-len 65536 --cache-size 65536`.

## Recommended request settings

For hard questions, send `"reasoning_effort": "medium"` and this system prompt:

> Think efficiently. Work through the problem once, carefully. If you notice yourself re-deriving the same result,
> second-guessing a conclusion you already checked, or cycling between options, stop and commit to the
> best-supported answer. A confident best answer beats an unfinished analysis.

The default effort (`xhigh`) tells the model to "consider plausible alternatives", and on GPQA it then spiralled until
it hit the 30k token cap on 7 of 45 questions. With `medium` and this prompt it hit the cap on none, scored higher, and
finished 4.6x faster.

## Results

All numbers are from the one RX 9070 XT above, measured in early October 2026 with the scripts in `bench/`.

### Decode speed by context depth

`bench/context_bench.py` through TabbyAPI: a fresh filler document with a code buried mid-way, then ~450 tokens of
Python at temperature 0; each run also checks that the model recalls the code. tok/s:

| Setup | 8k | 32k | 48k | 60-64k | 76k | 90-96k | 118-125k | 134k | 150-160k |
|---|---|---|---|---|---|---|---|---|---|
| **Default: DFlash2 3.0 bpw draft, replay (graph), 128k** | 149 | | | 122 | | | 97.6 | | |
| DFlash2 3.0 bpw draft, no replay, 96k | 136 | | | 121 | | 113 | | | |
| DFlash2, first (eager) replay version, 128k | 132 | | | 112 | | | 96.5 | | |
| DFlash2 5.0 bpw draft, 64k, before the attention patch | 128 | 109 | 96 | 89 | 80 | | | | |
| DFlash2 5.0 bpw draft, 64k, after the attention patch | 122 | 129 | 129 | 120 | 100 | | | | |
| MTP head, 128k / 144k / 160k | 82 | 83 | | 75 | | 68 | 62 | 57 | 59 |

End to end through the API (`bench/bench_api.py`, 600 tokens; code is greedy, prose is sampled at temperature 0.7):

| Setup | Code, greedy | Prose, sampled |
|---|---|---|
| DFlash2 5.0 bpw draft, 64k, before the patches | 127 | |
| DFlash2 5.0 bpw draft, 64k, all patches | 153.7 | ~58 |
| DFlash2 3.0 bpw draft, 96k | 150 | 60 |
| MTP head, before / after the WMMA patch | 82 / 90 | |
| No speculative decoding | ~35-38 | |

### Prefill

| Depth | Prefill |
|---|---|
| 8-32k | ~1,250 tok/s |
| 80k | ~1,035 tok/s |
| 125k (DFlash2, 128k profile) | 911 tok/s |
| 160k (xl profile) | ~830 tok/s, about 3 minutes for a full window |

### Accuracy: GPQA Diamond

First 50 questions of GPQA Diamond, thinking on, 30k token cap, temperature 1.0, top-p 0.95, top-k 20
(`bench/gpqa_eval.py`):

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

All against exllamav3 `dev` at f1cf869. `patches/exllamav3-rdna4.patch` is the combined patch the build applies;
`patches/features/` has the same work split by feature for reading (see `patches/README.md`). The runtime switches
are set in `setup/env.sh`.

**gfx12 WMMA multi-row GEMV** (`EXL3_GEMV_WMMA=1`). Speculative decoding verifies 2-8 draft tokens at once, and the
existing RDNA kernel for that ran at ~1.55x the cost of a 1-row decode. The new kernel decodes the trellis once per
tile with the tuned RDNA decoder, stages it through a per-wave LDS slot, and does all rows with one
`v_wmma_f32_16x16x16_f16`; the register layout was probed on the card (`kernels/wmma_layout_probe.hip`). An 8-row verify
now costs ~1.16x a 1-row decode. DFlash2 went from 127 to 145 tok/s and MTP from 82 to 90. Against the old kernel on
the full model: top-1 agreement 100%, KL ~3e-5. It includes the multi-row over-read fix that turboderp also committed
upstream as cc50bbb.

**HIP graph re-instantiation** (`EXL3_GRAPH_REINST`, default 100k). `hipGraphExecKernelNodeSetParams` exhausts a fixed
kernel-argument pool in this HIP runtime and segfaults after ~1.6M updates, which is about an hour of serving
(`kernels/graph_setparams_repro.hip` reproduces it). `Graph::launch` now re-instantiates the graph every 100k updates,
so HIP graphs stay on (+6%) without the crash. Output is byte-identical even when re-instantiating every 1,000 updates.

**Decode attention for GQA** (`EXL3_DEC_GROUP=1`, `EXL3_DEC_Q4W=1`, `EXL3_DEC_*` tuning). The split-decode attention
gave each program 16 query rows, so an 8-token verify split Qwen's 6-head GQA groups across 3 programs, each re-reading
and re-dequantizing the same Q4 K/V. Now each program takes a whole group, with warps, stages, tile and split counts
tuned for gfx1201. One layer at 60k depth: 1961 -> 605 µs at q_len 8, 1336 -> 450 at q_len 4, 550 -> 336 at q_len 1,
max error 2e-6 (`bench/bench_decode_attn.py`). In the decode-by-depth table this is the "before/after the attention
patch" pair: 89 -> 120 tok/s at 60-64k. The patch also fixes the C++ graph path, which recomputed the program count
with the old rule and faulted the GPU.

The Q4W kernel, in the same patch, is a split-decode variant for 4-bit caches: it loads the packed words once and does
8 word-sliced dots instead of unpacking a whole tile. 1.2x the packed kernel at q_len 8 and 1.3x at q_len 4 (600 -> 510 µs
per layer at 60k, q_len 8), and it helps at q_len 1 too. It is still bound by LDS shuffles: an fp16 cache runs the same
kernel at 582 GB/s (91% of peak) while the 4-bit cache gets ~140 GB/s.

**Review fixes.** A second model reviewed the RDNA4 patches (`notes/reviews.md`). Fixes: real LDS fences around the WMMA
exchanges (`__syncwarp()` instead of a bare wave barrier), WMMA gated to gfx1200/gfx1201 at runtime, the `EXL3_DEC_*`
overrides shared by the eager and graph attention paths and validated, and an early exit for idle Q4W splits (q_len 1
at 60k: 369 -> 350 µs per layer). Greedy output and speed unchanged.

**DeltaNet accepted-input replay** (`EXL3_GDN_REPLAY=1`, on in the default profile). Qwen3.8's 48 Gated DeltaNet
layers keep a recurrent state, and speculative decoding normally stores a snapshot of it per draft token so it can
rewind (~0.15 GB per draft token). With replay, the verify pass reads the committed state without writing it and
records the small post-projection inputs; on acceptance it replays only the accepted prefix. The verify runs inside the
captured graph and the replay commit is two launches per step. Result: -1.06 GB of VRAM and +3.4% decode (92.4 vs
89.3 tok/s in the A/B test), because the verify no longer writes 8 state snapshots per layer per step. The state is
bit-identical to the snapshot path for every accepted length (712 kernel-level comparisons), and 400-token greedy
output is identical. That 1.06 GB is what lets the DFlash2 profile go from 96k to 128k. The test is
`test_gdn_replay.py` in the exllamav3 checkout. The idea comes from TensorFold.

**TabbyAPI: allow RDNA3/RDNA4.** At f07131c, TabbyAPI's hardware check rejects every ROCm GPU for the exllamav3
backend. The patch accepts gfx11 and gfx12 devices. We have only run gfx1201.

**Draft requantization** (not a patch). The bf16 DFlash2 draft converted to EXL3 3.0 bpw (`setup/requant_draft.sh`)
is 0.89 GB instead of 1.4 GB for the 5.0 bpw quant, at the same speed (154.3 vs 153.8 tok/s). The freed memory brought
back the full 7-token draft and, before replay, took the DFlash2 profile from 80k to 96k.

## Where the memory and time go

VRAM in the DFlash2 profile at 64k (before the requant and replay): main weights 10.25 GB (the lm_head alone is
0.95 GB at 6 bpw), DFlash2 draft 1.67 GB, Q4 KV cache for 64k ~1 GB, state, scratch, reserve and runtime ~2.4 GB.
The embedding table and the vision encoder stay in system RAM. The KV cache costs 24,192 bytes per token including
scales and the draft's cache; the DeltaNet state is 0.155 GB plus 0.152 GB per draft token without replay.

An 8-token verify forward is ~32 ms of GPU kernels: EXL3 GEMVs ~25 ms, the DeltaNet recurrence 2 ms, small fp16
projections 2.6 ms. At 1 row the EXL3 GEMVs reach 53-83% of the 640 GB/s memory bandwidth, except the small k/v
projections at ~25%. A token takes ~1,250-1,450 kernel launches.

## What didn't work

- `EXL3_QC_STAGING=0`: the model then fails to load.
- 2-bit KV cache for the draft: 8% slower.
- K4/V3 KV cache for the main model: slower, and no VRAM headroom gained.
- A 4-token draft to save memory: 30% slower at 60k, because fewer tokens per step means more full KV reads.
- Lowering `autosplit_reserve`: it frees nothing that's actually resident.
- A "Q4P" permuted-gather unpack for the 4-bit attention: 2.3x slower.
- A hand-written gfx12 WMMA attention kernel (`kernels/`): 598 µs per layer at best vs 510 for Triton Q4W.
- Adaptive draft length: neutral (121.3 vs 121.4 tok/s code, 59.2 vs 58.7 prose).
- A sequential draft-tree fallback: slower (118.7 code, 54.2 prose). A real one-pass tree needs tree-masked attention
  and branched DeltaNet state.
- A 16-row WMMA GEMV works (lm_head at 16 rows: 1.85 ms vs 7.5 ms) but has no use while the DFlash2 block is 8 tokens.
  It isn't in the patch.
- Batching requests: with MTP (the only setup where batch 4 fits) four requests together got ~44 tok/s in total,
  less than one DFlash2 request alone.
- TensorFold itself has no working AMD path (its PR144 gives NaN logits on gfx1201). We ported the replay idea instead.
- 2.5 bpw to free memory: costs too much accuracy (above).

## Known limitations

- Speculative decoding isn't bit-identical to plain greedy decoding. Every speculative mode, plain DFlash2 included,
  diverged from non-speculative greedy output at the same token (375 of 400) in our test. That comes from the
  multi-row verify's accumulation order, not from the replay code.
- Replay is only tested with the DFlash2 draft. The MTP profiles turn it off.
- The WMMA GEMV only runs on gfx1200/gfx1201. Everything was tested on one RX 9070 XT; the RX 9070 and RDNA3 cards are
  untested.
- `max_batch_size: 1`. The profiles are sized for one request at a time.
- The Q4W attention kernel is still far from memory bandwidth on 4-bit caches (~140 GB/s); that needs a hand-written
  HIP kernel that beats Triton, which we don't have.
- On gfx1201 with Fedora 44, the HSA exit handler can segfault at interpreter shutdown and leave the GPU at 100%. The
  direct test scripts tear down explicitly and call `os._exit` to avoid it.
- `setup/env.sh` sets `HIP_VISIBLE_DEVICES=0` because device 0 is the 9070 XT on the test machine (its iGPU is
  hidden that way). Check yours with `rocminfo`.
- The patches target exllamav3 f1cf869. A newer upstream will likely need them ported; `setup/regen_patches.sh`
  regenerates the combined patch from a ported branch, and `setup/versions.sh` holds the pinned commits.
- Nothing starts on boot.

## Repository layout

```
setup/     versions.sh (pinned commits), fetch_deps.sh, make_venv.sh, requirements*.txt, build_exllamav3.sh,
           install_tabby.sh, download_models.sh, requant_draft.sh, env.sh, regen_patches.sh
serve/     run_tabby.sh, run_tabby_long.sh, run_tabby_xl.sh, config.example.yml, open-webui.md
patches/   exllamav3-rdna4.patch, tabbyapi-allow-rdna3-rdna4.patch, features/ (per-feature split)
bench/     speed, context, GPQA and kernel tests (bench/README.md)
kernels/   standalone HIP experiments: WMMA layout probe, HIP graph repro, hand-written attention
notes/     research-log.md (the full working notes), reviews.md (the code reviews)
```

The setup scripts create `.venv/`, `exllamav3/`, `tabbyAPI/`, `deps/`, `models/` and `logs/` in the repo root; all
are git-ignored.

## Credits

- [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3): EXL3, the ROCm path, the converter, and the
  Qwen3.8-27B EXL3 quants.
- [theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI): the server.
- DFlash2 by [z-lab](https://github.com/z-lab/dflash) / [inco.ai](https://inco.ai/blog/dflash2/): the draft model,
  [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (Apache-2.0). Mia-AiLab for the
  first EXL3 quant of it.
- TensorFold, for the idea of replaying only accepted DeltaNet inputs.
- The Qwen team, for Qwen3.8-27B.

## License

The scripts, notes and kernel experiments in this repo are MIT (`LICENSE`). The patches are modifications of other
projects and fall under those projects' licenses: the exllamav3 patches under exllamav3's MIT license, the TabbyAPI
patch under TabbyAPI's AGPL-3.0. Models and datasets keep their own licenses.
