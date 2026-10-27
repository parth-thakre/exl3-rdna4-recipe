# Measurements and details

The detail behind the main README: how each profile and patch performs, how it was measured, what didn't work, and
the known limitations. The scripts are in `bench/` (see `bench/README.md`); the working notes are in
`notes/research-log.md` and the code reviews in `notes/reviews.md`. Context sizes like "80k" mean the server's
`max_seq_len`/`cache_size` (80k = 81,920 tokens).

## Profiles

Decode speed is greedy code through the API (`bench/context_bench.py`), one request at a time, at the prompt
length shown. Rows marked "current base" were measured on 9cc3bd1 with our DeltaNet scan patch, which has the same
kernels as the 0f685aa build the recipe uses now (see "Current base"); the previous base is 0662fac and the old base
f1cf869.

| Script | Draft | Context | DeltaNet replay | Peak VRAM | Decode |
|---|---|---|---|---|---|
| `serve/run_tabby.sh` (default), current base, 3 slots | DFlash2 3.0 bpw, 7 tokens | 128k | on (upstream's) | 16.15 GB (first port, same memory use) | 122.2 at 67.6k, 106.6 at 127.8k (3-round medians) |
| `serve/run_tabby.sh` (default), previous base with batching, 4 slots | DFlash2 3.0 bpw, 7 tokens | 128k | on | 16.13 GB | 141 tok/s at 8.1k, 119 at 67.6k, 105.6 at 127.8k |
| `serve/run_tabby.sh` (default), old base, 1 slot | DFlash2 3.0 bpw, 7 tokens | 128k | on (`EXL3_GDN_REPLAY=1`) | 16.13 GB | 149.3 tok/s at 8.1k, 122.3 at 67.6k, 97.6 at 127.8k |
| `serve/run_tabby_long.sh`, current base | built-in MTP head | 128k | on (upstream's) | 14.09 GB | 92.3 at 8.1k, 84.0 at 67.6k, 66.0 at 127.8k |
| `serve/run_tabby_xl.sh`, current base | built-in MTP head | 160k | on (upstream's) | 14.83 GB | 68.7 at 127.8k, 63.6 at 160.0k |
| `serve/run_tabby_long.sh`, old base | built-in MTP head | 128k | off | not measured | 82.3 at 8.1k, 74.5 at 67.6k, 68.2 at 102.0k |
| `serve/run_tabby_xl.sh`, old base | built-in MTP head | 160k | off | 16.24 of 16.3 GB | 62.1 at 125.6k, 59.2 at 160.0k |

The default is faster than the MTP profiles at every prompt length we measured. The MTP profiles need no draft model.
On the previous and old bases they ran with replay off (it was never tested with MTP there), so every slot kept the
full DeltaNet history and `xl` needed the whole card. On the current base upstream's replay is always on, MTP included:
`xl` peaks at 14.83 GB with a full 160k prompt, and `long` with 3 slots peaks at 14.70 GB at 127.8k (one request).
Both scripts still allow one request at a time, because concurrent requests with MTP drafting are untested. Every
current-base run recalled the code; greedy code through `bench/bench_api.py` ran at about 101 tok/s and sampled prose
at 48-51.

Extra arguments go to TabbyAPI, e.g. `serve/run_tabby.sh --max-seq-len 65280 --cache-size 65536`, or set the context
with `CTX=64k serve/run_tabby.sh`.

### Context and other VRAM use (the default is 96k)

What fits depends on what else is on the card: a desktop, a browser. Measured with the published container image
(5cc1deb) on Parth's RX 9070 XT, 2026-10-13: 3 concurrent requests plus an image request, then the local-ai registry's
six gates (load, chat, reasoning, tools, context, speed). "In use before" is VRAM used before the server starts: the
real KDE desktop and a browser, plus a held allocation standing in for a heavier desktop where noted.

| Profile, context | In use before | Peak | Result |
|---|---|---|---|
| DFlash2, 128k | 1.14 GB (desktop) | 16,265 of 16,304 MiB | passes all six gates |
| DFlash2, 96k | 1.14 GB (desktop) | 16,206 MiB | passes |
| DFlash2, 80k | 1.14 GB (desktop) | 15,767 MiB | passes |
| DFlash2, 64k | 1.14 GB (desktop) | 15,978 MiB | passes |
| DFlash2, 128k | 1.93 GB (desktop + 1.2 GB held) | | server exits while loading |
| DFlash2, 96k | 1.93 GB (desktop + 1.2 GB held) | 16,276 MiB | passes, 112 tok/s |
| MTP (`run_tabby_long.sh`), 128k | 1.09 GB (desktop) | 16,288 MiB | passes, 107 tok/s |
| MTP (`run_tabby_long.sh`), 128k | 2.14 GB (desktop + 1.2 GB held) | | server exits while loading |

So the default is 96k, and `CTX=128k` is for a card with nothing else on it or a light desktop. The MTP profile isn't a
way to keep 128k next to a heavier desktop either: at 128k it also didn't load with ~2.1 GB in use, and it serves one
request at a time. PyTorch takes what's free, so the peak is close to the card's size in every row; it isn't the
minimum each needs.

### What 128k needs, and what lowers it (2026-10-14)

The same test at 128k on the d021207/6ffe066 image (aec2d311), started the way the plugin starts it, with a block of
VRAM held by another process standing in for a heavier desktop, stepped up until the server failed. A held block is
a cautious stand-in: with the earlier build, a real desktop using ~1.1 GB passed at 128k, while here ~0.8 GB in use,
nearly all of it held, didn't. Probably the driver can move a desktop's buffers to system memory when the card fills
up, but not compute allocations (not verified). "In use" is sysfs VRAM used before the server starts.

| Setting (128k) | Passes with | Fails with | Prefill, 64k / 96k / 120k prompt | Decode, 64k / 96k / 120k |
|---|---|---|---|---|
| default (Q4 cache) | 667 MiB in use | 783 MiB | 997 / 863 / 867 tok/s | 131 / 123 / 104 |
| `EXL3_QC_STAGING=0` (+ `--autosplit-reserve 256`) | 873 MiB | 975 MiB | 825 / 657 / 623 tok/s | 131 / 120 / 108 |
| `--cache-mode 4,3 --autosplit-reserve 256` | 1,025 MiB | 1,281 MiB | 1,096 / 966 / 869 tok/s | 119 / 106 / 95 |
| `--cache-mode 3,3 --autosplit-reserve 256` | 1,281 MiB | 1,537 MiB | 1,094 / 963 / 867 tok/s | 112 / 101 / 88 |

Two rounds for the first two rows; one round for the cache modes, run together with `--cache-mode 4,4` (the same as Q4)
as their reference: 1,097 / 968 / 869 prefill, 120 / 120 / 107 decode. Prefill at 64k and 96k came out ~10% higher
in that run than in the first rows' for the same cache, so compare within a run. Recall was right at every depth in
every mode.

- Where it fails: the long-context gate's 106k-token prefill. By default ExLlamaV3 dequantizes the whole Q4 cache pool
  of one layer into an fp16 scratch for prefill attention (K and V, 256 MiB each at 128k for this model's 4 KV heads
  of 256), then runs the fp16 kernel over it. `EXL3_QC_STAGING=0` reads the quantized cache in the kernel instead: no
  scratch, about 0.2 GB more room, but 17-28% slower prefill on long prompts (a full 128k prompt takes ~205 s instead of
  ~147 s). Decode and short prompts are unchanged (code 157 tok/s either way; 7.7k prefill 1,160 vs 1,116, 29k 1,020
  vs 1,068). Not the default; worth it only to fit 128k next to a slightly heavier desktop.
- Next in line after that: a 170 MiB fp16 weight that prefill reconstructs for the large MLP GEMMs, and ~0.5 GB that
  PyTorch holds reserved but can't use (fragmentation). Expandable segments would fix the fragmentation but corrupt
  memory on gfx1201 (see Limitations); `max_split_size_mb` 256 or 128 didn't help, and 64 made it worse.
- The load check: below that, ExLlamaV3's autosplit refuses to load ("Insufficient VRAM in split for model and cache")
  when free VRAM minus `autosplit_reserve` (768 MB in the configs) doesn't cover the model and cache. Lowering the
  reserve only moves the failure from load to the first long prompt, which is worse; keep it.
- No effect at the load check, where they all failed at the same point (not tested past it): `--chunk-size 512`,
  `--max-batch-size 2`, `--draft-num-tokens 5`, a 3-bit draft cache. turboderp's `SC_3.00bpw_H4_V4` quant is 0.46 GB smaller only because
  its vision encoder is 4-bit, which runs from system RAM here, so it saves no VRAM.
- A 3-bit cache: `--cache-mode 3,3` holds 3 bits per K and V value instead of 4, so the 128k cache is ~0.5 GB smaller
  (16 attention layers x 4 KV heads x 256 x 2 x 1/8 byte less per token), and the staging scratch stays the same.
  It gives the most room (~0.6 GB more than the default) without slowing prefill, but decode at long context is
  7-18% slower (the split-decode Q4W kernel is 4-bit only; other bit widths take the generic path). 4,3 is halfway.
  GPQA Diamond (198 questions, medium + anti-spiral, 3 at a time, through the image at 128k): 154 with 3,3 against 160
  with 4,4 in the same container (17 questions right only with 4,4, 11 only with 3,3). 154 is below every 4-bit run so
  far (158, 160, 164), so a small quality loss is likely, though one comparison can't confirm it. Opt-in only.
- What would: blockwise staging (dequantize a few thousand tokens at a time and merge the partial attention by its
  log-sum-exp) would keep the fast fp16 kernel without the full-pool scratch. The prefill kernel doesn't return the
  log-sum-exp yet, so that's kernel work, not done.

## Recommended request settings

For hard questions, send `"reasoning_effort": "medium"` and this system prompt:

> Think efficiently. Work through the problem once, carefully. If you notice yourself re-deriving the same result,
> second-guessing a conclusion you already checked, or cycling between options, stop and commit to the
> best-supported answer. A confident best answer beats an unfinished analysis.

The default effort (`xhigh`) tells the model to "consider plausible alternatives", and on GPQA it then spiralled until
it hit the 30k token cap on 7 of 45 questions. With `medium` and this prompt it hit the cap on none and scored higher.
On the first 45 questions, which both runs answered, it took 4.6x less total time.

When accuracy matters more than time, use `xhigh` with the same prompt and a large `max_tokens` (60k). On all 198 GPQA
Diamond questions it scored 88.4% against 82.8% for `medium` on the same build, at about 3x the median time per question (see
"Accuracy: GPQA Diamond" below).

## Results

All numbers are from the one RX 9070 XT, measured on 5-13 October 2026 with the scripts in `bench/`. The main model
is turboderp's plain 3.0 bpw quant except in "Current base" and "Self-calibrated 3.0 bpw" below and the rows marked SC H4 (the current
default, about 3% slower on greedy code; that section). The batching section, the self-calibrated comparison and rows
marked "new base" are on exllamav3 0662fac + our patches (the previous base) and TabbyAPI 2fd6cc7; "Current base"
is on 9cc3bd1 (the recipe now builds v1.6.5, with the same kernels; checks in that section); the rest are on the old base (exllamav3 f1cf869 + the single-request
patches, TabbyAPI f07131c). Context sizes like "80k" mean the server's `max_seq_len`/`cache_size` (80k = 81,920
tokens).

### Current base (since 2026-10-12)

Since 2026-10-27 the recipe builds the v1.6.5 release (cdc785d) with the same 15 commits (one hand edit, see
`patches/README.md`). Checks of the port against the 0f685aa build (`wt-up1013`, which the numbers in the rest of this
file were measured on), 2026-10-27, SC H4 model, DFlash2 3.0 bpw draft, 7 draft tokens:

- Kernel tests: attention 382 passed (64 skipped), recurrent 923 passed (22 skipped); `test_gemv32.py` passes (11
  projections, rows 1-32, fp16/fp32, row-tile invariance, fused gate+up, graphed MLP rows 1-32).
- Greedy output (`test_greedy_equal.py`, 4 x 400 tokens): with every patch switch at its stock value
  (`EXL3_GEMV_WMMA=0 EXL3_DEC_GROUP=0 EXL3_DEC_Q4W=0 EXL3_BC_MLP_MAX_ROWS=8`, no `EXL3_DEC_WARPS/STAGES/BLOCK_N`) the
  port is token-identical to plain v1.6.5, and plain v1.6.5 is identical to itself across runs. With the recipe's
  settings it differs at near-ties, as the patch always has: port vs plain v1.6.5, 1 of 4 prompts diverged (token 49);
  0f685aa build vs plain 0f685aa, 2 of 4 (tokens 230 and 398); port vs 0f685aa build, 2 of 4 (tokens 124 and 230).
- PPL, first 580 chunks: 7.0579 on the port and on plain v1.6.5, against 7.0487 on the 0f685aa build (10.5667 vs
  10.5374 with the BOS overwrite). The change comes with upstream, not the patch.
- API A/B, 64k / 1 slot (`ab-upstream/run_ab.sh`, TabbyAPI 2fd6cc7, two rounds alternating), tok/s, 0f685aa build vs
  port: code 156.2 / 156.0 vs 154.1 / 154.1; prose (sampled) 64.5 / 66.9 vs 56.5 / 60.5; decode at 8k 135.9 / 139.2 vs
  141.4 / 145.0, at 32k 127.5 / 138.1 vs 132.4 / 139.3, at 56k 122.9 / 118.3 vs 132.0 / 131.4 (the port's second
  context run was repeated after a restart cut it off); prefill even (1,259-1,270 tok/s at 30k, except the port's first run at 1,095). Peak VRAM 13.89 vs 13.74 GB. Recall correct
  everywhere, no OOM.
- TabbyAPI a3fed43 (the recipe's pin) with the port: loads, code 153.4 tok/s, prose 66.3, prefill 1,236 / 1,260 tok/s
  at 8k / 30k, no server errors.

Since 2026-10-13 the recipe built `dev` 0f685aa. turboderp merged our DeltaNet scan change and the Mamba2 replay fix
there (#505, #506), with the same kernel code as our `0016` and `0018`, so our patch is down to the other 15 commits;
the rest of what came in after 9cc3bd1 is the disk cache tier, tensor parallelism, a DSA top-k workspace fix and
conversion. The compute kernels this setup uses are unchanged and the disk tier and TP paths aren't used, though some
shared Python code changed (generator and page-table bookkeeping, a refactor of the DFlash2 selector). The numbers below were measured on 9cc3bd1 with our patches. Checks of the 0f685aa build against
it (2026-10-13): greedy output token-identical (4 x 400 tokens with DFlash2), PPL 7.0487 on both, the recurrent and
attention kernel tests pass (884 and 204, including upstream's new Mamba2 replay test), and a 3-round API A/B came out
even (code 152.4 vs 152.6 tok/s, prose 60.7 vs 60.0; the long-context runs that night spread too widely to compare,
96-124 tok/s at 64k for the same build).

The 2026-10-12 port was to exllamav3 `dev` 9cc3bd1 (v1.6.1 plus the next day's commits), with our patches ported onto it
(`patches/README.md`). Rows elsewhere in this file marked "new base", the batching section and the self-calibrated comparison
are on the previous base, 0662fac; the rest are on the old base (f1cf869). Same SC H4 model, same 128k / 3-slot server config and TabbyAPI 2fd6cc7. A/B through the API,
2026-10-12, three rounds alternating which build ran first; medians, tok/s (`bench/bench_api.py` 3x per round,
`bench/context_bench.py` 64k and 120k once per round):

| | Previous base (0662fac) | Current base (9cc3bd1) |
|---|---:|---:|
| Code, greedy, 600 tokens | 146.3 | 155.0 |
| Prose, sampled, 600 tokens | 63.5 | 62.6 |
| Prefill 8k / 30k | 1233 / 1253 | 1225 / 1251 |
| Decode at 67.6k / 127.8k | 121.7 / 103.9 | 122.2 / 106.6 |
| Prefill at 67.6k / 127.8k | 1093 / 902 | 1093 / 903 |
| VRAM after load | 14.63 GB | 14.53 GB |

Prose is sampled, and its draft acceptance swings from run to run (21-28%), so 62.6 vs 63.5 is within the spread
(57.9-65.9 vs 60.0-67.7). Every 64k and 120k run recalled the buried code. Peak VRAM during the 120k fill was 16.08-16.15
GB in the earlier rounds of the same port. PPL on the first 580 chunks is 7.0487 (7.0488 on the previous base).
Greedy output differs from the previous base at near-ties: upstream's dense row layout changes the attention
accumulation, so 2 of 4 400-token test prompts diverged, at tokens 29 and 91.

The first port was slower: 59 vs 64 tok/s on prose and ~1% on code, on both upstream snapshots we tried (10 and 12
October). Timing each phase of a verify step (`torch.cuda.synchronize()` around each, greedy code, 8-token verify)
found it in the DeltaNet layers:

| ms per step | Previous base | First port |
|---|---:|---:|
| Whole step | 43.1 | 45.2 |
| Verify forward (target model) | 36.2 | 38.9 |
| Draft forward | 2.50 | 2.46 |
| DeltaNet rewind | 1.45 | 0.88 |

The rest of the step (sampling, acceptance) was 2.9 ms in both. A second run that also synchronized around every module
(which adds overhead, so its totals don't compare with the table above) put all of the difference in the DeltaNet
layers; attention (6.1 / 6.2 ms) and the MLPs (18.7 / 18.4) were even:

| ms per step, per-module run | Previous base | First port | Current (with `0016`) |
|---|---:|---:|---:|
| GatedDeltaNet, 48 layers | 11.86 | 15.02 | 12.11 |
| DeltaNet rewind | 1.39 | 0.84 | 0.58 |

Upstream's v1.6.1 replay reads the base state on the first token and then advances a scratch row in memory, reading the
state twice and writing it once per token. Our previous replay kept the state in registers during the verify. Patch
`0016` does that for upstream's scan: each thread loads its slice once, carries it across the tokens and writes it once
at the end. The arithmetic is unchanged and the output bit-identical (4 x 400 greedy tokens with DFlash2, and the same
kernel instruction counts); the 128-wide verify kernel goes from 64 to 106 VGPRs, without spills.

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
| 1 request alone | 131-132 | ~128 (400 tokens in 3.119 s) |
| 2 at once | 118, 83 | 161 (800 tokens in 5.0 s) |
| 3 at once (two runs) | 90, 94, 68 / 90, 68, 95 | 194, 199 (1,200 tokens in 6.2 / 6.0 s) |
| 4 at once (three runs) | 88, 84, 69, 62 / 66, 83, 62, 89 / 62, 87, 84, 69 | 244, 240, 245 (1,600 tokens in 6.5-6.7 s) |
| Before the batching series (`rdna4-dev`), 2 / 4 at once | 36, 33 / 16, 18, 16, 20 | 63 / 61 (800 tokens in 12.7 s / 1,600 in 26.1 s) |
| Before the batching series, 2 at once with a 3-token draft | 58, 65 | 107 (800 tokens in 7.5 s) |
| `EXL3_DRAFT_ROW_BUDGET=16` / `24`, 4 at once | 52, 50, 54, 73 / 58, 50, 59, 55 | 196 / 194 (1,600 tokens in 8.2 s) |

Through the API, 4 at once is ~240-245 tok/s combined (239.6 / 243.6 / 244.6 across the three runs).

Generator-only (no API), a separate measurement that should not be compared with the table:
`bench/bench_batch_sanity.py` loads the model directly and times the same prompts over a common window. It gave
188.0 tok/s at 2 at once (800 tokens in 4.3 s) and 282.8 at 4 (1,600 tokens in 5.7 s), against 130.3 tok/s for
the same prompts one at a time. These leave out request handling.

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
(then set in `setup/env.sh`) and the same prior load, a gate suite (0xSero's local-ai-registry: load, chat, reasoning,
tools, context with the 106,294-token prompt recalled, speed) passed at 3 slots (peak 16.17 GB) and at 4 slots (peak
16.28 of 16.3 GB). The config defaults to 3 slots (`max_batch_size: 3`) for the extra headroom.

Expandable segments are now off (2026-10-13): on gfx1201 they corrupt memory (see "Allocator" under Limitations). With
them off and the same prior load (concurrent chats, image requests), the gate suite passed again at 3 slots on the
0f685aa build, the 106,294-token prompt recalled, no out-of-memory errors. We haven't repeated the 4-slot run without
them, so 4 slots is no longer recommended.

`EXL3_DRAFT_ROW_BUDGET` (shorter drafts when many requests are active, to stay within 16 or 24 rows) is an
experimental setting, off by default, that didn't help: through the API it was slower in every test (196 / 194 vs
~240-245 tok/s at 4 at once). 4 requests at once is the largest batch we document and test.

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
| DFlash2 5.0 bpw draft with WMMA, `EXL3_BC_GDN` (C++ DeltaNet decode, then graph-captured) off / on | 145.4 / 153.7 |
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

All 198 questions, 3.0 bpw, `medium` + anti-spiral, same sampling, through the server with 3 requests at a time
(batching build):

| Run | Correct | Hit the token cap |
|---|---|---|
| Pass 1 | 80.3% (159/198) | 1 |
| Pass 2 | 81.8% (162/198) | 1 |
| SC H4 quant (2026-10-09, see below) | 80.8% (160/198) | 4 |
| SC H4 quant, current base (2026-10-12) | 79.8% (158/198) | 4 |
| SC H4 quant, current base on 0f685aa (2026-10-13) | 82.8% (164/198) | 5 |
| SC H4 quant, the container image (aec2d311: TabbyAPI a3fed43, expandable segments off), 128k (2026-10-14) | 80.8% (160/198) | 4 |

**All 198 at `xhigh` (2026-10-13): 88.4% (175/198).** Same build, quant, prompt and sampling as the 0f685aa row, with
a 60k token cap and two requests at a time (`GPQA_EFFORT=xhigh GPQA_MAX_TOKENS=60000 GPQA_WORKERS=2`):

| On all 198 | `medium` (0f685aa row) | `xhigh` |
|---|---|---|
| Correct | 164 | **175** |
| Hit the token cap | 5 (30k cap) | 6 (60k cap), all wrong |
| Median reasoning | 5.5k characters | 18.7k characters |
| Median time per question | 36 s | 105 s |

`xhigh` got 18 of the 34 questions `medium` missed and missed 7 that `medium` got. The requests took 11.7 hours in
total, two at a time; elapsed time was 6 h 19 min, including a 21-minute pause (the client was stopped and resumed).
VRAM samples every 30 s ranged from 15,221 to 15,721 MB, and no server errors came from this run. It ran with
expandable segments on, before they were turned off. 88.4% is 1.5 points under the published BF16 score (89.9%,
settings not identical). This is one sampled run.

So `medium` scores about 81% on the full set; the first 50 questions happen to be easier than average. Both SC H4 runs answered 147
questions correctly and both missed 27; 11 and 13 went the other way, about the spread we see between two sampled passes
at temperature 1.0 (the two passes of the plain quant differ by 3), though different builds can contribute too. The current
base's run took 6% longer in total because its answers were 6% longer; per request it produced the same characters per
second (176.9 vs 177.2). It was also a 92-minute soak of the live server: VRAM rose 105 MB in the first 10 minutes and
13 MB in the 82 after, with no errors. The 0f685aa build's run (164) is within the same spread: against the 9cc3bd1
build, 145 questions were right in both and 19 and 13 went the other way. Its server ran 88 minutes without errors,
VRAM between 15.2 and 16.1 GB with no upward drift (16.09 GB at 10 minutes, 15.55 at the end); the client was
interrupted once at question 85 and resumed. Published results on all
198 questions (ISTA, settings not identical to ours): BF16 89.9%, UD-IQ3_S 89.9%, UD-Q2_K_XL 86.9%.

To see how much of that gap is the `medium` effort, we asked the 48 questions that either pass got wrong again, at
`xhigh` with the same anti-spiral prompt and sampling, a 60k token cap and two requests at a time (2026-10-07, same server;
`GPQA_IDS`, `GPQA_EFFORT=xhigh`, `GPQA_MAX_TOKENS=60000`, `GPQA_WORKERS=2`):

| On those 48 questions | Correct | Hit the token cap | Median reasoning | Median time |
|---|---|---|---|---|
| `medium`, pass 1 | 9 | 1 | 19k characters | 105 s |
| `medium`, pass 2 | 12 | 1 | 17k characters | 105 s |
| **`xhigh`** | **28** | 8 | 98k characters | 529 s |

`xhigh` got 12 of the 27 questions that `medium` missed in both passes, and 16 of the 21 it missed in one. If
`xhigh` also got right the 150 questions both `medium` passes did, the full set would be 178/198 (89.9%). That's an
upper estimate, not a measured `xhigh` score: those 150 weren't asked at `xhigh`, and questions picked because
`medium` missed them flatter the gain (on a third pass `medium` would get some of them right by chance). The rerun
also changed the token cap (60k vs 30k), and we haven't run the unquantized model, so this doesn't show how much of
the gap to the published scores is the reasoning effort and how much the quantization. It does show that `medium`
leaves accuracy on the table. The cost is time: about 5x as much reasoning per hard question, and 8 of the 48 still
ran into the 60k cap. (The times are wall clock with other requests in flight, three for `medium` and two for
`xhigh`.)

### Self-calibrated 3.0 bpw (the default since 2026-10-09)

turboderp's `SC_3.00bpw_H4` branch is a 3.0 bpw quant calibrated on the model's own output, with a 4-bit instead of a
6-bit lm_head. Against the plain 3.00bpw quant, same server and settings:

| | Plain 3.00bpw | SC_3.00bpw_H4 |
|---|---|---|
| Weights on disk | 13.8 GB | 13.4 GB |
| VRAM after load (128k, 3 slots, `amd-smi`) | 15,395 MB | 15,151 MB |
| Perplexity (below) | 7.063 | 7.049 |
| KL divergence from BF16 (turboderp's model card) | 0.033 | 0.026 |
| GPQA Diamond, 198 q, `medium` + anti-spiral, 3 at a time | 159 / 162 (two passes) | 160 |
| Hit the 30k token cap | 1 / 1 | 4 |
| Code, 600 greedy tokens (`bench_api.py`, two runs) | 151.7 / 150.5 tok/s | 147.0 / 145.8 tok/s |
| Prose, 600 sampled tokens (two runs) | 63.6 / 62.6 tok/s | 69.5 / 63.1 tok/s |
| Decode at 8k / 64k / 120k (`context_bench.py`, one run each) | 138.0 / 118.4 / 107.5 | 128.2 / 127.7 / 109.0 |
| Prefill at 8k / 30k (first of the two runs) | 1,234 / 1,253 tok/s | 1,242 / 1,265 tok/s |

Greedy code is about 3% slower with the SC quant (146.4 vs 151.1 tok/s, both runs lower). Prose and the
`context_bench.py` depths are too noisy here to call either way: prose is sampled, and single runs at one depth have
differed by 10-25 tok/s (at 64k on one build, 102.0 to 127.4 tok/s across three runs). Repeated runs on 2026-10-10,
SC quant on this build: 144.6 tok/s code and 60.3 prose (averages of 6). On GPQA the two scores are within sampling
noise of each other: against pass 1, 11 questions were right only with the SC quant and 10 only with the plain one
(against pass 2, 11 and 13), so neither has a shown advantage. It hit the token cap 4 times to 1; with numbers that
small that may be chance. We took the closer distribution (perplexity, and turboderp's KL divergence on his own eval
set) and 244 MB less VRAM over the 3% on code, so the SC quant is the default now
(`setup/download_models.sh 3.0bpw-plain` still fetches the plain one).

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
| EXL3 3.0 bpw, SC_3.00bpw_H4 | 7.049 |
| EXL3 3.0 bpw | 7.063 |
| EXL3 2.5 bpw | 7.248 |
| GGUF UD-Q3_K_XL (llama.cpp) | 7.068 |

### Vision

The model's vision encoder runs from system RAM (`vision_offload: true`), which costs about 20 ms per image. Keeping
it in VRAM doesn't fit at 128k. Images lower the DFlash2 draft's acceptance rate, so expect decode to be roughly
10-25% slower in chats with images.

## What the patches do

`patches/exllamav3-rdna4.patch` is the combined patch the build applies; `patches/features/` has the same work as 18
commits for reading, and `patches/README.md` explains both. The runtime switches are set in `setup/env.sh`. Some of the
work below was only in the previous base's patch (`patches/previous/`), because upstream has since shipped its own
version; each such item says so.

**gfx12 WMMA multi-row GEMV** (`EXL3_GEMV_WMMA=1`). Speculative decoding verifies 2-8 draft tokens at once, and the
existing RDNA kernel for that ran at ~1.55x the cost of a 1-row decode. The new kernel decodes the trellis once per
tile with the tuned RDNA decoder, stages it through a per-wave LDS slot, and does all rows with one
`v_wmma_f32_16x16x16_f16`; the register layout was probed on the card (`kernels/wmma_layout_probe.hip`). An 8-row
verify now costs ~1.16x a 1-row decode. In the generator alone, DFlash2 went from 127.1 to 145.1 tok/s and MTP from
82.4 to 90.2; through the API, 136.8 to 145.4. Against the old kernel on the full model: top-1 agreement 100%,
KL ~3e-5. WMMA only runs on gfx1200/gfx1201.

**HIP graphs off** (upstream 96838c9: backported in the previous base, part of the current one; `EXL3_GRAPHS=1` turns
them back on). ExLlamaV3 captures each decode
kernel sequence into a HIP graph and patches its arguments before every replay. In this HIP runtime,
`hipGraphExecKernelNodeSetParams` writes the new arguments into fresh kernel-argument space that isn't reclaimed until
the graph exec is destroyed (ROCm/rocm-systems#10713; `kernels/graph_setparams_repro.hip` reproduces it), so a server
eventually segfaults inside it. Our first fix re-instantiated each graph every `EXL3_GRAPH_REINST` updates. With the
patch's default of 100k the live server still crashed after 16-23 min of sustained 3-request load, and at 10k after
~5.5h of 2-3 request load; 1000 then saw no crash in ~11 h of intermittent, lighter use (with restarts in between),
which isn't a sustained-load soak. turboderp then found that each exec also reserves at least 2 MB of kernel-argument
memory up front (about 1.2 GB over the hundreds of execs a speculative-decoding server holds, by his measurement), and
that a graph launch on ROCm is no faster than launching its kernels one by one, so upstream turned graphs off on ROCm.
With them off, every graphed site still runs the same C++ kernel sequence, launched directly, and the re-instantiation
code is never reached.

Same build, 128k / batch-3 server config, greedy 400-token code requests, 2026-10-07 (`wt-batch` vs the same plus
96838c9; one warm-up, then 3 single and 2 three-at-once runs per build, about 20 s of requests each). Decode tok/s per
request from the server log:

| | Graphs on (`EXL3_GRAPH_REINST=1000`) | Graphs off |
|---|---|---|
| 1 request (3 runs) | 133.6, 137.2, 136.7 | 136.6, 137.2, 136.9 |
| 3 at once (2 runs) | 97.6, 92.1, 69.5 / 97.5, 95.3, 70.5 | 98.2, 93.6, 70.4 / 98.1, 93.5, 70.4 |
| Greedy final answers (last run of each) | | identical to graphs on (1 request: 1 answer; 3 at once: 2 answers, the third spent all 400 tokens thinking in both) |
| VRAM in use after these runs (`rocm-smi`) | 16.06 GB | 15.98 GB |

Only the final answers were saved, not the reasoning, so this compares answers rather than every token; the kernels and
their order are the same either way. The VRAM difference is small here, probably because so short a run creates few
graph execs and updates (we didn't count them); the leak's growth takes hours. The older "graphs on" gains in these
notes (the `EXL3_BC_GDN` off / on row above, 145.4 vs 153.7, and the eager vs graph DeltaNet replay) switched between
the C++ path and the Python-orchestrated one, which `EXL3_BC_*=0` selects, so they changed both the path and graph use
at once and don't isolate a graph gain. The C++ path still runs with graphs off, and this newer comparison shows no
measurable graph benefit.

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

**DeltaNet accepted-input replay** (previous base only, `EXL3_GDN_REPLAY=1`; upstream's v1.6.1 replay replaces it in the
current base, see "Current base" above). Qwen3.8's 48 Gated DeltaNet
layers keep a recurrent state, and speculative decoding normally stores a snapshot of it per draft token so it can
rewind (~0.15 GB per draft token). With replay, the verify pass reads the committed state without writing it and
records the small post-projection inputs; on acceptance it replays only the accepted prefix. The verify runs inside the
captured graph and the replay commit is two launches per step. In the `test_gdn_replay.py` A/B (379-token prompt, 400
greedy tokens), allocated VRAM went from 13.20 to 12.14 GB and decode from 89.3 to 92.4 tok/s, because the verify no
longer writes 8 state snapshots per layer per step. The state is bit-identical to the snapshot path for every accepted
length (712 kernel-level comparisons), and the greedy output is identical. That 1.06 GB is what lets the DFlash2
profile go from 96k to 128k. The test, `test_gdn_replay.py`, ships in the previous base's patch; run it from a checkout built with that patch. The idea comes from
TensorFold.

**Batching series.** The gfx12 WMMA GEMV runs 16 and 32 rows (two M tiles), so a batched verify of up to 4 requests
x 8 rows stays on the fast kernel; the GatedMLP runs as a captured graph up to 32 rows, with graphs keyed by row count
and output dtype; LoRA adapters and per-projection controls take the unfused path; DFlash2 skips the head and selector
rows a short draft doesn't use. Effect, through the API: 4 requests at once go from 61 to ~240-245 tok/s combined
(see Batching above). The DFlash2 row skipping and the opt-in `EXL3_DRAFT_ROW_BUDGET` (off by default because it was
slower) were only in the previous base's patch; they clashed with upstream's new hybrid draft mode.

**DeltaNet scan in registers** (upstream since 0f685aa; `0016` on 9cc3bd1). Upstream's scan read and wrote the
recurrent state in memory for every token of a pass; each thread now keeps its slice in registers and writes it once. GatedDeltaNet 15.0 -> 12.1
ms per verify step, bit-identical output (see "Current base" above). `0018` fixed upstream's Mamba2 replay launch, which
picked a v split that Mamba2 has no kernel for (Qwen3.8 doesn't use Mamba2 layers). We filed both as issues and
turboderp merged them (#505, #506).

**Split-decode stage fallback** (current base, `0014`, `0015`). With `EXL3_DEC_STAGES=2` and the Q4W kernel, the
128-row tile at head_dim 256 (16 q heads per kv head, an 8-token verify) needs 67.5 KB of shared memory, more than the 64
KB on gfx1201, and the launch failed instead of falling back. It now retries with one stage, and the cached launch pick
is keyed on the single-split variant, whose footprint differs. Qwen3.8 (6 q heads per kv head) never reaches that tile.

**Optional: PR #423, DFlash2 rejection sampling** (by Rafa, [@rafatxf](https://github.com/rafatxf); off by default,
previous base only, `WITH_PR423=1 setup/build_exllamav3.sh` opts in there). This is an open upstream PR. For sampled requests, the draft path is
sampled from the drafter's own distribution and accepted with probability min(1, p/q), instead of only accepting draft
tokens that match a sample from the target. Greedy decoding doesn't use it. Our review found four problems (token masks
skipped, a top-k tie mismatch, calibration silently off, extra syncs); Rafa fixed all four in a second commit, which
the patch includes.

On the GPU, both commits on the batching series, `EXL3_DFLASH_SPEC=1` vs `0` on the same build: the PR's GPU tests pass;
six sampled prose requests capped at 500 tokens went from 52.6 to 53.2 tok/s end to end (acceptance 20.3% -> 21.5%), one
run each, so not an established speedup; `min_tokens = 64` holds; greedy output identical. Details in
`patches/README.md`.

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

- `EXL3_QC_STAGING=0` on the old base: the model failed to load. On the current base it loads and frees ~0.2 GB at
  128k for slower long-prompt prefill ("What 128k needs, and what lowers it").
- 2-bit KV cache for the draft: slower in our runs, and we didn't keep a clean A/B.
- K4/V3 KV cache for the main model on the old base (96k, 2-bit draft cache in both): 110.0 vs 118.1 tok/s at 8.1k and
  53.2 vs 66.1 at 97.7k against Q4, and no VRAM headroom gained. On the current base at 128k it gains room (above).
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
  ~240-245 without it. It was in the previous base's patch as an experimental setting, off by default.
- TensorFold itself has no working AMD path (its PR144 gives NaN logits on gfx1201). We ported the replay idea instead.
- 2.5 bpw to free memory: costs too much accuracy (above).

## Known limitations

- Most single-request tables were measured on the old base (exllamav3 f1cf869, TabbyAPI f07131c); the batching results
  and the "new base" rows are on the previous base (0662fac). Only "Current base", the profile rows and GPQA rows marked
  current base were measured on the current one (9cc3bd1 with our scan patch, or 0f685aa where it says so).
- Speculative decoding isn't bit-identical to plain greedy decoding. Every speculative mode, plain DFlash2 included,
  diverged from non-speculative greedy output at the same token (375 of 400) in our test. That comes from the
  multi-row verify's accumulation order, not from the replay code.
- On the previous base, replay was only tested with the DFlash2 draft and the MTP profiles turned it off. On the current
  base upstream's replay is always on, MTP included; concurrent requests with MTP are untested.
- The WMMA GEMV only runs on gfx1200/gfx1201. Everything was tested on one RX 9070 XT; the RX 9070 and RDNA3 cards are
  untested.
- Concurrency is `max_batch_size: 3`. 4 passed the long-context gate only with expandable segments, which are now off
  (next item), so it's untested in the current setup.
- Allocator: on gfx1201, PyTorch's expandable segments (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) corrupt
  memory. They map memory through HIP's virtual memory API, and after a remap at the same address written data can be
  lost (ROCm/TheRock#8744, pytorch/pytorch#195202). AMD root-caused it on 2026-10-13 in the `amdgpu` kernel driver, not
  ROCm: with all VRAM visible through Resizable BAR, the driver updates the page tables from the CPU and then skips the
  GPU TLB flush after an unmap, so a later map at the same address writes into the old pages. The fix is in review on
  the amd-gfx list (https://lore.kernel.org/amd-gfx/20261008124232.23825-1-christian.koenig@amd.com/), so it arrives
  with a kernel update, and expandable segments can come back once it does (they would help batch 4 fit). Fedora's
  7.2.9-201 kernel (2026-10-09, with an unrelated gmc12 TLB `flush_type` fix) doesn't have it: the HIP-only loop still
  lost data in 150 of 300 rounds. Our HIP-only loop (map, fill, sync, check)
  lost data in 19-99% of rounds depending on the variant (final version of the probe), on ROCm 7.1.1 and 7.2.4, and
  never with `hipMalloc`. In ExLlamaV3, with it on, the recurrent kernel tests failed in 3 of 10 full runs and upstream's
  spec-rewind e2e test failed 7 of its 8 cases in 3 of 45 runs; with it off, 0 of 10 and 0 of 45. `setup/env.sh` turns
  them off; we had turned them on to avoid fragmentation (see Batching).
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
patches/   exllamav3-rdna4.patch, features/ (the same as 15 commits), previous/ (0662fac base and PR #423)
bench/     speed, context, batching, GPQA and kernel tests (bench/README.md)
kernels/   standalone HIP experiments: WMMA layout probe, HIP graph repro, hand-written attention
notes/     measurements.md (this file), research-log.md (the working notes), reviews.md (the code reviews)
```

The setup scripts create `.venv/`, `exllamav3/`, `tabbyAPI/`, `deps/`, `models/` and `logs/` in the repo root; all
are git-ignored.
