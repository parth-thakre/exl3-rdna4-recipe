# Benchmarks and checks

Run everything from the repo root with the venv active (`source setup/env.sh`). Results go to `logs/` and `gpqa/`.

## Against the running server

These talk to TabbyAPI over its OpenAI-compatible API. They default to `http://127.0.0.1:8096/v1` and the first
`api_key` in `tabbyAPI/api_tokens.yml`; set `TABBY_URL` and `TABBY_API_KEY` (or `TABBY_TREE` for another TabbyAPI
checkout) to point elsewhere. Every script exits nonzero if a request fails or returns no text. They count tokens with the
model's tokenizer (`models/Qwen3.8-27B-EXL3-3.0bpw/tokenizer.json`, or `MODEL_DIR`).

| Script | What it does |
|---|---|
| `bench_api.py NAME [URL] [KEY]` | Greedy code (600 tokens), sampled prose (600 tokens), prefill at 8k and 30k. Engine-neutral: works against llama-server too. |
| `context_bench.py LABEL 8 32 60 ...` | Decode speed and recall by depth: a fresh filler document per depth with a code buried mid-way, then ~450 tokens of Python. This is the decode-by-depth table. |
| `ctx_probe.sh LABEL CTX_K [args]` | Restarts TabbyAPI with a context size and extra args, then runs `context_bench.py` near the top. Reports VRAM via `amd-smi`. Stops this repo's TabbyAPI first (`serve/stop_tabby.sh`), gives up after `START_TIMEOUT` seconds, and stops the server it started if anything fails. |
| `needle_test.py [N]` | Quick long-context recall check: a code buried in N filler sections (default 1400). |
| `gpqa_eval.py NAME [URL] [KEY]` | GPQA Diamond, first 50 questions (`GPQA_LIMIT`), thinking on, 30k token cap, temperature 1.0. Resumable. `GPQA_EFFORT=medium GPQA_SYSTEM=antispiral` gives the recommended settings. |
| `run_gpqa_all.sh` | Starts a server per model (3.0 bpw, then 2.5 bpw if present), checks it serves that model, runs `gpqa_eval.py`, stops it. Refuses to start if something already answers on `TABBY_URL`. |
| `gpqa_resume.sh` | Restarts `run_gpqa_all.sh` after a crash or reboot. |
| `overthink_test.py VARIANT` | The 11 GPQA questions that spiralled, under one effort/system-prompt variant. This is how the anti-spiral prompt was picked. |

### GPQA data

GPQA is gated on Hugging Face and its authors ask that it not be reposted in plain text, so it isn't in this repo.
Accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa, log in with `hf auth login`, then:

```bash
hf download Idavidrein/gpqa gpqa_diamond.csv --repo-type dataset --local-dir gpqa
```

## Direct model and kernel tests (GPU, no server)

These load the model or call kernels directly, so stop TabbyAPI first (`serve/stop_tabby.sh`); there isn't room for
two copies in 16 GB.

| Script | What it does |
|---|---|
| `smoke_test.py [--draft DIR \| --mtp] [--long]` | Load, generate, print TTFT and decode speed. The first thing to run after a build. |
| `test_wmma_model.py` | Full-model check of `EXL3_GEMV_WMMA`: logits of an 8/4/2-token forward with and without it (max logit diff, KL, top-1 agreement). |
| `test_wmma_gemv.py` | WMMA vs fdot2 multi-row GEMV per layer shape: relative error and µs per call. |
| `bench_gemv.py [--rows 1 8]` | Achieved weight bandwidth of each EXL3 linear (µs and GB/s, % of 640 GB/s). |
| `bench_decode_attn.py [depth_k] [q_len]` | Split-decode attention on a synthetic Q4 cache, over the `EXL3_DEC_*` launch configs; checks every output against the default. |
| `bench_batch_sanity.py [--bsz 2 4]` | Runs N different prompts at once and checks each output against the same prompt run alone: no garbage (also after the point where it diverges), no cross-talk between batch rows, all N actually decoding together. Fails on any generator error or unfinished job. Prefix and "follows" columns are informational. |
| `vram_breakdown.py` | Allocated VRAM per component (draft, main weights, KV) and the biggest tensors. |

The DeltaNet replay A/B test (`test_gdn_replay.py`) ships inside the exllamav3 patch. It loads the main model and the
3.0 bpw draft from `models/` and compares greedy output with replay off and on; `--state-check-only` runs just the
kernel-level state comparisons without loading models:

```bash
source setup/env.sh && cd exllamav3 && python test_gdn_replay.py --state-check --state-channelwise
```

It checks that a venv is active and is the one running it, that `HIP_VISIBLE_DEVICES` names a single GPU, and that
`Python.h` is reachable, and it needs the extension built in place in that checkout (`setup/build_exllamav3.sh` does
that).
