# Qwen3.8-27B on a 16 GB RX 9070 XT: ExLlamaV3 + TabbyAPI

Run Qwen3.8-27B (EXL3, 3.0 bpw) on an AMD RX 9070 XT or RX 9070 under Linux, as an OpenAI-compatible server with a
128k context, several chats at once and image input. It uses upstream
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [TabbyAPI](https://github.com/theroyallab/tabbyAPI) with
our RDNA4 patches on top, and a DFlash2 draft model for speculative decoding.

This is an independent project, not affiliated with or endorsed by ExLlamaV3, TabbyAPI, DFlash2 or Qwen. It has been
tested on one machine:

| | |
|---|---|
| GPU | AMD RX 9070 XT, 16 GB (gfx1201) |
| OS | Fedora 44, kernel 7.2 |
| ROCm | system ROCm 7.1.1 (Fedora packages) to build; PyTorch 2.13.0 + ROCm 7.2 wheels to run |
| Upstream | ExLlamaV3 `dev` 0662fac, TabbyAPI `main` 2fd6cc7 |

## What we added on top of upstream

- **WMMA GEMV for draft verification (gfx12):** checking 8 drafted tokens costs about 1.16x one normal decode step
  (was 1.55x), so speculative decoding pays off.
- **Batching up to 32 rows:** several chats decode together at ~300 tok/s combined instead of ~70.
- **Attention tuned for Qwen's grouped heads, plus a 4-bit cache kernel:** faster decode at long context.
- **DeltaNet replay:** about 1 GB less VRAM, which is what makes 128k fit next to the draft model.
- **HIP graph re-instantiation:** fixes a crash after about an hour of serving, so HIP graphs can stay on.
- **3.0 bpw DFlash2 draft (requant script):** 0.5 GB smaller than the 5.0 bpw quant, same speed.
- **Setup scripts:** pinned versions, missing headers fetched without sudo, one command each to build and serve.

## Expected speed

Typical ranges on an RX 9070 XT; they depend on the content.

| Workload | Speed |
|---|---|
| One chat, code | ~130 tok/s at short context, falling to ~100 at 120k |
| One chat, prose or conversation (sampled, temperature ~0.8) | ~55-65 tok/s (the draft is accepted less often on prose) |
| With images | roughly 10-25% slower than the same chat as text |
| Several chats at once (measured with 4), code | ~300 tok/s combined, ~60-90 each; prose is lower |

Prompt processing runs at about 1,300 tok/s for short prompts and 900 at 128k, so a full 128k prompt takes about
2.5 minutes before the first token.

## Requirements

- AMD RX 9070 XT or RX 9070 (gfx1201, 16 GB). Only the 9070 XT has been tested.
- Fedora 44 with its ROCm packages (`rocm-hip-devel rocm-runtime-devel hipcc rocm-clang-devel rocm-llvm-devel
  rocm-comgr-devel rocm-libc++-devel rocblas-devel hipblas-devel hipblas-common-devel rocminfo rocm-smi`), plus
  `sudo dnf install gcc gcc-c++ python3.12 git curl`.
- Your user in the `render` group (access to `/dev/kfd` and `/dev/dri/renderD*`).
- About 20 GB of disk for the models and 15 GB for the Python environment.
- Nothing else on the GPU while serving: the server uses almost all of the 16 GB.

## Install

From the repo root:

```bash
setup/fetch_deps.sh          # missing -devel headers into deps/ (no sudo; or dnf install them, see the script)
setup/make_venv.sh           # .venv/ with PyTorch 2.13 for ROCm 7.2 and the tested package versions
setup/build_exllamav3.sh     # clone ExLlamaV3 at the pinned commit, apply our patch, compile for gfx1201
setup/install_tabby.sh       # clone TabbyAPI at the pinned commit, write tabbyAPI/config.yml
setup/download_models.sh     # Qwen3.8-27B EXL3 3.0 bpw (13.8 GB) + the DFlash2 draft source (3.8 GB)
setup/requant_draft.sh       # turn the draft into a 3.0 bpw EXL3 model (uses the GPU, a few minutes)
```

The scripts are safe to rerun. If a checkout has local changes they stop and say so; `RESET=1` starts it over.

## Run

```bash
serve/run_tabby.sh           # http://127.0.0.1:8096/v1, 128k context
serve/stop_tabby.sh
```

TabbyAPI writes a random API key to `tabbyAPI/api_tokens.yml` on first start. Send it as
`Authorization: Bearer <key>`:

```bash
KEY=$(awk '/^api_key:/{print $2}' tabbyAPI/api_tokens.yml)
curl -s http://127.0.0.1:8096/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "x", "messages": [{"role": "user", "content": "Write a haiku about VRAM."}], "max_tokens": 400}'
```

Settings live in `tabbyAPI/config.yml` (copied from `serve/config.example.yml`). Up to 3 chats run at once
(`max_batch_size: 3`); 4 also works, with less memory headroom. The server only listens on 127.0.0.1; to use it from other
devices over Tailscale, set `network.host` to the machine's Tailscale IP. `serve/open-webui.md` sets up a chat UI.

Two other profiles use the model's built-in MTP head instead of the draft model: `serve/run_tabby_long.sh` (128k, no
draft model needed) and `serve/run_tabby_xl.sh` (160k, needs the whole GPU). Both are slower than the default.

## Details and measurements

Full benchmark tables and methodology, what each patch does, what didn't work and the known limitations are in
[notes/measurements.md](notes/measurements.md). The patches are described in [patches/README.md](patches/README.md),
the benchmark scripts in [bench/README.md](bench/README.md), and the code reviews in
[notes/reviews.md](notes/reviews.md).

## Credits and licences

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp (MIT), and turboderp's EXL3 quants of
  Qwen3.8-27B.
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI) by theroyallab (AGPL-3.0), used unmodified.
- DFlash2 by [z-lab](https://github.com/z-lab/dflash) / [inco.ai](https://inco.ai/blog/dflash2/): the draft model
  [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (Apache-2.0). Mia-AiLab for the first
  EXL3 quant of it.
- Rafa ([@rafatxf](https://github.com/rafatxf)) for DFlash2 rejection sampling, exllamav3 PR #423, included as an
  optional patch (off by default; see `patches/README.md`).
- TensorFold, for the idea behind the DeltaNet replay.
- The Qwen team, for Qwen3.8-27B.

Our scripts, configs and docs are MIT (`LICENSE`). The patches modify ExLlamaV3 and are under its MIT license. Models
keep their own licences. Every change here was written by Claude Opus 5.5 and reviewed by GPT-6.1 Sol.
