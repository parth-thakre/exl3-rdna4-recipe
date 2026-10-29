# Qwen3.8-27B on a 16 GB RX 9070 XT: ExLlamaV3 + TabbyAPI

Run Qwen3.8-27B (EXL3, 3.0 bpw: turboderp's self-calibrated `SC_3.00bpw_H4` quant) on an AMD RX 9070 XT or RX 9070
under Linux as a local OpenAI-compatible server for coding agents and chat. It uses upstream [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and
[TabbyAPI](https://github.com/theroyallab/tabbyAPI) with our RDNA4 patches on top, and a DFlash2 draft model for
speculative decoding.

## What you get

- **Code, one request:** ~150 tok/s at short context, ~105 tok/s at 120k.
- **Prose and conversation** (sampled, temperature ~0.8): ~55-65 tok/s; the draft is accepted less often on prose.
- **96k-token context by default, 128k optional.** 96k leaves room for a desktop on the same card; 128k
  (`CTX=128k`) needs the card to itself or a light desktop (it passes with ~1.1 GB of VRAM already in use and doesn't
  load with ~1.9 GB). Prompt processing runs at about 1,100-1,250 tok/s for short prompts and 900 at 128k, so a full 128k
  prompt takes about 2.5 minutes before the first token.
- **Accuracy:** GPQA Diamond 88.4% (175/198) at `xhigh` effort (one run), about 81% at `medium` (158-164 of 198
  over four runs; how in [notes/measurements.md](notes/measurements.md#accuracy-gpqa-diamond)). A third-party run of
  the BF16 model scored 89.9%, with settings not identical to ours.
- **Tool calling and image input** work through the OpenAI API. Chats with images decode roughly 10-25% slower.
- **Several requests:** handles 3 requests at once (e.g. agent subagents or parallel tool calls) instead of queuing
  them (`max_batch_size` in `tabbyAPI/config.yml`). They share the one context: a single request can use all 96k, or
  several split it.
- **Setup:** one line to install and run (a container, any distro, Omarchy included), or a handful of scripts for a
  host install.

Speeds are typical for an RX 9070 XT and depend on the content.

This is an independent project, not affiliated with or endorsed by ExLlamaV3, TabbyAPI, DFlash2 or Qwen. It has been
tested on one machine:

| | |
|---|---|
| GPU | AMD RX 9070 XT, 16 GB (gfx1201) |
| OS | Fedora 44, kernel 7.2 (install also tested in Ubuntu 24.04 and Arch containers, see Requirements) |
| ROCm | system ROCm 7.1.1 (Fedora packages) to build; PyTorch 2.13.0 + ROCm 7.2 wheels to run |
| Upstream | ExLlamaV3 `dev` 0f685aa (v1.6.1 plus two days of commits), TabbyAPI `main` a3fed43 |

## What we added on top of upstream

- **WMMA GEMV for draft verification (gfx12):** checking 8 drafted tokens costs about 1.16x one normal decode step
  (was 1.55x), so speculative decoding pays off.
- **Batched verification up to 32 rows:** the WMMA GEMV and a fused C++ MLP path keep 16- and 32-row verify batches
  on the fast path.
- **Attention tuned for Qwen's grouped heads, plus a 4-bit cache kernel:** faster decode at long context.
- **DeltaNet scan with its state in registers (now upstream):** upstream's DeltaNet replay (v1.6.1, about 1 GB less
  VRAM, which is what makes 128k fit next to the draft model) re-read and re-wrote the recurrent state for every drafted
  token; keeping it in registers cuts the DeltaNet layers' time per verify step by a fifth (15.0 to 12.1 ms), with
  bit-identical output. turboderp merged it into `dev` (0f685aa, from our issue #505), so it's no longer in our patch.
- **3.0 bpw DFlash2 draft:** 0.5 GB smaller than the 5.0 bpw quant, same speed;
  [on Hugging Face](https://huggingface.co/parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw), or make it with
  `setup/requant_draft.sh`.
- **Setup scripts:** pinned versions, missing headers fetched without sudo, one command each to build and serve.

## Quick start: container (any distro, Omarchy)

With Docker or Podman and the in-kernel amdgpu driver, nothing else to install (no ROCm, no Python):

```bash
curl -fsSL https://raw.githubusercontent.com/parth-thakre/exl3-rdna4-recipe/main/install.sh | bash
```

That puts the recipe in `~/.local/share/exl3-rdna4-recipe`, downloads the models once (~14 GB) and serves
http://127.0.0.1:8096/v1 (a chat page at `/ui`). Run it again to update. If you'd rather read the script first, the
same thing by hand:

```bash
git clone https://github.com/parth-thakre/exl3-rdna4-recipe && cd exl3-rdna4-recipe
container/run.sh             # downloads the models once (~14 GB), then serves http://127.0.0.1:8096/v1
```

On Omarchy, Docker is already there, so that's all (tested so far with Podman on Fedora; the Docker path not yet). [`container/README.md`](container/README.md) has the details, the
plain `docker run` command and how to build the image. The rest of this README is the host install, which is what the
measurements were taken on (the container builds the same code with the same scripts).

## Requirements (host install)

- AMD RX 9070 XT or RX 9070 (gfx1201, 16 GB). Only the 9070 XT has been tested.
- Linux with a ROCm HIP SDK (compiler, HIP headers and the math libraries' headers); ROCm 7.1.1 and 7.2.4 tested. The
  server runs on Fedora 44 with its own ROCm 7.1.1 packages. The install was also tested end to end in Ubuntu 24.04 and
  Arch containers with ROCm 7.2.4 in `/opt/rocm`, on the same GPU: same greedy output as the Fedora build
  (`notes/research-log.md`). The scripts find ROCm through `hipconfig` (or set `ROCM_PATH`), and
  `setup/fetch_deps.sh` checks the headers and prints the missing packages for your distro.
  - Fedora: `sudo dnf install rocm-hip-devel rocm-runtime-devel hipcc rocm-clang-devel rocm-llvm-devel rocm-comgr-devel
    rocm-libc++-devel rocblas-devel hipblas-devel hipblas-common-devel rocminfo rocm-smi gcc gcc-c++ python3.12 git curl`
    (`setup/fetch_deps.sh` fetches the remaining -devel headers without sudo)
  - Ubuntu 24.04: AMD's ROCm apt repository (Ubuntu's own ROCm packages are too old), then
    `sudo apt install rocm-hip-sdk rocm-smi-lib gcc g++ python3.12-venv python3.12-dev git curl`
  - Arch: `sudo pacman -S rocm-hip-sdk rocm-smi-lib hipblaslt hipsparselt gcc git curl uv`, then Python 3.12 from uv:
    `uv python install 3.12` and `export PYTHON=$(uv python find 3.12)` before `setup/make_venv.sh`
  - Other distros: the same packages under your distro's names; Python 3.12 from uv if the distro has a newer one.
- Python 3.12, the version we tested.
- Your user in the `render` group (access to `/dev/kfd` and `/dev/dri/renderD*`).
- About 20 GB of disk for the models and 15 GB for the Python environment.
- Nothing else on the GPU while serving: the server uses almost all of the 16 GB.

## Install

From the repo root:

```bash
setup/fetch_deps.sh          # check the ROCm and Python headers; on Fedora fetch missing ones into deps/ (no sudo)
setup/make_venv.sh           # .venv/ with PyTorch 2.13 for ROCm 7.2 and the tested package versions
setup/build_exllamav3.sh     # clone ExLlamaV3 at the pinned commit, apply our patch, compile for gfx1201
setup/install_tabby.sh       # clone TabbyAPI at the pinned commit, write tabbyAPI/config.yml
setup/download_models.sh     # Qwen3.8-27B EXL3 3.0 bpw SC H4 (13.4 GB) + the 3.0 bpw DFlash2 draft (0.9 GB)
```

The draft is our 3.0 bpw EXL3 quant of inco.ai's DFlash2 draft,
[published on Hugging Face](https://huggingface.co/parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw). To make it yourself
instead: `setup/download_models.sh draft-bf16`, then `setup/requant_draft.sh` (uses the GPU, a few minutes).

The scripts are safe to rerun. If a checkout has local changes they stop and say so; `RESET=1` starts it over.

### Windows

Not tested. ROCm under WSL2 reportedly supports the RX 9070 / 9070 XT, so the Ubuntu 24.04 instructions above inside
WSL2 are the likely route; native Windows has open gfx1201 bugs in ROCm (silent zero results, failing kernels) and we
haven't tried it. Reports welcome.

## Run

```bash
serve/run_tabby.sh           # http://127.0.0.1:8096/v1, 96k context
CTX=128k serve/run_tabby.sh  # 128k: nothing else (or only a light desktop, ~1 GB) on the GPU
serve/stop_tabby.sh
```

TabbyAPI writes a random API key to `tabbyAPI/api_tokens.yml` on first start. Send it as
`Authorization: Bearer <key>`:

```bash
KEY=$(awk '/^api_key:/{print $2}' tabbyAPI/api_tokens.yml)
curl -s http://127.0.0.1:8096/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "x", "messages": [{"role": "user", "content": "Write a haiku about VRAM."}], "max_tokens": 400}'
```

Settings live in `tabbyAPI/config.yml` (copied from `serve/config.example.yml`). The server only listens on
127.0.0.1; to use it from other devices over Tailscale, set `network.host` to the machine's Tailscale IP.
TabbyAPI has a simple chat page at http://127.0.0.1:8096/ui/ (the startup log prints a link that logs you in);
`serve/open-webui.md` sets up a fuller chat UI.

Two other profiles use the model's built-in MTP head instead of the draft model: `serve/run_tabby_long.sh` (128k, no
draft model needed, same VRAM limit as `CTX=128k`) and `serve/run_tabby_xl.sh` (160k context). Both are slower than the default.

## If something doesn't work

`./triage.sh` collects what's needed to debug it (distro, kernel, GPU access, container runtime, recipe version, the
last server and build logs, with anything that looks like an API key removed) and opens Claude Code, Codex or
OpenCode, whichever you have, with it. `./triage.sh "what you saw"` adds your own description; `./triage.sh --print`
just writes the file for you to paste anywhere. `container/run.sh` offers it when it fails.

### GPU hangs and wrong output

A GPU hang resets the card: the server log shows `HW Exception ... GPU Hang`, the kernel log `amdgpu ... reset`, and a
desktop on the same card restarts. Wrong output usually looks like a run of `!` (what greedy decoding picks when the
logits are NaN) with none of the draft accepted. Neither has happened on the test machine, whose display runs on the
CPU's integrated GPU, so its RX 9070 XT runs nothing but this. If either happens to you, these tests narrow it down.
Hangs come and go, so give each test a few tries, and change one thing at a time:

1. **The patch's kernels at upstream's defaults:** `EXL3_TUNING=0 container/run.sh` (or `EXL3_TUNING=0
   serve/run_tabby.sh`). Decode is slower, and greedy output is the same as plain upstream's, token for token. If the
   problem stops, it's in our code.
2. **Nothing else on the card:** plug the monitor into the motherboard (most Ryzen 7000 and 9000 CPUs have an
   integrated GPU), or log out of the desktop and send requests over SSH.
3. **Stock clocks and voltage:** undo any undervolt, overclock or power limit (LACT, CoreCtrl). Long compute work can
   fail at settings that games pass. `./triage.sh --print` shows the card's overdrive table.
4. **Which kernel hangs:** `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3 AMD_LOG_MASK=2097152 container/run.sh` runs one GPU
   kernel at a time and logs each one's name (`ShaderName`), so the last name in `container/run.sh logs` before the hang
   is the kernel that hung. The mask must be decimal. It's slow and the log grows fast, so use it for one reproduction.

For an issue, attach `./triage.sh --print` (it records which GPU runs the display, the clocks and overdrive table, the
amdgpu settings and loaded firmware versions, recent amdgpu kernel errors and the container's settings) and the
request that triggered it (prompt size, settings, whether it happens again). If the hang ended in a reboot and
`triage.sh` couldn't read the previous boot's kernel log, add `sudo journalctl -k -b -1 | grep -iE 'amdgpu|kfd'`:
the lines before the reset say which queue timed out.

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
  [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (Apache-2.0).
- [Mia's AI Lab](https://github.com/MiaAI-Lab), whose
  [Qwen3.8-27B one-click install for Nvidia GPUs](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install)
  started this project, and for the first EXL3 quant of the DFlash2 draft
  ([Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)).
- Rafa ([@rafatxf](https://github.com/rafatxf)) for DFlash2 rejection sampling, exllamav3 PR #423, included as an
  optional patch for the previous base (off by default; see `patches/README.md`).
- TensorFold, for the idea behind our earlier DeltaNet replay, and HermiG for the replay upstream now ships (#489).
- The Qwen team, for Qwen3.8-27B.

Our scripts, configs and docs are MIT (`LICENSE`). The patches modify ExLlamaV3 and are under its MIT license. Models
keep their own licences. Every change here was written by Claude Opus 5.5 and reviewed by GPT-6.1 Sol, except the
optional PR #423 patch by Rafa (and, in the previous base's patch, turboderp's graphs-off commit 96838c9, kept
unchanged under his name).
