# Container

The whole recipe in one image: ExLlamaV3 with the RDNA4 patches, TabbyAPI, and the tuned settings, serving
Qwen3.8-27B with the DFlash2 draft at 96k context (128k optional). The models are downloaded once into a directory on
the host and mounted read-only.

The host needs only:

- an RX 9070 or RX 9070 XT (gfx1201). A desktop can share it at the default 96k context (tested with ~1.9 GB of its
  16 GB already in use); 128k needs it to itself or a light desktop (~1 GB)
- Docker or Podman
- the amdgpu kernel driver, which every current distro kernel has (check: `/dev/kfd` exists)

No ROCm, compiler or Python on the host, so the distro doesn't matter beyond those three. Tested so far with rootless
Podman on Fedora 44 (Parth's RX 9070 XT): output identical to the host install, same speed, and the local-ai registry's
gates pass. The Docker path (Omarchy's default) is written for rootful and rootless Docker but not tested yet.

## Run

```bash
curl -fsSL https://raw.githubusercontent.com/parth-thakre/exl3-rdna4-recipe/main/install.sh | bash   # or, by hand:
git clone https://github.com/parth-thakre/exl3-rdna4-recipe && cd exl3-rdna4-recipe
container/run.sh          # first run downloads ~14 GB of models; then http://127.0.0.1:8096/v1, 96k context
CTX=128k container/run.sh # 128k context, if nothing else (or only a light desktop) uses the GPU
container/run.sh logs
container/run.sh stop
```

`run.sh` uses Docker if it works for you, otherwise Podman. The models go to `~/.cache/exl3-rdna4/models` (set
`MODELS` to change it); an interrupted download resumes on the next run. The API is OpenAI-compatible and has no key:
it listens on 127.0.0.1 only and refuses browser (cross-origin) callers, so web pages you open can't use it. Put a
proxy with a key in front before exposing it anywhere else. For a quick chat in the browser, TabbyAPI's own chat page
is at http://127.0.0.1:8096/ui/.

If something fails, `run.sh` offers to open Claude Code, Codex or OpenCode with the details (`./triage.sh` in the repo
root, which you can also run yourself). For a GPU hang or garbled output, see "GPU hangs and wrong output" in the main
README: `run.sh` passes your `EXL3_*` settings (`EXL3_TUNING=0` among them) and ROCm's debug variables into the
container.

**Omarchy:** Docker comes installed, so `container/run.sh` is all you need. If Docker says permission denied, add
yourself to the `docker` group (`sudo usermod -aG docker $USER`) and log in again. Once the image is in the local-ai
registry, Omarchy's Local AI plugin will offer it as a recipe for the RX 9070 XT, and starting it there needs nothing
from this page.

**Podman:** rootless Podman passes your own groups to the container, so you need access to `/dev/kfd` and
`/dev/dri/renderD*` yourself; `run.sh` says which group to join if you don't have it (usually `render`).

## By hand

```bash
docker run -d --name exl3-rdna4 --device /dev/kfd --device /dev/dri \
    --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" --shm-size 8g \
    -p 127.0.0.1:8096:8096 -v ~/.cache/exl3-rdna4/models:/models:ro ghcr.io/parth-thakre/exl3-rdna4:latest
```

With Podman, use `--group-add keep-groups --security-opt label=disable` instead of the two `--group-add` options.
The models it expects in `/models`, and what `entrypoint.sh download` fetches:

| Directory | Source |
|---|---|
| `Qwen3.8-27B-EXL3-SC3.0bpw-H4` | [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3), revision `86b9553` (branch `SC_3.00bpw_H4`) |
| `draft/Qwen3.8-27B-DFlash2-EXL3-3.0bpw` | [parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw](https://huggingface.co/parth-thakre/Qwen3.8-27B-DFlash2-EXL3-3.0bpw), revision `3d915b1` (our 3.0 bpw quant of inco.ai's draft) |

Add `-e CTX=128k` for 128k context, `-e EXL3_TUNING=0` for the patch's kernels at upstream's defaults. Arguments after the image name go to TabbyAPI (`--max-batch-size 2`, `--config
/path/in/container.yml`, ...). The built-in config is [`config.yml`](config.yml): the same settings as a host install.

## Build

```bash
podman build -f container/Dockerfile -t exl3-rdna4 container                    # the recipe from GitHub (main)
podman build -f container/Dockerfile --build-arg RECIPE_COMMIT=<commit> -t exl3-rdna4 container
```

The build stage runs the recipe's own setup scripts on AMD's ROCm 7.2.4 Ubuntu image (so the image is built the way the
README's install builds), the runtime stage is plain Ubuntu 24.04 with Python 3.12: the extension's ROCm libraries come
from the PyTorch ROCm wheel. The image is about 17 GB, mostly that wheel. A build takes 30-60 minutes and needs no GPU.
`.github/workflows/container.yml` builds and publishes `ghcr.io/parth-thakre/exl3-rdna4` with a build attestation.
