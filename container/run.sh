#!/bin/bash
# Serve Qwen3.8-27B with this recipe's container on any Linux with an RX 9070 / 9070 XT. Needs Docker or Podman and the
# amdgpu kernel driver (in every current distro kernel), not ROCm or Python.
#
#   container/run.sh            download the models on first use (~14 GB), then serve on http://127.0.0.1:8096/v1
#   CTX=128k container/run.sh   128k context instead of 96k, if nothing else (or only a light desktop) uses the GPU
#   container/run.sh logs       follow the server log
#   container/run.sh stop
#
# Environment: CTX (context, default 96k), IMAGE (default below), MODELS (default ~/.cache/exl3-rdna4/models), PORT (default 8096),
# CT=docker|podman (default: whichever works, Docker first), NO_TRIAGE=1 (don't offer ./triage.sh on failure),
# EXL3_QC_STAGING=0 (~0.2 GB less VRAM at 128k for 17-28% slower long-prompt prefill; notes/measurements.md).
# Every other EXL3_* variable (EXL3_TUNING=0 and the rest of setup/env.sh) and the ROCm debug variables AMD_LOG_LEVEL,
# AMD_LOG_MASK, AMD_SERIALIZE_KERNEL, AMD_SERIALIZE_COPY and HIP_LAUNCH_BLOCKING are passed into the container too
# (README, "GPU hangs and wrong output").
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/parth-thakre/exl3-rdna4:latest}
MODELS=${MODELS:-${XDG_CACHE_HOME:-$HOME/.cache}/exl3-rdna4/models}
PORT=${PORT:-8096}
CTX=${CTX:-96k}
NAME=exl3-rdna4
ROOT=$(cd "$(dirname "$0")/.." && pwd)

die() {
    echo "run.sh: $*" >&2
    # In a terminal, offer to open an AI coding agent with the details (./triage.sh)
    if [ -t 0 ] && [ -t 1 ] && [ -z "${NO_TRIAGE:-}" ]; then
        read -r -p "Open an AI assistant (Claude Code, Codex or OpenCode) with the details to help? [y/N] " ans || ans=
        case $ans in [yY]*) exec "$ROOT/triage.sh" "container/run.sh failed: $*" ;; esac
    fi
    echo "For help: ./triage.sh opens an AI coding agent with the details." >&2
    exit 1
}

if [ -z "${CT:-}" ]; then
    if command -v docker >/dev/null && docker info >/dev/null 2>&1; then CT=docker
    elif command -v podman >/dev/null; then CT=podman
    elif command -v docker >/dev/null; then
        die "Docker is installed but not usable by you. Start it (sudo systemctl enable --now docker) and add yourself to the docker group (sudo usermod -aG docker \$USER, then log in again), or install Podman"
    else die "needs Docker or Podman (Arch / Omarchy: sudo pacman -S docker, or podman)"
    fi
fi

case ${1:-serve} in
    logs) exec "$CT" logs -f "$NAME" ;;
    stop) exec "$CT" rm -f "$NAME" ;;
    serve) ;;
    *) die "usage: $0 [serve|logs|stop]" ;;
esac
command -v curl >/dev/null || die "needs curl (to check when the server is up)"

# The GPU: /dev/kfd (ROCm compute) and the render nodes, from the amdgpu kernel driver
[ -e /dev/kfd ] || die "/dev/kfd is missing: the amdgpu driver isn't loaded (is the RX 9070 XT installed and the kernel recent enough?)"
render=$(ls /dev/dri/renderD* 2>/dev/null | head -1) || true
[ -n "$render" ] || die "no /dev/dri/renderD* node"

# need_own_access: rootless runtimes pass your own groups through, so you need access to the devices yourself
need_own_access() {
    local d groups=""
    for d in /dev/kfd "$render"; do
        if [ ! -r "$d" ] || [ ! -w "$d" ]; then groups="$groups $(stat -c %G "$d")"; fi
    done
    [ -z "$groups" ] && return 0
    groups=$(tr ' ' '\n' <<<"$groups" | grep . | sort -u | paste -sd, -)
    die "no read/write access to /dev/kfd or $render. Add yourself to the group(s) $groups: sudo usermod -aG $groups \$USER (then log in again)"
}

gpu_args=(--device /dev/kfd --device /dev/dri)
# ROCm numbers the GPUs in the kernel's KFD order, which can put an AMD CPU's integrated GPU first; pin the container to
# the RDNA4 card (gfx1200 / gfx1201; the image is built for gfx1201) by its UUID. The container still needs every render node: ROCm finds no GPU
# at all when one in the topology has none.
for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    [[ $(awk '$1 == "gfx_target_version" {print $2}' "$p") =~ ^12000[01]$ ]] || continue
    u=$(awk '$1 == "unique_id" {print $2}' "$p")
    [[ $u =~ ^[0-9]+$ && $u != 0 ]] && gpu_args+=(-e "ROCR_VISIBLE_DEVICES=$(printf 'GPU-%016x' "$u")")
    break
done
user_args=()
if [ "$CT" = podman ]; then
    need_own_access
    # keep-groups (your groups inside the container) needs the crun runtime
    [ "$(podman info --format '{{.Host.OCIRuntime.Name}}' 2>/dev/null)" = crun ] ||
        die "Podman's runtime isn't crun, which passing your groups to the container needs (Arch: sudo pacman -S crun)"
    gpu_args+=(--group-add keep-groups --security-opt label=disable)
    user_args=(--security-opt label=disable)    # rootless Podman maps its root to you, so files in MODELS stay yours
else
    sec=$(docker info --format '{{json .SecurityOptions}}' 2>/dev/null || echo)
    if grep -q rootless <<<"$sec"; then
        # Rootless Docker: like Podman, the container's root is you, and only your own device access carries over
        need_own_access
    elif grep -q userns <<<"$sec"; then
        die "Docker runs with user-namespace remapping, which this script doesn't handle; use Podman (CT=podman $0)"
    else
        # Rootful Docker: run as you (files in MODELS stay yours) with the devices' groups
        user_args=(--user "$(id -u):$(id -g)")
        gpu_args+=(--group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g "$render")")
    fi
fi

# Models: downloaded once with the image's own hf CLI into a directory you own. setup/download_models.sh leaves a
# .download-complete marker only after a model is fully downloaded; an interrupted download resumes.
mkdir -p "$MODELS" || die "can't create $MODELS"
# The draft lives in draft/ (so the API lists only the main model); move one downloaded before that
if [ -d "$MODELS/Qwen3.8-27B-DFlash2-EXL3-3.0bpw" ] && [ ! -e "$MODELS/draft/Qwen3.8-27B-DFlash2-EXL3-3.0bpw" ]; then
    mkdir -p "$MODELS/draft" && mv "$MODELS/Qwen3.8-27B-DFlash2-EXL3-3.0bpw" "$MODELS/draft/" || die "can't move the draft into $MODELS/draft"
fi
if [ ! -f "$MODELS/Qwen3.8-27B-EXL3-SC3.0bpw-H4/.download-complete" ] ||
   [ ! -f "$MODELS/draft/Qwen3.8-27B-DFlash2-EXL3-3.0bpw/.download-complete" ]; then
    echo "downloading the models into $MODELS (about 14 GB)"
    "$CT" run --rm "${user_args[@]}" -v "$MODELS:/models" "$IMAGE" download || die "the model download failed (rerun to resume)"
fi

# Your EXL3_* settings and ROCm debug variables; not EXL3_MODELS (the image's /models) or EXL3_TREE (a host path)
env_args=() passed=()
while read -r v; do
    [[ $v =~ ^(EXL3_.+|AMD_LOG_LEVEL|AMD_LOG_MASK|AMD_SERIALIZE_KERNEL|AMD_SERIALIZE_COPY|HIP_LAUNCH_BLOCKING)$ ]] || continue
    [[ $v = EXL3_MODELS || $v = EXL3_TREE ]] && continue
    env_args+=(-e "$v=${!v}") passed+=("$v=${!v}")
done < <(compgen -e)
[ ${#passed[@]} -eq 0 ] || echo "passing to the container: ${passed[*]}"

"$CT" rm -f "$NAME" >/dev/null 2>&1 || true
"$CT" run -d --name "$NAME" "${gpu_args[@]}" "${user_args[@]}" --shm-size 8g \
    -e CTX="$CTX" "${env_args[@]}" -p "127.0.0.1:$PORT:8096" -v "$MODELS:/models:ro" "$IMAGE" >/dev/null || die "the container didn't start"
echo "starting $NAME ($CT, $CTX context); loading the model takes a minute or two"
for _ in $(seq 1 120); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "ready: http://127.0.0.1:$PORT/v1 (OpenAI-compatible, no API key; local only)"
        exit 0
    fi
    "$CT" ps -q -f name="^$NAME\$" | grep -q . || { "$CT" logs --tail 30 "$NAME" >&2 || true; die "the server exited"; }
    sleep 3
done
die "not ready after 6 minutes; see: $0 logs"
