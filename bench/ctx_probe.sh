#!/bin/bash
# Start TabbyAPI with a given context/cache config, fill the context nearly to the top and measure.
# usage: bench/ctx_probe.sh LABEL CTX_K [extra TabbyAPI args...]   e.g. bench/ctx_probe.sh k4v3-96k 96 --cache-mode 4,3
# Stops this repo's TabbyAPI first (serve/stop_tabby.sh), then prints the load result and VRAM and runs
# context_bench.py at 8k and at (CTX_K - 4)k (override with DEPTHS="8 32 60"). Leaves the server running on success
# and stops it on failure. Startup gives up after START_TIMEOUT seconds (default 900).
# VRAM comes from amd-smi; GPU_INDEX picks the card (default 0). Exits nonzero if anything fails.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
[ $# -ge 2 ] || { sed -n '2,7p' "$0"; exit 2; }
label=$1; ctx_k=$2; shift 2
[[ $ctx_k =~ ^[0-9]+$ ]] || { echo "CTX_K must be a number of k tokens" >&2; exit 2; }
ctx=$((ctx_k * 1024))
gpu=${GPU_INDEX:-0}
vram() { amd-smi metric -g "$gpu" --mem-usage 2>/dev/null | awk '/USED_VRAM:/{print $2}' || true; }
source setup/env.sh
source "$ROOT/bench/server.sh"
mkdir -p logs
log=logs/tabby_$label.log

serve/stop_tabby.sh
# wait for the previous server's VRAM to be released
for _ in $(seq 1 60); do v=$(vram); [ -n "$v" ] && [ "$v" -lt 1000 ] && break; sleep 1; done

model=$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["model"]["model_name"])' \
        "${TABBY_DIR:-$ROOT/tabbyAPI}/config.yml" 2>/dev/null || true)
for ((i = 1; i <= $#; i++)); do [ "${!i}" = --model-name ] && { j=$((i + 1)); model=${!j}; }; done
[ -n "$model" ] || { echo "can't tell which model the server will load" >&2; exit 1; }

if ! start_server "$log" "$model" --max-seq-len $ctx --cache-size $ctx "$@"; then
    echo "$label: LOAD FAILED: $(grep -oE '(RuntimeError|OutOfMemoryError|Error).*' "$log" | head -1)"
    exit 1
fi
trap 'echo "$label: failed, stopping the server"; stop_server' ERR
echo "$label: loaded, VRAM $(vram) MB"
python bench/context_bench.py "$label" ${DEPTHS:-8 $((ctx_k - 4))} 2> >(grep -v Warning >&2)
trap - ERR
echo "$label: VRAM after fill $(vram) MB; OOM lines: $(grep -c OutOfMemory "$log" || true)"
