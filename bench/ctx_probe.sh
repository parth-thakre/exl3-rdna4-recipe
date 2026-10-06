#!/bin/bash
# Start TabbyAPI with a given context/cache config, fill the context nearly to the top and measure.
# usage: bench/ctx_probe.sh LABEL CTX_K [extra TabbyAPI args...]   e.g. bench/ctx_probe.sh k4v3-96k 96 --cache-mode 4,3
# Prints the load result and VRAM, then runs context_bench.py at 8k and at (CTX_K - 4)k (override with DEPTHS="8 32 60").
# Stops any running TabbyAPI first (any `python main.py` process of this user) and leaves the new one running on success.
# Uses amd-smi for VRAM; GPU_INDEX picks the card (default 0).
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
[ $# -ge 2 ] || { sed -n '2,6p' "$0"; exit 2; }
label=$1; ctx_k=$2; shift 2
ctx=$((ctx_k * 1024))
gpu=${GPU_INDEX:-0}
vram() { amd-smi metric -g "$gpu" --mem-usage 2>/dev/null | awk '/USED_VRAM:/{print $2}'; }
mkdir -p logs

for p in $(ps -u "$(id -u)" -o pid=,args= | awk '$2=="python" && $3=="main.py"{print $1}'); do
    kill "$p"; while kill -0 "$p" 2>/dev/null; do sleep 1; done
done
# wait for the previous server's VRAM to be released
for _ in $(seq 1 60); do [ "$(vram)" -lt 1000 ] 2>/dev/null && break; sleep 1; done

setsid serve/run_tabby.sh --max-seq-len $ctx --cache-size $ctx "$@" > "logs/tabby_$label.log" 2>&1 < /dev/null &
until grep -qE 'Serving OAI|RuntimeError|Traceback' "logs/tabby_$label.log"; do sleep 3; done
if ! grep -q 'Serving OAI' "logs/tabby_$label.log"; then
    echo "$label: LOAD FAILED: $(grep -oE 'RuntimeError.*' "logs/tabby_$label.log" | head -1)"; exit 1
fi
echo "$label: loaded, VRAM $(vram) MB"
source setup/env.sh
python bench/context_bench.py "$label" ${DEPTHS:-8 $((ctx_k - 4))} 2>&1 | grep -v Warning
echo "$label: VRAM after fill $(vram) MB; OOM lines: $(grep -c OutOfMemory "logs/tabby_$label.log")"
