#!/bin/bash
# GPQA Diamond (first 50 questions) on the 3.0 bpw and, if downloaded, 2.5 bpw model, one after the other. Each run
# starts its own TabbyAPI (default profile) and stops it afterwards. Needs gpqa/gpqa_diamond.csv (bench/README.md).
# Resumable: rerun (or bench/gpqa_resume.sh) and each model continues where its results file stops.
# Progress: tail -f logs/gpqa_*.log
#
# Defaults are the settings behind the 86% result: reasoning_effort=medium + the anti-spiral system prompt.
# GPQA_EFFORT= GPQA_SYSTEM= bench/run_gpqa_all.sh runs with the model's default effort and no system prompt instead.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source setup/env.sh
export GPQA_EFFORT=${GPQA_EFFORT-medium} GPQA_SYSTEM=${GPQA_SYSTEM-antispiral}
suffix=${GPQA_EFFORT:+-$GPQA_EFFORT}${GPQA_SYSTEM:+-as}
BASE=${TABBY_URL:-http://127.0.0.1:8096/v1}
mkdir -p logs

key() { python -c 'import sys; sys.path.insert(0, "bench"); from common import api_key; print(api_key() or "")'; }
wait_up() {
    for _ in $(seq 1 200); do
        curl -sf -H "Authorization: Bearer $(key)" "$BASE/models" >/dev/null && return 0
        kill -0 "$1" 2>/dev/null || return 1
        sleep 3
    done
    return 1
}
stop() { kill "$1" 2>/dev/null; while kill -0 "$1" 2>/dev/null; do sleep 1; done; }

run_eval() {   # run_eval NAME [TabbyAPI args...]
    local name=$1; shift
    setsid serve/run_tabby.sh "$@" > "logs/gpqa_server_$name.log" 2>&1 < /dev/null &
    local pid=$!
    wait_up $pid || { echo "$name: server did not come up (logs/gpqa_server_$name.log)"; stop $pid; return 1; }
    python bench/gpqa_eval.py "$name" > "logs/gpqa_$name.log" 2>&1
    stop $pid
}

# One question at a time: batching (only MTP fits batch 4) measured ~44 tok/s total vs 55-86 for single DFlash2 requests
run_eval "exl3-3.0bpw$suffix"
if [ -d models/Qwen3.8-27B-EXL3-2.5bpw ]; then
    run_eval "exl3-2.5bpw$suffix" --model-name Qwen3.8-27B-EXL3-2.5bpw
fi
echo "GPQA_ALL_DONE"
