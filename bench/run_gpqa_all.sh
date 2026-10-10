#!/bin/bash
# GPQA Diamond (first 50 questions) on the SC 3.0 bpw H4 model, then, if downloaded, the plain 3.0 bpw and the 2.5 bpw
# one, one after the other. Each run starts its own TabbyAPI (default profile), checks that it serves the expected
# model, and stops it afterwards.
# Refuses to start if something already answers on TABBY_URL. Needs gpqa/gpqa_diamond.csv (bench/README.md).
# Resumable: rerun (or bench/gpqa_resume.sh) and each model continues where its results file stops.
# Progress: tail -f logs/gpqa_*.log. Exits nonzero if any run fails; prints GPQA_ALL_DONE only if all succeed.
#
# Defaults are the settings behind the 86% result: reasoning_effort=medium + the anti-spiral system prompt.
# GPQA_EFFORT= GPQA_SYSTEM= bench/run_gpqa_all.sh runs with the model's default effort and no system prompt instead.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source setup/env.sh || exit 1
source bench/server.sh
export GPQA_EFFORT=${GPQA_EFFORT-medium} GPQA_SYSTEM=${GPQA_SYSTEM-antispiral}
suffix=${GPQA_EFFORT:+-$GPQA_EFFORT}${GPQA_SYSTEM:+-as}
mkdir -p logs
trap 'stop_server' EXIT

run_eval() {   # run_eval NAME MODEL [TabbyAPI args...]
    local name=$1 model=$2; shift 2
    start_server "logs/gpqa_server_$name.log" "$model" "$@" || { echo "$name: server did not come up"; return 1; }
    python bench/gpqa_eval.py "$name" > "logs/gpqa_$name.log" 2>&1
    local rc=$?
    stop_server
    [ $rc -eq 0 ] || echo "$name: gpqa_eval.py failed (exit $rc), see logs/gpqa_$name.log"
    return $rc
}

failed=0
run_eval "exl3-sc3.0bpw-h4$suffix" Qwen3.8-27B-EXL3-SC3.0bpw-H4 --model-name Qwen3.8-27B-EXL3-SC3.0bpw-H4 || failed=1
if [ -d models/Qwen3.8-27B-EXL3-3.0bpw ]; then
    run_eval "exl3-3.0bpw$suffix" Qwen3.8-27B-EXL3-3.0bpw --model-name Qwen3.8-27B-EXL3-3.0bpw || failed=1
fi
if [ -d models/Qwen3.8-27B-EXL3-2.5bpw ]; then
    run_eval "exl3-2.5bpw$suffix" Qwen3.8-27B-EXL3-2.5bpw --model-name Qwen3.8-27B-EXL3-2.5bpw || failed=1
fi
[ $failed -eq 0 ] || { echo "GPQA_ALL_FAILED"; exit 1; }
echo "GPQA_ALL_DONE"
