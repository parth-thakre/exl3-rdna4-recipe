#!/bin/bash
# Resume the GPQA run after a crash or reboot: bench/gpqa_resume.sh
# Finished questions are skipped (gpqa/results_*.jsonl); a fully finished model only costs a server start.
# Stops this repo's TabbyAPI first (serve/stop_tabby.sh), since it would hold the GPU and the port.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if ps -u "$(id -u)" -o args= | awk '$1 ~ /bash$/ && $2 ~ /run_gpqa_all\.sh$/ {f=1} END {exit !f}'; then
    echo "a GPQA run is already going; progress: tail -f logs/gpqa_*.log"
    exit 1
fi

serve/stop_tabby.sh
mkdir -p logs
nohup setsid bench/run_gpqa_all.sh >> logs/gpqa_all.log 2>&1 < /dev/null &
echo "resumed; progress: tail -f logs/gpqa_*.log (logs/gpqa_all.log ends with GPQA_ALL_DONE or GPQA_ALL_FAILED)"
for f in gpqa/results_*.jsonl; do
    if [ -f "$f" ]; then echo "  $(basename "$f" .jsonl | sed 's/^results_//'): $(wc -l < "$f")/${GPQA_LIMIT:-50}"; fi
done
