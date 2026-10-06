#!/bin/bash
# Resume the GPQA run after a crash or reboot: bench/gpqa_resume.sh
# Finished questions are skipped (gpqa/results_*.jsonl); a fully finished model only costs a server start.
# Stops any TabbyAPI of this user first, since a leftover server would hold the GPU and the port.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if ps -u "$(id -u)" -o args= | awk '$1 ~ /bash$/ && $2 ~ /run_gpqa_all\.sh$/ {f=1} END {exit !f}'; then
    echo "a GPQA run is already going; progress: tail -f logs/gpqa_*.log"
    exit 1
fi

for pid in $(ps -u "$(id -u)" -o pid=,args= | awk '$2=="python" && $3=="main.py" {print $1}'); do
    kill "$pid"
    while kill -0 "$pid" 2>/dev/null; do sleep 1; done
done

mkdir -p logs
nohup setsid bench/run_gpqa_all.sh >> logs/gpqa_all.log 2>&1 < /dev/null &
echo "resumed; progress: tail -f logs/gpqa_*.log"
for f in gpqa/results_*.jsonl; do [ -f "$f" ] && echo "  $(basename "$f" .jsonl | sed 's/^results_//'): $(wc -l < "$f")/${GPQA_LIMIT:-50}"; done
