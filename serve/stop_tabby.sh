#!/bin/bash
# Stop the TabbyAPI server started from this repo (or from TABBY_TREE), and nothing else.
# A process counts as ours if it is `python main.py` with its working directory in our TabbyAPI checkout: the PID in
# logs/tabby.pid (TABBY_PIDFILE) is checked that way, and any other such process of this user is stopped too.
# Waits up to STOP_TIMEOUT seconds (default 60) for each to exit, then sends SIGKILL.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TABBY_DIR=$(cd "${TABBY_TREE:-${TABBY_DIR:-$ROOT/tabbyAPI}}" 2>/dev/null && pwd -P) || { echo "no TabbyAPI checkout; nothing to stop"; exit 0; }
PIDFILE=${TABBY_PIDFILE:-$ROOT/logs/tabby.pid}
timeout=${STOP_TIMEOUT:-60}

is_ours() {   # is_ours PID
    local pid=$1
    [ -r "/proc/$pid/cmdline" ] || return 1
    [ "$(stat -c %u "/proc/$pid")" = "$(id -u)" ] || return 1
    [ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" = "$TABBY_DIR" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -qE '(^|/)python[0-9.]* main\.py( |$)'
}

pids=()
if [ -f "$PIDFILE" ]; then
    p=$(cat "$PIDFILE")
    [[ $p =~ ^[0-9]+$ ]] && is_ours "$p" && pids+=("$p")
fi
for d in /proc/[0-9]*; do
    p=${d#/proc/}
    [[ " ${pids[*]} " == *" $p "* ]] && continue
    is_ours "$p" 2>/dev/null && pids+=("$p")
done

for p in "${pids[@]}"; do
    echo "stopping TabbyAPI (PID $p)"
    kill "$p" 2>/dev/null || continue
    for _ in $(seq 1 "$timeout"); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    if kill -0 "$p" 2>/dev/null; then echo "PID $p still running after ${timeout}s, sending SIGKILL"; kill -9 "$p" 2>/dev/null || true; fi
done
[ ${#pids[@]} -gt 0 ] || echo "no TabbyAPI from this repo is running"
rm -f "$PIDFILE"
