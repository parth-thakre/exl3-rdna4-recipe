#!/bin/bash
# Stop the TabbyAPI server started from this repo (or from TABBY_TREE), and nothing else.
# A process counts as ours if it is `python main.py` (see is_ours) running in our TabbyAPI checkout: the PID in
# logs/tabby.pid (TABBY_PIDFILE) is checked that way, and any other such process of this user is stopped too.
# Waits up to STOP_TIMEOUT seconds (default 60) for each to exit, then sends SIGKILL.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TABBY_DIR=$(cd "${TABBY_TREE:-${TABBY_DIR:-$ROOT/tabbyAPI}}" 2>/dev/null && pwd -P) || { echo "no TabbyAPI checkout; nothing to stop"; exit 0; }
PIDFILE=${TABBY_PIDFILE:-$ROOT/logs/tabby.pid}
timeout=${STOP_TIMEOUT:-60}
VENV_PY=$(readlink -f "${VENV:-$ROOT/.venv}/bin/python" 2>/dev/null || true)

# is_ours PID: owned by this user, working directory is our TabbyAPI checkout, and argv is exactly
# <python> main.py [args...], where <python> is python, python3, python3.N or the venv's python, and main.py is the
# file in our checkout (given as "main.py" or as a path to it). The cmdline is read as NUL-separated argv, so text
# inside later arguments can't match.
is_ours() {
    local pid=$1 argv=() exe
    [ -r "/proc/$pid/cmdline" ] || return 1
    [ "$(stat -c %u "/proc/$pid" 2>/dev/null)" = "$(id -u)" ] || return 1
    [ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" = "$TABBY_DIR" ] || return 1
    mapfile -d '' -t argv < "/proc/$pid/cmdline" 2>/dev/null || return 1
    [ ${#argv[@]} -ge 2 ] || return 1
    exe=${argv[0]##*/}
    if ! [[ $exe =~ ^python(3(\.[0-9]+)?)?$ ]]; then
        [ -n "$VENV_PY" ] && [ "$(readlink -f "${argv[0]}" 2>/dev/null)" = "$VENV_PY" ] || return 1
    fi
    case ${argv[1]} in
        main.py) return 0 ;;
        */main.py) [ "$(cd "/proc/$pid/cwd" 2>/dev/null && readlink -f "${argv[1]}")" = "$TABBY_DIR/main.py" ] ;;
        *) return 1 ;;
    esac
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
