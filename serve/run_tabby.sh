#!/bin/bash
# Default profile: Qwen3.8-27B EXL3 3.0 bpw + DFlash2 3.0 bpw draft (7 tokens), 128k context (tabbyAPI/config.yml).
# EXL3_GDN_REPLAY=1: the DeltaNet layers verify without a recurrent-state snapshot per draft token and replay only the
# accepted inputs (-1.06 GB VRAM). That memory is what fits 128k next to the DFlash2 draft.
# Extra arguments go to TabbyAPI's main.py and override config.yml, e.g. --max-seq-len 65536 --cache-size 65536.
#
# Which code runs:
#   EXL3_TREE   exllamav3 checkout put first on PYTHONPATH (default: exllamav3/ in the repo root, if it has a built
#               extension). This also runs a NO_INSTALL=1 build, or any other built checkout, without pip.
#   TABBY_TREE  TabbyAPI checkout (default: tabbyAPI/ in the repo root; TABBY_DIR works too)
# Writes its PID to logs/tabby.pid (TABBY_PIDFILE); serve/stop_tabby.sh uses it.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TABBY_TREE=${TABBY_TREE:-${TABBY_DIR:-$ROOT/tabbyAPI}}
exl3_explicit=${EXL3_TREE:+1}
EXL3_TREE=${EXL3_TREE:-$ROOT/exllamav3}
# Absolute paths: PYTHONPATH is used after the cd into TabbyAPI below, and stop_tabby.sh compares the server's cwd
TABBY_TREE=$(cd "$TABBY_TREE" 2>/dev/null && pwd -P) || { echo "run_tabby.sh: no TabbyAPI checkout; run setup/install_tabby.sh" >&2; exit 1; }
if [ -d "$EXL3_TREE" ]; then
    EXL3_TREE=$(cd "$EXL3_TREE" && pwd -P)
elif [ -n "$exl3_explicit" ]; then
    echo "run_tabby.sh: EXL3_TREE=$EXL3_TREE doesn't exist" >&2; exit 1
fi
PIDFILE=${TABBY_PIDFILE:-$ROOT/logs/tabby.pid}

source "$ROOT/setup/env.sh" || { echo "run_tabby.sh: environment setup failed" >&2; exit 1; }
[ -f "$TABBY_TREE/main.py" ] || { echo "run_tabby.sh: no TabbyAPI in $TABBY_TREE; run setup/install_tabby.sh" >&2; exit 1; }
[ -f "$TABBY_TREE/config.yml" ] || { echo "run_tabby.sh: no $TABBY_TREE/config.yml; run setup/install_tabby.sh" >&2; exit 1; }
if compgen -G "$EXL3_TREE/exllamav3_ext*.so" >/dev/null; then
    export PYTHONPATH="$EXL3_TREE${PYTHONPATH:+:$PYTHONPATH}"
elif [ -n "$exl3_explicit" ]; then
    echo "run_tabby.sh: no built exllamav3_ext in $EXL3_TREE" >&2; exit 1
fi
python -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("exllamav3") is None)' || {
    echo "run_tabby.sh: exllamav3 isn't importable; run setup/build_exllamav3.sh" >&2; exit 1; }
cd "$TABBY_TREE"
export EXL3_GDN_REPLAY=${EXL3_GDN_REPLAY:-1}

mkdir -p "$(dirname "$PIDFILE")"
echo $$ > "$PIDFILE"     # exec keeps the PID, so this is the server's PID
exec python main.py "$@"
