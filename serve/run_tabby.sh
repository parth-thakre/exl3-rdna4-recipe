#!/bin/bash
# Default profile: Qwen3.8-27B EXL3 3.0 bpw + DFlash2 3.0 bpw draft (7 tokens), 128k context (tabbyAPI/config.yml).
# EXL3_GDN_REPLAY=1: the DeltaNet layers verify without a recurrent-state snapshot per draft token and replay only the
# accepted inputs (-1.06 GB VRAM, +3% decode). That memory is what fits 128k next to the DFlash2 draft.
# Extra arguments go to TabbyAPI's main.py and override config.yml, e.g. --max-seq-len 65536 --cache-size 65536.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/setup/env.sh"
export EXL3_GDN_REPLAY=${EXL3_GDN_REPLAY:-1}
cd "${TABBY_DIR:-$ROOT/tabbyAPI}"
exec python main.py "$@"
