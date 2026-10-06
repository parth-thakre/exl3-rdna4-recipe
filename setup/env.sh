# Source this (bash): activates the venv and sets the runtime environment for exllamav3 on the RX 9070 XT.
#   source setup/env.sh
_RECIPE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$_RECIPE_ROOT/.venv}
DEPS_DIR=${DEPS_DIR:-$_RECIPE_ROOT/deps}
source "$VENV/bin/activate"

# Pick the 9070 XT. On the test machine device 0 is the 9070 XT and the CPU's gfx1036 iGPU is hidden this way.
# Check `rocminfo` / `amd-smi list` if you have more than one GPU.
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export ROCM_HOME=${ROCM_HOME:-/usr} ROCM_PATH=${ROCM_PATH:-/usr}   # Fedora installs ROCm under /usr

# Headers that Fedora's ROCm 7.1.1 doesn't install by default, unpacked by setup/fetch_deps.sh. Triton also builds a
# small C helper at runtime and needs Python.h. Harmless if you installed the -devel packages with dnf instead.
export CPATH="$DEPS_DIR/usr/include:$DEPS_DIR/usr/include/python3.12${CPATH:+:$CPATH}"

# HIP graphs stay on. Graph::launch re-instantiates every 100k node updates (EXL3_GRAPH_REINST), which avoids the
# kernel-argument pool exhaustion in libamdhip64 (segfault after ~1.6M updates, about 1 h of serving).

# gfx12 WMMA multi-row GEMV for draft verification: DFlash2 127 -> 145 tok/s, MTP 82 -> 90.
export EXL3_GEMV_WMMA=1

# Split-decode attention tuned for gfx1201: the whole GQA group per program, 4 warps, 16-token tiles.
# 60k deep, 8-row verify: 1961 -> ~605 us per layer; 1-row decode 550 -> ~336 us (bench/bench_decode_attn.py).
export EXL3_DEC_GROUP=1 EXL3_DEC_WARPS=4 EXL3_DEC_BLOCK_N=16
# Q4W word-sliced kernel for 4-bit caches, 2 stages: 1.2-1.3x the packed kernel at q_len 4-8.
export EXL3_DEC_Q4W=1 EXL3_DEC_STAGES=2

unset _RECIPE_ROOT
