# Source this (bash): activates the venv and sets the runtime environment for exllamav3 on the RX 9070 XT.
#   source setup/env.sh
# Returns nonzero (without exiting your shell) if the venv is missing.
_RECIPE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$_RECIPE_ROOT/.venv}
DEPS_DIR=${DEPS_DIR:-$_RECIPE_ROOT/deps}
unset _RECIPE_ROOT
if [ ! -f "$VENV/bin/activate" ]; then
    echo "setup/env.sh: no venv at $VENV (run setup/make_venv.sh, or set VENV)" >&2
    return 1
fi
source "$VENV/bin/activate" || return 1

# Pick the 9070 XT. On the test machine device 0 is the 9070 XT and the CPU's gfx1036 iGPU is hidden this way.
# Check `rocminfo` / `amd-smi list` if you have more than one GPU.
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export ROCM_HOME=${ROCM_HOME:-/usr} ROCM_PATH=${ROCM_PATH:-/usr}   # Fedora installs ROCm under /usr

# Triton compiles a small C helper at runtime: it needs a host C compiler (gcc; sudo dnf install gcc gcc-c++) and
# Python.h. The headers come from python3.12-devel, or from deps/ as unpacked by setup/fetch_deps.sh.
export CC=${CC:-gcc}
command -v "$CC" >/dev/null || echo "setup/env.sh: warning: C compiler '$CC' not found; Triton kernels won't compile" >&2
export CPATH="$DEPS_DIR/usr/include:$DEPS_DIR/usr/include/python3.12${CPATH:+:$CPATH}"

# Expandable segments cut allocator fragmentation, which otherwise can OOM a long prefill after concurrent use
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# HIP graphs stay on. Each node update leaks ~4 KB of device memory until the graph is re-instantiated
# (ROCm/rocm-systems#10713), so Graph::launch re-instantiates every EXL3_GRAPH_REINST node updates. The patch's
# built-in default of 100k is too high with several requests at once: the server segfaulted inside
# hipGraphExecKernelNodeSetParams after 16-23 min of sustained 3-request load; at 10k the same load ran 78 min
# without a crash, at no measurable speed cost.
export EXL3_GRAPH_REINST=${EXL3_GRAPH_REINST:-10000}

# gfx12 WMMA multi-row GEMV for draft verification.
export EXL3_GEMV_WMMA=1

# Split-decode attention tuned for gfx1201: the whole GQA group per program, 4 warps, 16-token tiles.
# 60k deep, 8-row verify: 1961 -> ~605 us per layer; 1-row decode 550 -> ~336 us (bench/bench_decode_attn.py).
export EXL3_DEC_GROUP=1 EXL3_DEC_WARPS=4 EXL3_DEC_BLOCK_N=16
# Q4W word-sliced kernel for 4-bit caches, 2 stages: 1.2-1.3x the packed kernel at q_len 4-8.
export EXL3_DEC_Q4W=1 EXL3_DEC_STAGES=2
