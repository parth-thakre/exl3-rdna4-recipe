# Source this (bash): activates the venv and sets the runtime environment for exllamav3 on the RX 9070 XT.
#   source setup/env.sh
# Returns nonzero (without exiting your shell) if the venv is missing.
_RECIPE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-$_RECIPE_ROOT/.venv}
DEPS_DIR=${DEPS_DIR:-$_RECIPE_ROOT/deps}
source "$_RECIPE_ROOT/setup/rocm.sh"
unset _RECIPE_ROOT
if [ ! -f "$VENV/bin/activate" ]; then
    echo "setup/env.sh: no venv at $VENV (run setup/make_venv.sh, or set VENV)" >&2
    return 1
fi
source "$VENV/bin/activate" || return 1

# Pick the 9070 XT. On the test machine device 0 is the 9070 XT and the CPU's gfx1036 iGPU is hidden this way.
# Check `rocminfo` / `amd-smi list` if you have more than one GPU.
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
# System ROCm (setup/rocm.sh sets ROCM_PATH/ROCM_HOME: /opt/rocm on most distros, /usr on Fedora). Only building the
# extension needs it; serving uses the runtime in PyTorch's ROCm wheel, so without it (the container image) carry on.
find_rocm 2>/dev/null || true

# Triton compiles a small C helper at runtime: it needs a host C compiler (gcc) and
# Python.h. The headers come with Python (python3.12-devel / python3.12-dev on distros that split them), or from
# deps/ as unpacked by setup/fetch_deps.sh.
export CC=${CC:-gcc}
command -v "$CC" >/dev/null || echo "setup/env.sh: warning: C compiler '$CC' not found; Triton kernels won't compile" >&2
export CPATH="$DEPS_DIR/usr/include:$DEPS_DIR/usr/include/python3.12${CPATH:+:$CPATH}"

# Expandable segments off. On gfx1201, memory mapped through HIP's virtual memory API (which expandable segments use)
# can lose writes after a remap at the same address (ROCm/TheRock#8744, pytorch/pytorch#195202): a missing TLB flush
# in the amdgpu kernel driver with Resizable BAR, fixed by a kernel patch in review (notes/measurements.md). With them
# on, the recurrent kernel tests failed in 3 of 10 runs and upstream's spec-rewind e2e test in 3 of 45; off, none did.
# ExLlamaV3 v1.6.5 turns them on by default except on ROCm or when this variable is set; this sets an explicit default,
# and a value you set yourself is kept.
# Off, a 106k-token prefill after concurrent and image requests still fits at 3 slots (notes/measurements.md, Batching).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}

# HIP graphs are off: upstream (96838c9, in the base since v1.6.0) makes that the ROCm default. On this card replaying a
# graph is no faster than launching its kernels one by one (same tok/s at 1 and 3 requests, same greedy answers), and
# every graph exec holds kernel-argument memory that grows with each update until the server crashes
# (ROCm/rocm-systems#10713). EXL3_GRAPHS=1 turns them back on, which is only advisable on the previous base: its patch
# re-instantiates each graph every EXL3_GRAPH_REINST node updates to dodge the crash (the patch's 100k and then 10k both
# still crashed under sustained 2-3 request load; 1000 then saw no crash in ~11 h of lighter use). The current patch
# doesn't have that, so the setting below only matters on the previous base.
export EXL3_GRAPH_REINST=${EXL3_GRAPH_REINST:-1000}

# The patch's kernel settings. Each keeps a value you set yourself. EXL3_TUNING=0 puts them all at upstream's defaults
# instead, which tells a GPU hang or wrong output caused by our kernels from one that isn't (README, "GPU hangs and
# wrong output"); it costs speed, not quality.
if [ "${EXL3_TUNING:-1}" = 0 ]; then
    export EXL3_GEMV_WMMA=0 EXL3_GEMV_MAX_M=8 EXL3_BC_MLP_MAX_ROWS=8 EXL3_DEC_GROUP=0 EXL3_DEC_Q4W=0
    unset EXL3_DEC_WARPS EXL3_DEC_BLOCK_N EXL3_DEC_STAGES
else
    # gfx12 WMMA multi-row GEMV for draft verification.
    export EXL3_GEMV_WMMA=${EXL3_GEMV_WMMA:-1}
    # Split-decode attention tuned for gfx1201: the whole GQA group per program, 4 warps, 16-token tiles.
    # 60k deep, 8-row verify: 1961 -> ~605 us per layer; 1-row decode 550 -> ~336 us (bench/bench_decode_attn.py).
    export EXL3_DEC_GROUP=${EXL3_DEC_GROUP:-1} EXL3_DEC_WARPS=${EXL3_DEC_WARPS:-4} EXL3_DEC_BLOCK_N=${EXL3_DEC_BLOCK_N:-16}
    # Q4W word-sliced kernel for 4-bit caches, 2 stages: 1.2-1.3x the packed kernel at q_len 4-8.
    export EXL3_DEC_Q4W=${EXL3_DEC_Q4W:-1} EXL3_DEC_STAGES=${EXL3_DEC_STAGES:-2}
fi
