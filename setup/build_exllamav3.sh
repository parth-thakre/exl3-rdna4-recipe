#!/bin/bash
# Clone ExLlamaV3 at the pinned commit, apply the RDNA4 patch (and, by default, the optional PR #423 patch), build the
# extension for gfx1201 and install it (editable) into the venv.
#
# Environment overrides:
#   EXLLAMAV3_GIT_URL / EXLLAMAV3_COMMIT / EXLLAMAV3_PATCH   see setup/versions.sh (a local mirror works as the URL)
#   WITH_PR423=0       skip patches/optional/pr423-dflash2-rejection-sampling.patch
#   EXLLAMAV3_DIR      checkout location (default: exllamav3/ in the repo root)
#   RESET=1            discard local changes, other patches and build outputs in EXLLAMAV3_DIR, then patch afresh
#   VENV               venv to build with and install into (default: .venv/)
#   DEPS_DIR           unpacked header packages from setup/fetch_deps.sh (default: deps/)
#   PYTORCH_ROCM_ARCH  GPU target (default: gfx1201)
#   BUILD_CC/BUILD_CXX compilers for the extension (default: ROCm's clang)
#   MAX_JOBS           parallel compile jobs (default: 12; each needs a few GB of RAM)
#   NO_INSTALL=1       only build the extension in place; don't pip install anything
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/setup/versions.sh"
source "$ROOT/setup/lib.sh"
EXLLAMAV3_DIR=${EXLLAMAV3_DIR:-$ROOT/exllamav3}
VENV=${VENV:-$ROOT/.venv}
DEPS_DIR=${DEPS_DIR:-$ROOT/deps}

patches=("$EXLLAMAV3_PATCH")
[ "$WITH_PR423" = 1 ] && patches+=("$PR423_PATCH")
checkout_patched "$EXLLAMAV3_DIR" "$EXLLAMAV3_GIT_URL" "$EXLLAMAV3_COMMIT" "${patches[@]}"

[ -f "$VENV/bin/activate" ] || die "no venv at $VENV; run setup/make_venv.sh first"
source "$VENV/bin/activate"
cd "$EXLLAMAV3_DIR"
export ROCM_HOME=${ROCM_HOME:-/usr} ROCM_PATH=${ROCM_PATH:-/usr}       # Fedora's split ROCm layout lives under /usr
# Not CC/CXX: setup/env.sh sets CC=gcc for Triton, and the extension must be built with ROCm's clang
export CC=${BUILD_CC:-/usr/lib64/rocm/llvm/bin/clang} CXX=${BUILD_CXX:-/usr/lib64/rocm/llvm/bin/clang++}
export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx1201}                # only the 9070 XT, not an iGPU
export CPATH="$DEPS_DIR/usr/include:$DEPS_DIR/usr/include/python3.12${CPATH:+:$CPATH}"
export MAX_JOBS=${MAX_JOBS:-12}

mkdir -p "$ROOT/logs"
if ! python setup.py build_ext --inplace 2>&1 | tee "$ROOT/logs/build_exllamav3.log"; then
    die "build failed, see logs/build_exllamav3.log"
fi

if [ "${NO_INSTALL:-0}" != 1 ]; then
    EXLLAMA_NOCOMPILE=1 python -m pip install -e . --no-deps --no-build-isolation
    python -c "from exllamav3.ext import exllamav3_ext; print('exllamav3_ext loaded')"
fi
