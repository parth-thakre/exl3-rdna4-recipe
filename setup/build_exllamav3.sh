#!/bin/bash
# Clone ExLlamaV3 at the pinned commit, apply the RDNA4 patch, build the extension for gfx1201 and install it
# (editable) into the venv.
#
# Environment overrides:
#   EXLLAMAV3_URL / EXLLAMAV3_COMMIT / EXLLAMAV3_PATCH   see setup/versions.sh (a local mirror works as the URL)
#   EXLLAMAV3_DIR      checkout location (default: exllamav3/ in the repo root)
#   VENV               venv to build with and install into (default: .venv/)
#   DEPS_DIR           unpacked header packages from setup/fetch_deps.sh (default: deps/)
#   PYTORCH_ROCM_ARCH  GPU target (default: gfx1201)
#   MAX_JOBS           parallel compile jobs (default: 12; each needs a few GB of RAM)
#   NO_INSTALL=1       only build the extension in place; don't pip install anything
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/setup/versions.sh"
EXLLAMAV3_DIR=${EXLLAMAV3_DIR:-$ROOT/exllamav3}
VENV=${VENV:-$ROOT/.venv}
DEPS_DIR=${DEPS_DIR:-$ROOT/deps}
PATCH_FILE=$ROOT/$EXLLAMAV3_PATCH

if [ ! -d "$EXLLAMAV3_DIR/.git" ]; then
    git clone "$EXLLAMAV3_URL" "$EXLLAMAV3_DIR"
fi
cd "$EXLLAMAV3_DIR"

git cat-file -e "$EXLLAMAV3_COMMIT^{commit}" 2>/dev/null || git fetch origin
if [ "$(git rev-parse HEAD)" = "$(git rev-parse "$EXLLAMAV3_COMMIT^{commit}")" ] &&
   git apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "patch already applied on $(git rev-parse --short HEAD)"
else
    [ -z "$(git status --porcelain --untracked-files=no)" ] || {
        echo "$EXLLAMAV3_DIR has local changes and the patch isn't applied; refusing to touch it" >&2; exit 1; }
    git -c advice.detachedHead=false checkout -q --detach "$EXLLAMAV3_COMMIT"
    git apply --check "$PATCH_FILE"
    git apply "$PATCH_FILE"
    echo "applied $(basename "$PATCH_FILE") on $(git rev-parse --short HEAD)"
fi

source "$VENV/bin/activate"
export ROCM_HOME=${ROCM_HOME:-/usr} ROCM_PATH=${ROCM_PATH:-/usr}       # Fedora's split ROCm layout lives under /usr
export CC=${CC:-/usr/lib64/rocm/llvm/bin/clang} CXX=${CXX:-/usr/lib64/rocm/llvm/bin/clang++}
export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx1201}                # only the 9070 XT, not an iGPU
export CPATH="$DEPS_DIR/usr/include:$DEPS_DIR/usr/include/python3.12${CPATH:+:$CPATH}"
export MAX_JOBS=${MAX_JOBS:-12}

mkdir -p "$ROOT/logs"
if ! python setup.py build_ext --inplace 2>&1 | tee "$ROOT/logs/build_exllamav3.log"; then
    echo "build failed, see logs/build_exllamav3.log" >&2; exit 1
fi

if [ "${NO_INSTALL:-0}" != 1 ]; then
    EXLLAMA_NOCOMPILE=1 python -m pip install -e . --no-deps --no-build-isolation
    python -c "from exllamav3.ext import exllamav3_ext; print('exllamav3_ext loaded')"
fi
