#!/bin/bash
# Clone TabbyAPI at the pinned commit, apply the patch that lets it accept RDNA3/RDNA4 GPUs, install its
# dependencies into the same venv, and copy serve/config.example.yml to tabbyAPI/config.yml if there's none yet.
#
# Environment overrides:
#   TABBY_URL / TABBY_COMMIT / TABBY_PATCH   see setup/versions.sh (TABBY_PATCH= skips patching)
#   TABBY_DIR    checkout location (default: tabbyAPI/ in the repo root)
#   VENV         default: .venv/
#   NO_INSTALL=1 clone and patch only
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/setup/versions.sh"
TABBY_DIR=${TABBY_DIR:-$ROOT/tabbyAPI}
VENV=${VENV:-$ROOT/.venv}

if [ ! -d "$TABBY_DIR/.git" ]; then
    git clone "$TABBY_URL" "$TABBY_DIR"
fi
cd "$TABBY_DIR"

git cat-file -e "$TABBY_COMMIT^{commit}" 2>/dev/null || git fetch origin
if [ -z "$TABBY_PATCH" ]; then
    git -c advice.detachedHead=false checkout -q --detach "$TABBY_COMMIT"
    echo "TabbyAPI at $(git rev-parse --short HEAD), no patch"
elif [ "$(git rev-parse HEAD)" = "$(git rev-parse "$TABBY_COMMIT^{commit}")" ] &&
     git apply --reverse --check "$ROOT/$TABBY_PATCH" 2>/dev/null; then
    echo "patch already applied on $(git rev-parse --short HEAD)"
else
    [ -z "$(git status --porcelain --untracked-files=no)" ] || {
        echo "$TABBY_DIR has local changes and the patch isn't applied; refusing to touch it" >&2; exit 1; }
    git -c advice.detachedHead=false checkout -q --detach "$TABBY_COMMIT"
    git apply --check "$ROOT/$TABBY_PATCH"
    git apply "$ROOT/$TABBY_PATCH"
    echo "applied $(basename "$TABBY_PATCH") on $(git rev-parse --short HEAD)"
fi

if [ ! -f config.yml ]; then
    cp "$ROOT/serve/config.example.yml" config.yml
    echo "wrote $TABBY_DIR/config.yml from serve/config.example.yml"
fi

if [ "${NO_INSTALL:-0}" != 1 ]; then
    source "$VENV/bin/activate"
    # TabbyAPI runs from its checkout (python main.py); only its dependencies go into the venv. requirements.txt pins
    # the versions we tested, which cover TabbyAPI's pyproject dependencies. torch and exllamav3 come from
    # make_venv.sh and build_exllamav3.sh, not from TabbyAPI's CUDA wheel pins.
    python -m pip install -r "$ROOT/setup/requirements.txt"
    python -c "import fastapi, uvicorn, sse_starlette, ruamel.yaml, loguru; print('TabbyAPI dependencies OK')"
fi
