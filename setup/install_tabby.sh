#!/bin/bash
# Clone TabbyAPI at the pinned commit (applying TABBY_PATCH if one is set), install its dependencies into the same
# venv, and copy serve/config.example.yml to tabbyAPI/config.yml if there's none yet.
#
# Environment overrides:
#   TABBY_GIT_URL / TABBY_COMMIT / TABBY_PATCH   see setup/versions.sh
#   TABBY_DIR    checkout location (default: tabbyAPI/ in the repo root)
#   RESET=1      discard local changes in TABBY_DIR first (e.g. the old hardware.py patch). config.yml and
#                api_tokens.yml are git-ignored by TabbyAPI; RESET=1 deletes them too, so back them up.
#   VENV         default: .venv/
#   NO_INSTALL=1 clone (and patch) only
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/setup/versions.sh"
source "$ROOT/setup/lib.sh"
TABBY_DIR=${TABBY_DIR:-$ROOT/tabbyAPI}
VENV=${VENV:-$ROOT/.venv}

patches=()
[ -n "$TABBY_PATCH" ] && patches+=("$TABBY_PATCH")
checkout_patched "$TABBY_DIR" "$TABBY_GIT_URL" "$TABBY_COMMIT" "${patches[@]}"

if [ ! -f "$TABBY_DIR/config.yml" ]; then
    cp "$ROOT/serve/config.example.yml" "$TABBY_DIR/config.yml"
    echo "wrote $TABBY_DIR/config.yml from serve/config.example.yml"
fi

if [ "${NO_INSTALL:-0}" != 1 ]; then
    [ -f "$VENV/bin/activate" ] || die "no venv at $VENV; run setup/make_venv.sh first"
    source "$VENV/bin/activate"
    # TabbyAPI runs from its checkout (python main.py); only its dependencies go into the venv. requirements.txt pins
    # the versions we tested, which cover TabbyAPI's pyproject dependencies. torch and exllamav3 come from
    # make_venv.sh and build_exllamav3.sh, not from TabbyAPI's CUDA wheel pins.
    python -m pip install -r "$ROOT/setup/requirements.txt"
    python -c "import fastapi, uvicorn, sse_starlette, ruamel.yaml, loguru; print('TabbyAPI dependencies OK')"
fi
