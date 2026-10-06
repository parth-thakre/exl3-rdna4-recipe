#!/bin/bash
# Create the Python 3.12 venv with the exact package versions we tested: torch 2.13.0+rocm7.2, torchvision 0.28.0,
# triton-rocm 3.7.1 from the PyTorch ROCm index, plus the exllamav3 and TabbyAPI dependencies.
# Needs python3.12 (Fedora: sudo dnf install python3.12). The finished venv is about 15 GB, mostly torch's ROCm libraries.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/setup/versions.sh"
VENV=${VENV:-$ROOT/.venv}
PYTHON=${PYTHON:-python3.12}

if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON" -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip

# Plain PyPI packages first (none of them pulls in torch), then torch from the ROCm index so pip can't pick a CUDA build
python -m pip install -r "$ROOT/setup/requirements.txt"
python -m pip install --index-url "$TORCH_INDEX_URL" -r "$ROOT/setup/requirements-torch.txt"

python - <<'EOF'
import torch
print("torch", torch.__version__, "HIP", torch.version.hip)
EOF
echo "venv ready: $VENV"
