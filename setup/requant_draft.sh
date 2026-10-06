#!/bin/bash
# Requantize the bf16 DFlash2 draft to EXL3 with the patched exllamav3 converter. Uses the GPU.
#
#   setup/requant_draft.sh            3.0 bpw -> models/Qwen3.8-27B-DFlash2-EXL3-3.0bpw (what we run)
#   setup/requant_draft.sh 3.5        3.5 bpw (not benchmarked)
#   setup/requant_draft.sh --dry-run  print the command only
#
# Source: incoai/Qwen3.8-27B-DFlash2, revision 015e795645c74b1a0eeef3b570031fb62e769bc5, Apache-2.0
# (setup/download_models.sh draft-bf16). model.safetensors is 3,848,817,896 bytes, 1.92B bf16 parameters.
#
# Notes:
# - Upstream exllamav3 dev knows DFlash2DraftModel and skips tokenizer/calibration loading for it, so no target model
#   or tokenizer is needed to convert. The converter disables checkpoint resumption for this architecture, so a
#   failed run has to start over in a fresh work directory.
# - --out_scales always matches the existing Mia-AiLab quant. The selector codebooks stay bf16 and the selector
#   projection fp16. This converter also quantizes the dynamic-convolution projections, which the Mia-AiLab 5.0 bpw
#   file keeps in fp16. Speed through TabbyAPI was the same as with the 5.0 bpw draft (154.3 vs 153.8 tok/s).
# - Result: 0.89 GB on disk vs 1.4 GB for the 5.0 bpw draft.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXLLAMAV3_DIR=${EXLLAMAV3_DIR:-$ROOT/exllamav3}
SOURCE=${SOURCE:-$ROOT/models/Qwen3.8-27B-DFlash2-bf16}

dry_run=false
if [ "${1:-}" = --dry-run ]; then dry_run=true; shift; fi
bits=${1:-3.0}
case $bits in
    3.0|3.5) ;;
    *) echo "usage: $0 [--dry-run] [3.0|3.5]" >&2; exit 2 ;;
esac

output=$ROOT/models/Qwen3.8-27B-DFlash2-EXL3-${bits}bpw
work=$ROOT/requant_work/dflash2-${bits}bpw
cmd=(python "$EXLLAMAV3_DIR/convert.py" -i "$SOURCE" -o "$output" -w "$work" -b "$bits"
     --out_scales always -cb mul1 -d 0)

if $dry_run; then
    printf 'source %q\n' "$ROOT/setup/env.sh"
    printf '%q ' "${cmd[@]}"; printf '\n'
    exit 0
fi

[ -f "$SOURCE/config.json" ] && [ -f "$SOURCE/model.safetensors" ] || {
    echo "missing bf16 source in $SOURCE; run setup/download_models.sh draft-bf16" >&2; exit 1; }
[ -f "$EXLLAMAV3_DIR/convert.py" ] || { echo "missing $EXLLAMAV3_DIR/convert.py; run setup/build_exllamav3.sh" >&2; exit 1; }
# The job can't resume, so never reuse a work dir or overwrite an output
[ ! -e "$output" ] && [ ! -e "$work" ] || {
    echo "$output or $work already exists; remove them to convert again" >&2; exit 1; }

source "$ROOT/setup/env.sh"
mkdir -p "$ROOT/requant_work"
"${cmd[@]}"
echo "draft written to $output (you can delete $work)"
