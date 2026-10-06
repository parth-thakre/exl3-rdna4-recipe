#!/bin/bash
# Long profile: draft with the model's built-in MTP head instead of DFlash2 (~0.9 GB less VRAM), 128k context.
# Replay is off here because we haven't tested it with MTP.
# One request at a time: with replay off, every slot holds the full DeltaNet history (about 2.44 GB extra at the default 3 slots).
EXL3_GDN_REPLAY=0 exec "$(cd "$(dirname "$0")" && pwd)/run_tabby.sh" \
    --draft-mode mtp --max-seq-len 131072 --cache-size 131072 --max-batch-size 1 "$@"
