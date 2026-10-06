#!/bin/bash
# XL profile: MTP head, 160k context. A full prompt peaks at 16.24 of 16.3 GB, so nothing else can use the GPU.
# One request at a time: with replay off, extra slots would each hold the full DeltaNet history and not fit.
EXL3_GDN_REPLAY=0 exec "$(cd "$(dirname "$0")" && pwd)/run_tabby.sh" \
    --draft-mode mtp --max-seq-len 163840 --cache-size 163840 --max-batch-size 1 "$@"
