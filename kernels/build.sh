#!/bin/bash
# Build the kernel experiments with the system hipcc. Outputs go next to the sources (git-ignored).
# Fedora's hipcc doesn't link the HIP runtime by itself, hence -lamdhip64.
#   kernels/build.sh                         gfx1201 (RX 9070 / 9070 XT)
#   OFFLOAD_ARCH=gfx1200 kernels/build.sh    RX 9060 XT (untested)
set -euo pipefail
cd "$(dirname "$0")"
ARCH=${OFFLOAD_ARCH:-gfx1201}
HIPCC=${HIPCC:-hipcc}
"$HIPCC" --offload-arch="$ARCH" -O3 -o wmma_layout_probe wmma_layout_probe.hip -lamdhip64
"$HIPCC" --offload-arch="$ARCH" -O3 -o graph_setparams_repro graph_setparams_repro.hip -lamdhip64
"$HIPCC" --offload-arch="$ARCH" -O3 -fPIC -shared -o libattn_q4.so attn_q4_hip.hip -lamdhip64
echo "built wmma_layout_probe, graph_setparams_repro, libattn_q4.so for $ARCH"
