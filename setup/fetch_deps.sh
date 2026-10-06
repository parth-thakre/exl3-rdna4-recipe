#!/bin/bash
# Fetch the Fedora -devel packages whose headers the exllamav3 build (and Triton at runtime) needs but that a stock
# Fedora 44 ROCm 7.1.1 install lacks, and unpack them into deps/usr. No sudo needed.
#
# With sudo you can skip this script and install them system-wide instead:
#   sudo dnf install hipblaslt-devel hipcub-devel hiprand-devel hipsolver-devel hipsparse-devel hipsparselt-devel \
#       python3.12-devel rocm-smi-devel rocprim-devel rocsolver-devel rocsparse-devel rocthrust-devel
#
# Versions on the test machine (Fedora 44): hipblaslt-devel 7.1.1-7, hipcub-devel 7.1.0-5, hiprand-devel 7.1.0-6,
# hipsolver-devel 7.1.0-5, hipsparse-devel 7.1.1-5, hipsparselt-devel 7.1.1-4, python3.12-devel 3.12.15-1,
# rocm-smi-devel 7.1.1-3, rocprim-devel 7.1.1-3, rocsolver-devel 7.1.1-4, rocsparse-devel 7.1.0-6,
# rocthrust-devel 7.1.1-3.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEPS_DIR=${DEPS_DIR:-$ROOT/deps}

PACKAGES=(hipblaslt-devel hipcub-devel hiprand-devel hipsolver-devel hipsparse-devel hipsparselt-devel
          python3.12-devel rocm-smi-devel rocprim-devel rocsolver-devel rocsparse-devel rocthrust-devel)

for tool in dnf rpm2cpio cpio; do
    command -v "$tool" >/dev/null || { echo "missing $tool (this script is for Fedora)" >&2; exit 1; }
done

mkdir -p "$DEPS_DIR/rpms"
dnf download --arch x86_64 --arch noarch --destdir "$DEPS_DIR/rpms" "${PACKAGES[@]}"

cd "$DEPS_DIR"
for rpm in rpms/*.rpm; do
    echo "unpacking $(basename "$rpm")"
    rpm2cpio "$rpm" | cpio -idm --quiet
done
echo "headers in $DEPS_DIR/usr/include (setup/env.sh and setup/build_exllamav3.sh add it to CPATH)"
