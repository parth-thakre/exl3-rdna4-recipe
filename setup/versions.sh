# Upstream commits and patch files. This is the one place to bump them.
# Sourced by build_exllamav3.sh, install_tabby.sh and regen_patches.sh. Every value can be overridden from the
# environment, e.g. EXLLAMAV3_URL=/path/to/local/mirror ./setup/build_exllamav3.sh

# ExLlamaV3, dev branch. f1cf869 is the base our patch was made against.
EXLLAMAV3_URL=${EXLLAMAV3_URL:-https://github.com/turboderp-org/exllamav3}
EXLLAMAV3_COMMIT=${EXLLAMAV3_COMMIT:-f1cf8696ca854df49116d9f04e948698024f2c24}
EXLLAMAV3_PATCH=${EXLLAMAV3_PATCH:-patches/exllamav3-rdna4.patch}   # relative to the repo root

# TabbyAPI. Leave TABBY_PATCH empty to skip patching (a TabbyAPI that already accepts ROCm GPUs).
TABBY_URL=${TABBY_URL:-https://github.com/theroyallab/tabbyAPI}
TABBY_COMMIT=${TABBY_COMMIT:-f07131cd8fe34e449fe87cdd3a066b52b96d3cac}
TABBY_PATCH=${TABBY_PATCH-patches/tabbyapi-allow-rdna3-rdna4.patch}

# PyTorch for ROCm. The wheels bundle their own ROCm 7.2 runtime; the system ROCm (7.1.1 on Fedora 44) is only
# used to compile the exllamav3 extension.
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}
