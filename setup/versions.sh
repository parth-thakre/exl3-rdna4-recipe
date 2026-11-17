# Upstream commits and patch files. This is the one place to bump them.
# Sourced by build_exllamav3.sh, install_tabby.sh and regen_patches.sh. Every value can be overridden from the
# environment, e.g. EXLLAMAV3_GIT_URL=/path/to/local/mirror setup/build_exllamav3.sh
# (TABBY_URL, without GIT, is the running server's address in the bench scripts; it isn't used here.)

# ExLlamaV3 v1.6.7 (release tag). Our patch is `git diff <commit> live-167` (our local work branch); see
# patches/README.md for how to regenerate it. The previous base is still available (patches/README.md, "Previous base"):
#   EXLLAMAV3_COMMIT=0662fac591992c6d5f11d041b1cf2136d62e536d EXLLAMAV3_PATCH=patches/previous/exllamav3-rdna4-0662fac.patch
EXLLAMAV3_GIT_URL=${EXLLAMAV3_GIT_URL:-https://github.com/turboderp-org/exllamav3}
EXLLAMAV3_COMMIT=${EXLLAMAV3_COMMIT:-8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0}
EXLLAMAV3_PATCH=${EXLLAMAV3_PATCH:-patches/exllamav3-rdna4.patch}   # paths are relative to the repo root

# Optional: upstream PR #423 (DFlash2 rejection sampling, by @rafatxf), applied after the main patch. Off by default
# (patches/README.md). Only ported to the previous base so far: WITH_PR423=1 needs the two previous-base settings above.
WITH_PR423=${WITH_PR423:-0}
PR423_PATCH=${PR423_PATCH:-patches/previous/pr423-dflash2-rejection-sampling-0662fac.patch}

# TabbyAPI, main branch (2026-11-14). It accepts ROCm GPUs natively (since 2fd6cc7), so no patch. Set TABBY_PATCH to a file to apply one.
TABBY_GIT_URL=${TABBY_GIT_URL:-https://github.com/theroyallab/tabbyAPI}
TABBY_COMMIT=${TABBY_COMMIT:-47281587fafbd69e50aaf77c7626f390811126b6}
TABBY_PATCH=${TABBY_PATCH:-}

# PyTorch for ROCm. The wheels ship their own ROCm 7.2 libraries; the system ROCm (7.1.1 on Fedora 44) is only
# used to compile the exllamav3 extension.
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}
