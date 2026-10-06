# Upstream commits and patch files. This is the one place to bump them.
# Sourced by build_exllamav3.sh, install_tabby.sh and regen_patches.sh. Every value can be overridden from the
# environment, e.g. EXLLAMAV3_GIT_URL=/path/to/local/mirror setup/build_exllamav3.sh
# (TABBY_URL, without GIT, is the running server's address in the bench scripts; it isn't used here.)

# ExLlamaV3, dev branch. Our patch is `git diff <commit> rdna4-dev`; see patches/README.md for how to regenerate it.
EXLLAMAV3_GIT_URL=${EXLLAMAV3_GIT_URL:-https://github.com/turboderp-org/exllamav3}
EXLLAMAV3_COMMIT=${EXLLAMAV3_COMMIT:-0662fac591992c6d5f11d041b1cf2136d62e536d}
EXLLAMAV3_PATCH=${EXLLAMAV3_PATCH:-patches/exllamav3-rdna4.patch}   # paths are relative to the repo root

# Optional: upstream PR #423 (DFlash2 rejection sampling, by @rafatxf), applied after the main patch.
# On by default; WITH_PR423=0 skips it.
WITH_PR423=${WITH_PR423:-1}
PR423_PATCH=${PR423_PATCH:-patches/optional/pr423-dflash2-rejection-sampling.patch}

# TabbyAPI, main branch. 2fd6cc7 accepts ROCm GPUs natively, so no patch. Set TABBY_PATCH to a file to apply one.
TABBY_GIT_URL=${TABBY_GIT_URL:-https://github.com/theroyallab/tabbyAPI}
TABBY_COMMIT=${TABBY_COMMIT:-2fd6cc76203a66e13042daf7d76e5898b21c1ad8}
TABBY_PATCH=${TABBY_PATCH:-}

# PyTorch for ROCm. The wheels ship their own ROCm 7.2 libraries; the system ROCm (7.1.1 on Fedora 44) is only
# used to compile the exllamav3 extension.
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}
