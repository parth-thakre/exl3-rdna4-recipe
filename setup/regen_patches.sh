#!/bin/bash
# Regenerate a combined patch from a local git checkout: the diff from an upstream base to a work branch, source files
# only. Use it to rebase the recipe onto a newer upstream: port the work branch, regenerate, then bump
# setup/versions.sh.
#
#   setup/regen_patches.sh BASE BRANCH                   exllamav3/ -> patches/exllamav3-rdna4.patch
#   setup/regen_patches.sh --repo DIR --out FILE BASE BRANCH
#
# e.g. setup/regen_patches.sh f1cf869 rdna4-work
#      setup/regen_patches.sh --repo tabbyAPI --out patches/tabbyapi-allow-rdna3-rdna4.patch f07131c HEAD
# For a TabbyAPI patch kept as uncommitted changes, use BRANCH=WORKTREE (diffs BASE against the working tree).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/setup/versions.sh"
repo=$ROOT/exllamav3
out=$ROOT/$EXLLAMAV3_PATCH
while [ $# -gt 0 ]; do
    case $1 in
        --repo) repo=$2; shift 2 ;;
        --out)  out=$2; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) break ;;
    esac
done
[ $# -eq 2 ] || { echo "usage: $0 [--repo DIR] [--out FILE] BASE BRANCH" >&2; exit 2; }
base=$1 branch=$2

# Build outputs and caches never belong in the patch
excludes=(':(exclude)build' ':(exclude)*.so' ':(exclude)*.o' ':(exclude)*.hip' ':(exclude)**/__pycache__'
          ':(exclude)*.egg-info' ':(exclude)**/__disk_lru_cache__')
# hipify writes *.hip next to each .cu; a real .hip source would be dropped too, so warn about any tracked one
if [ "$branch" = WORKTREE ]; then
    range=("$base")
else
    range=("$base" "$branch")
fi
tracked_hip=$(git -C "$repo" diff --name-only "${range[@]}" -- '*.hip' || true)
[ -z "$tracked_hip" ] || echo "warning: excluded tracked .hip changes: $tracked_hip" >&2

git -C "$repo" diff "${range[@]}" -- . "${excludes[@]}" > "$out.tmp"
mv "$out.tmp" "$out"
echo "wrote $out ($(grep -c '^diff --git' "$out") files):"
git -C "$repo" diff --stat "${range[@]}" -- . "${excludes[@]}" | tail -1
grep '^diff --git' "$out" | awk '{print "  " substr($3, 3)}'
