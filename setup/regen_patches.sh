#!/bin/bash
# Regenerate patches from a local git checkout of the work branches, source files only. Use it after porting the work
# onto a newer upstream: regenerate, then bump setup/versions.sh.
#
#   setup/regen_patches.sh [--repo DIR] [--out FILE] BASE BRANCH        combined diff BASE..BRANCH (plain git diff)
#   setup/regen_patches.sh --format-patch [--repo DIR] --out FILE BASE BRANCH
#                                                                       the commits as mail patches (keeps authors)
#   setup/regen_patches.sh --features DIR [--repo DIR] BASE BRANCH      one numbered file per commit into DIR
#   BRANCH=WORKTREE diffs BASE against the working tree (plain mode only).
#
# What we ran for the current files, with WORK = an exllamav3 checkout that has branches rdna4-dev and
# rdna4-dev-pr423 (the PR #423 commit rebased on rdna4-dev):
#   setup/regen_patches.sh --repo "$WORK" 0662fac rdna4-dev
#   setup/regen_patches.sh --features patches/features --repo "$WORK" 0662fac rdna4-dev
#   setup/regen_patches.sh --format-patch --repo "$WORK" \
#       --out patches/optional/pr423-dflash2-rejection-sampling.patch rdna4-dev rdna4-dev-pr423
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/setup/versions.sh"
repo=$ROOT/exllamav3
out=$ROOT/$EXLLAMAV3_PATCH
mode=diff
features_dir=
while [ $# -gt 0 ]; do
    case $1 in
        --repo) repo=$2; shift 2 ;;
        --out)  out=$2; shift 2 ;;
        --format-patch) mode=mail; shift ;;
        --features) mode=features; features_dir=$2; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[ $# -eq 2 ] || { echo "usage: $0 [--repo DIR] [--out FILE | --format-patch --out FILE | --features DIR] BASE BRANCH" >&2; exit 2; }
base=$1 branch=$2

# Build outputs and caches never belong in a patch. hipify writes *.hip next to each .cu; exllamav3 doesn't track
# any .hip file, and we warn below if that ever changes.
excludes=(':(exclude)build' ':(exclude)*.so' ':(exclude)*.o' ':(exclude)*.hip' ':(exclude)**/__pycache__'
          ':(exclude)*.egg-info' ':(exclude)**/__disk_lru_cache__')
if [ "$branch" = WORKTREE ]; then
    [ "$mode" = diff ] || { echo "WORKTREE only works for a plain diff" >&2; exit 2; }
    range=("$base")
else
    range=("$base" "$branch")
fi
tracked_hip=$(git -C "$repo" diff --name-only "${range[@]}" -- '*.hip' || true)
[ -z "$tracked_hip" ] || echo "warning: excluded tracked .hip changes: $tracked_hip" >&2

case $mode in
    diff)
        git -C "$repo" diff "${range[@]}" -- . "${excludes[@]}" > "$out.tmp"
        mv "$out.tmp" "$out"
        echo "wrote $out ($(grep -c '^diff --git' "$out") files)"
        git -C "$repo" diff --stat "${range[@]}" -- . "${excludes[@]}" | tail -1
        ;;
    mail)
        mkdir -p "$(dirname "$out")"
        git -C "$repo" format-patch --stdout --no-signature "$base..$branch" -- . "${excludes[@]}" > "$out.tmp"
        mv "$out.tmp" "$out"
        echo "wrote $out ($(grep -c '^From [0-9a-f]\{40\} ' "$out") commit(s), $(grep -c '^diff --git' "$out") file diffs)"
        ;;
    features)
        mkdir -p "$features_dir"
        rm -f "$features_dir"/*.patch
        git -C "$repo" format-patch --no-signature -o "$(cd "$features_dir" && pwd)" "$base..$branch" -- . "${excludes[@]}"
        ;;
esac
