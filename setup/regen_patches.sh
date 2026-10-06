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
# Check everything before writing anything: the repo, both refs, and the .hip guard
git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || { echo "$repo is not a git checkout" >&2; exit 1; }
git -C "$repo" rev-parse --verify --quiet "$base^{commit}" >/dev/null || { echo "unknown BASE: $base" >&2; exit 1; }
if [ "$branch" = WORKTREE ]; then
    [ "$mode" = diff ] || { echo "WORKTREE only works for a plain diff" >&2; exit 2; }
    range=("$base")
else
    git -C "$repo" rev-parse --verify --quiet "$branch^{commit}" >/dev/null || { echo "unknown BRANCH: $branch" >&2; exit 1; }
    range=("$base" "$branch")
fi
tracked_hip=$(git -C "$repo" diff --name-only "${range[@]}" -- '*.hip')
[ -z "$tracked_hip" ] || echo "warning: excluded tracked .hip changes: $tracked_hip" >&2

# Generate into a temporary file or directory next to the destination; replace the destination only on success
tmp=
trap '[ -n "$tmp" ] && rm -rf "$tmp"' EXIT
case $mode in
    diff)
        mkdir -p "$(dirname "$out")"
        tmp=$(mktemp "$out.XXXXXX")
        git -C "$repo" diff "${range[@]}" -- . "${excludes[@]}" > "$tmp"
        [ -s "$tmp" ] || { echo "empty diff for ${range[*]}; not writing $out" >&2; exit 1; }
        mv "$tmp" "$out"; tmp=
        echo "wrote $out ($(grep -c '^diff --git' "$out") files)"
        git -C "$repo" diff --stat "${range[@]}" -- . "${excludes[@]}" | tail -1
        ;;
    mail)
        mkdir -p "$(dirname "$out")"
        tmp=$(mktemp "$out.XXXXXX")
        git -C "$repo" format-patch --stdout --no-signature "$base..$branch" -- . "${excludes[@]}" > "$tmp"
        [ -s "$tmp" ] || { echo "no commits in $base..$branch; not writing $out" >&2; exit 1; }
        mv "$tmp" "$out"; tmp=
        echo "wrote $out ($(grep -c '^From [0-9a-f]\{40\} ' "$out") commit(s), $(grep -c '^diff --git' "$out") file diffs)"
        ;;
    features)
        # DIR must be a real directory (or not exist yet). It is never replaced or re-created: only its top-level
        # regular *.patch files are swapped, so other entries, permissions and ownership are left alone.
        if [ -L "$features_dir" ] || { [ -e "$features_dir" ] && [ ! -d "$features_dir" ]; }; then
            echo "$features_dir exists and is not a plain directory; refusing to touch it" >&2; exit 1
        fi
        mkdir -p "$features_dir"
        dest=$(cd "$features_dir" && pwd -P)
        # Staging lives inside DIR (same filesystem, so every move below is a rename); its names aren't *.patch
        tmp=$(mktemp -d "$dest/.regen-new.XXXXXX")
        old=$(mktemp -d "$dest/.regen-old.XXXXXX")
        git -C "$repo" format-patch -q --no-signature -o "$tmp" "$base..$branch" -- . "${excludes[@]}"
        mapfile -t new_names < <(find "$tmp" -mindepth 1 -maxdepth 1 -type f -name '*.patch' -printf '%f\n' | sort)
        [ ${#new_names[@]} -gt 0 ] || { echo "no commits in $base..$branch; leaving $features_dir alone" >&2; exit 1; }
        mapfile -t old_names < <(find "$dest" -mindepth 1 -maxdepth 1 -type f -name '*.patch' -printf '%f\n' | sort)
        # Rollback puts the old set back and removes whatever part of the new set got in. It runs on any failure
        # and on interruption until the swap has finished; afterwards cleanup only removes the empty staging dirs.
        moved_old=() moved_new=() swapped=
        rollback() {
            local n
            for n in "${moved_new[@]}"; do rm -f "$dest/$n"; done
            for n in "${moved_old[@]}"; do mv -T "$old/$n" "$dest/$n" || echo "could not restore $n; it is in $old" >&2; done
        }
        cleanup() {
            [ -n "$swapped" ] || rollback
            rm -rf "$tmp"
            # After a swap $old holds the replaced patches, which are no longer wanted; after a rollback it is empty
            # unless a restore failed, and then it is kept so nothing is lost
            if [ -n "$swapped" ]; then rm -rf "$old"; else rmdir "$old" 2>/dev/null || true; fi
        }
        trap cleanup EXIT
        trap 'exit 130' INT TERM
        for n in "${old_names[@]}"; do mv -T "$dest/$n" "$old/$n" || exit 1; moved_old+=("$n"); done
        for n in "${new_names[@]}"; do
            [ ! -e "$dest/$n" ] && [ ! -L "$dest/$n" ] || { echo "$features_dir/$n is in the way (not a regular patch file)" >&2; exit 1; }
            mv -T "$tmp/$n" "$dest/$n" || exit 1; moved_new+=("$n")
        done
        swapped=1
        echo "wrote ${#new_names[@]} patches to $features_dir (replaced ${#old_names[@]}):"
        printf '  %s\n' "${new_names[@]}"
        ;;
esac
