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
        # DIR must be a real directory (or not exist yet); it is never replaced. The only entries ever touched are
        # top-level regular files named like git format-patch output (NNNN-name.patch, plain characters). Anything
        # else named *.patch is ambiguous, so the run refuses before changing anything; other entries are never read.
        d=$features_dir
        while [ "${#d}" -gt 1 ] && [ "${d%/}" != "$d" ]; do d=${d%/}; done
        if [ -L "$d" ] || { [ -e "$d" ] && [ ! -d "$d" ]; }; then
            echo "$features_dir exists and is not a plain directory; refusing to touch it" >&2; exit 1
        fi
        mkdir -p "$d"
        dest=$(cd "$d" && pwd -P)
        patch_re='^[0-9]{4}-[A-Za-z0-9._-]+\.patch$'
        # Staging lives inside DIR (same filesystem, so every move is a rename); cleanup is armed before anything
        # is created. The manifests are written before the first move, so rollback works from them and from what
        # is actually on disk, whatever point an interruption lands on.
        tmp= old= moving= swapped=
        rollback() {
            local n
            [ -n "$moving" ] || return 0   # nothing in DIR has been touched yet
            while IFS= read -r n; do
                grep -qxF -- "$n" "$old/.manifest-old" || rm -f -- "$dest/$n"
            done < "$old/.manifest-new"
            while IFS= read -r n; do
                if [ -e "$old/$n" ]; then mv -T -- "$old/$n" "$dest/$n" || echo "could not restore $n; it is in $old" >&2; fi
            done < "$old/.manifest-old"
            return 0
        }
        cleanup() {
            # Runs from the EXIT trap: keep the script's exit status, and never stop halfway (no errexit in here)
            local rc=$?
            set +e
            if [ -n "$old" ]; then
                [ -n "$swapped" ] || rollback
                if [ -n "$swapped" ]; then rm -rf -- "$old"
                else rm -f -- "$old"/.manifest-new "$old"/.manifest-old "$old"/.list.*; rmdir -- "$old" 2>/dev/null; fi
            fi
            [ -z "$tmp" ] || rm -rf -- "$tmp"
            exit "$rc"
        }
        trap cleanup EXIT
        trap 'exit 130' INT TERM
        tmp=$(mktemp -d "$dest/.regen-new.XXXXXX")
        old=$(mktemp -d "$dest/.regen-old.XXXXXX")
        git -C "$repo" format-patch -q --no-signature -o "$tmp" "$base..$branch" -- . "${excludes[@]}"
        # list_patches DIR OUT: every top-level *.patch entry, checked; fails if listing fails or any name is unsafe
        list_patches() {
            local raw; raw=$(mktemp "$old/.list.XXXXXX")
            find "$1" -mindepth 1 -maxdepth 1 -name '*.patch' -print0 > "$raw" || { echo "cannot list $1" >&2; return 1; }
            : > "$2"
            local p n
            while IFS= read -r -d '' p; do
                n=${p##*/}
                if [[ ! $n =~ $patch_re ]] || [ -L "$p" ] || [ ! -f "$p" ]; then
                    printf '%q is not a plain git patch file; refusing\n' "$p" >&2; return 1
                fi
                printf '%s\n' "$n" >> "$2"
            done < "$raw"
            rm -f -- "$raw"
            sort -o "$2" "$2"
        }
        list_patches "$tmp" "$old/.manifest-new"
        [ -s "$old/.manifest-new" ] || { echo "no commits in $base..$branch; leaving $features_dir alone" >&2; exit 1; }
        list_patches "$dest" "$old/.manifest-old"
        while IFS= read -r n; do
            if [ -e "$dest/$n" ] || [ -L "$dest/$n" ]; then
                grep -qxF -- "$n" "$old/.manifest-old" || { echo "$features_dir/$n is in the way; refusing" >&2; exit 1; }
            fi
        done < "$old/.manifest-new"
        moving=1
        while IFS= read -r n; do mv -T -- "$dest/$n" "$old/$n"; done < "$old/.manifest-old"
        while IFS= read -r n; do mv -T -- "$tmp/$n" "$dest/$n"; done < "$old/.manifest-new"
        swapped=1
        echo "wrote $(wc -l < "$old/.manifest-new") patches to $features_dir (replaced $(wc -l < "$old/.manifest-old")):"
        sed 's/^/  /' "$old/.manifest-new"
        ;;
esac
