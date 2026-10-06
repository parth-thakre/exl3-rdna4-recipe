# Shared helpers for the setup scripts. Source it; don't run it.

die() { echo "error: $*" >&2; exit 1; }

# is_checkout DIR: DIR is the top level of a git work tree (a clone or a worktree, where .git is a file)
is_checkout() {
    [ -d "$1" ] || return 1
    local top
    top=$(git -C "$1" rev-parse --show-toplevel 2>/dev/null) || return 1
    [ "$top" = "$(cd "$1" && pwd -P)" ]
}

# worktree_tree DIR: the git tree hash of DIR's working tree as it is on disk (tracked files plus untracked files that
# aren't ignored), computed in a throwaway index so the real index is untouched
worktree_tree() {
    local dir=$1 tmpd tree
    tmpd=$(mktemp -d)
    GIT_INDEX_FILE=$tmpd/index git -C "$dir" read-tree HEAD &&
        GIT_INDEX_FILE=$tmpd/index git -C "$dir" add -A . &&
        tree=$(GIT_INDEX_FILE=$tmpd/index git -C "$dir" write-tree)
    local rc=$?
    rm -rf "$tmpd"
    [ $rc -eq 0 ] && echo "$tree"
    return $rc
}

# checkout_patched DIR URL COMMIT [PATCH...]
# Clone URL into DIR if needed, then make DIR exactly COMMIT plus PATCHes (paths relative to $ROOT, applied in order).
# A marker in the checkout's git dir records the inputs and the resulting tree hash. A rerun with the same inputs
# changes nothing, but only if the files on disk still hash to that tree: an edited, restored or added file is caught.
# If DIR holds anything else it stops, unless RESET=1, which always discards local changes and untracked/ignored
# files (including build outputs) and starts over.
checkout_patched() {
    local dir=$1 url=$2 commit=$3; shift 3
    local patches=("$@") p
    for p in "${patches[@]}"; do [ -f "$ROOT/$p" ] || die "missing patch $p"; done

    if [ -e "$dir" ] && ! is_checkout "$dir"; then
        die "$dir exists but isn't a git checkout; move it away or set another directory"
    fi
    if [ ! -e "$dir" ]; then
        git clone "$url" "$dir"
    fi
    git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$dir" fetch origin
    local want head marker stamp
    want=$(git -C "$dir" rev-parse --verify --quiet "$commit^{commit}") || die "commit $commit not found in $url"
    marker=$(git -C "$dir" rev-parse --path-format=absolute --git-path recipe-applied)
    stamp=$( { echo "$want"; for p in "${patches[@]}"; do echo "$p $(sha256sum < "$ROOT/$p" | cut -d' ' -f1)"; done; } )

    if [ "${RESET:-0}" = 1 ]; then
        echo "$(basename "$dir"): RESET=1, discarding local changes and untracked files"
        git -C "$dir" reset -q --hard
        git -C "$dir" clean -qfdx
        rm -f "$marker"
    fi

    head=$(git -C "$dir" rev-parse HEAD)
    if [ "$head" = "$want" ] && [ -f "$marker" ] && [ "$(head -n -1 "$marker")" = "$stamp" ]; then
        local expected actual
        expected=$(tail -n 1 "$marker")
        actual=$(worktree_tree "$dir") || die "can't hash the working tree of $dir"
        if [ "$actual" = "$expected" ]; then
            echo "$(basename "$dir"): already at ${want:0:7} with ${#patches[@]} patch(es) applied, files unchanged"
            return 0
        fi
        die "$dir was patched by this script but files have changed since (edited, restored or added).
       Rerun with RESET=1 to discard the changes and patch afresh (this also deletes build outputs)."
    fi
    if [ -n "$(git -C "$dir" status --porcelain)" ]; then
        die "$dir has local changes or other patches applied. Rerun with RESET=1 to discard them (this also deletes build outputs)."
    fi
    rm -f "$marker"
    git -C "$dir" -c advice.detachedHead=false checkout -q --detach "$want"
    for p in "${patches[@]}"; do
        git -C "$dir" apply --check "$ROOT/$p" || die "$p doesn't apply on ${want:0:7}"
        git -C "$dir" apply "$ROOT/$p"
        echo "$(basename "$dir"): applied $p"
    done
    local tree
    tree=$(worktree_tree "$dir") || die "can't hash the working tree of $dir"
    { echo "$stamp"; echo "$tree"; } > "$marker"
    echo "$(basename "$dir"): at ${want:0:7} with ${#patches[@]} patch(es)"
}
