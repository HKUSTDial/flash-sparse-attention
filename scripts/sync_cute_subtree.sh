#!/usr/bin/env bash

set -euo pipefail

INIT=0
UPSTREAM_REPO="https://github.com/Dao-AILab/flash-attention.git"
UPSTREAM_PREFIX="flash_attn/cute"
PREFIX="flash_sparse_attn/ops/cute"
TEMP_BRANCH="sync/cute-upstream-temp"
CACHE_DIR=".ref_repo/flash-attention"
SKIP_FETCH=0
KEEP_TEMP_BRANCH=0
NO_TEMPORARY_WORKTREE=0
TEMP_WORKTREE_PATH=""
TEMP_WORKTREE_BRANCH=""
REWRITE_COMMIT_MESSAGE="Rewrite vendored CuTe namespace to flash_sparse_attn.ops.cute"

usage() {
    cat <<'EOF'
Usage: sync_cute_subtree.sh [options]

Options:
  --init                       Perform the first subtree import.
  --upstream-repo <url|path>   Upstream Git URL or local repo path.
  --upstream-prefix <path>     Upstream subdirectory to split.
  --prefix <path>              Destination path in this repo.
  --temp-branch <name>         Temporary branch name used in the upstream cache.
  --cache-dir <path>           Local cache path used when upstream-repo is a URL.
  --skip-fetch                 Skip git fetch origin in the upstream cache.
  --keep-temp-branch           Keep the temporary split branch for debugging.
    --no-temporary-worktree      Fail instead of using a temporary worktree when the current tree is dirty.
  -h, --help                   Show this help message.
EOF
}

is_git_remote_spec() {
    local value="$1"
    [[ "$value" =~ ^[A-Za-z][A-Za-z0-9+.-]*:// ]] || [[ "$value" =~ ^[^[:space:]]+@[^[:space:]:]+:[^[:space:]]+$ ]]
}

invoke_git() {
    if [[ $# -lt 1 ]]; then
        echo "invoke_git requires arguments" >&2
        exit 1
    fi
    git "$@"
}

git_output() {
    if [[ $# -lt 1 ]]; then
        echo "git_output requires arguments" >&2
        exit 1
    fi
    git "$@"
}

test_worktree_clean() {
    local repo="$1"
    local label="$2"
    local status
    status="$(git -C "$repo" status --porcelain)"
    if [[ -n "$status" ]]; then
        echo "$label has uncommitted changes. Commit or stash them before syncing." >&2
        echo "$status" >&2
        exit 1
    fi
}

dirty_status() {
    local repo="$1"
    git -C "$repo" status --porcelain
}

is_git_repo() {
    local repo="$1"
    git -C "$repo" rev-parse --show-toplevel >/dev/null 2>&1
}

ensure_git_identity() {
    local repo="$1"
    local current_name current_email fallback_name fallback_email

    current_name="$(git -C "$repo" config --get user.name || true)"
    current_email="$(git -C "$repo" config --get user.email || true)"

    if [[ -n "$current_name" && -n "$current_email" ]]; then
        return
    fi

    fallback_name="$(git -C "$repo" log -1 --format=%an)"
    fallback_email="$(git -C "$repo" log -1 --format=%ae)"

    if [[ -z "$current_name" ]]; then
        git -C "$repo" config user.name "$fallback_name"
    fi
    if [[ -z "$current_email" ]]; then
        git -C "$repo" config user.email "$fallback_email"
    fi
}

get_commit_subject() {
    local repo="$1"
    local commit="$2"
    git -C "$repo" log -1 --format=%s "$commit"
}

cleanup_worktree() {
    if [[ -n "$TEMP_WORKTREE_PATH" ]]; then
        git -C "$REPO_ROOT" worktree remove --force "$TEMP_WORKTREE_PATH" >/dev/null 2>&1 || true
        TEMP_WORKTREE_PATH=""
    fi
    if [[ -n "$TEMP_WORKTREE_BRANCH" ]]; then
        git -C "$REPO_ROOT" branch -D "$TEMP_WORKTREE_BRANCH" >/dev/null 2>&1 || true
        TEMP_WORKTREE_BRANCH=""
    fi
}

invoke_core_sync() {
    local work_repo_root="$1"
    local cutlass_repo="$work_repo_root/csrc/cutlass"
    local target_path="$work_repo_root/$PREFIX"
    local start_head
    start_head="$(git_output -C "$work_repo_root" rev-parse HEAD)"

    invoke_git -C "$work_repo_root" rev-parse --show-toplevel >/dev/null
    invoke_git -C "$UPSTREAM_REPO_FOR_SPLIT" rev-parse --show-toplevel >/dev/null

    test_worktree_clean "$work_repo_root" "Superproject"
    if [[ -e "$cutlass_repo" ]] && is_git_repo "$cutlass_repo"; then
        test_worktree_clean "$cutlass_repo" "csrc/cutlass submodule"
    fi

    if [[ "$SKIP_FETCH" -eq 0 ]]; then
        echo "Fetching latest upstream changes from origin..."
        invoke_git -C "$UPSTREAM_REPO_FOR_SPLIT" fetch origin
    fi

    echo "Splitting upstream history for $UPSTREAM_PREFIX ..."
    SPLIT_COMMIT="$(git_output -C "$UPSTREAM_REPO_FOR_SPLIT" subtree split --prefix="$UPSTREAM_PREFIX" HEAD | tail -n 1 | tr -d '\r')"
    invoke_git -C "$UPSTREAM_REPO_FOR_SPLIT" branch -f "$TEMP_BRANCH" "$SPLIT_COMMIT"

    cleanup_core() {
        if [[ "$KEEP_TEMP_BRANCH" -eq 0 ]]; then
            git -C "$UPSTREAM_REPO_FOR_SPLIT" update-ref -d "refs/heads/$TEMP_BRANCH" >/dev/null 2>&1 || true
        fi
    }

    trap cleanup_core RETURN

    if [[ "$INIT" -eq 1 ]]; then
        if [[ -e "$target_path" ]]; then
            echo "$PREFIX already exists. Remove --init to do an update instead." >&2
            exit 1
        fi
        echo "Adding subtree into $PREFIX ..."
        invoke_git -C "$work_repo_root" subtree add --prefix="$PREFIX" "$UPSTREAM_REPO_FOR_SPLIT" "$TEMP_BRANCH"
    else
        if [[ ! -e "$target_path" ]]; then
            echo "$PREFIX does not exist yet. Run this script once with --init first." >&2
            exit 1
        fi
        echo "Pulling upstream updates into $PREFIX ..."
        invoke_git -C "$work_repo_root" subtree pull --prefix="$PREFIX" "$UPSTREAM_REPO_FOR_SPLIT" "$TEMP_BRANCH"
    fi

    echo "Rewriting vendored CuTe imports to flash_sparse_attn.ops.cute ..."
    python "$REWRITE_SCRIPT" "$target_path"

    if [[ -n "$(git -C "$work_repo_root" status --porcelain -- "$PREFIX")" ]]; then
        ensure_git_identity "$work_repo_root"
        invoke_git -C "$work_repo_root" add -- "$PREFIX"
        invoke_git -C "$work_repo_root" commit -m "Rewrite vendored CuTe namespace to flash_sparse_attn.ops.cute"
    fi

    END_HEAD="$(git_output -C "$work_repo_root" rev-parse HEAD)"
    SYNC_START_HEAD="$start_head"
    SYNC_END_HEAD="$END_HEAD"
}

invoke_temporary_worktree_sync() {
    local timestamp temp_branch_name temp_worktree original_head stash_name current_status current_prefix_status commits commit cherry_pick_commits apply_rewrite_after_restore
    timestamp="$(date +%Y%m%d-%H%M%S)"
    temp_branch_name="sync/cute-worktree-$timestamp"
    temp_worktree="$(cd "$REPO_ROOT/.." && pwd)/.cute-sync-worktree-$timestamp"
    original_head="$(git_output -C "$REPO_ROOT" rev-parse HEAD)"

    echo "Current worktree is dirty. Syncing in temporary worktree at $temp_worktree ..."
    invoke_git -C "$REPO_ROOT" worktree add -b "$temp_branch_name" "$temp_worktree" "$original_head"
    TEMP_WORKTREE_PATH="$temp_worktree"
    TEMP_WORKTREE_BRANCH="$temp_branch_name"

    invoke_core_sync "$temp_worktree"

    commits="$(git -C "$temp_worktree" rev-list --reverse HEAD "^$original_head")"
    if [[ -z "$commits" ]]; then
        echo "No new subtree commits were created."
        return
    fi

    current_status="$(dirty_status "$REPO_ROOT")"
    current_prefix_status="$(git -C "$REPO_ROOT" status --porcelain -- "$PREFIX")"
    stash_name="sync-cute-autostash-$timestamp"
    if [[ -n "$current_status" ]]; then
        echo "Stashing current worktree before cherry-picking synced commits back ..."
        invoke_git -C "$REPO_ROOT" stash push -u -m "$stash_name"
    fi

    cherry_pick_commits=()
    apply_rewrite_after_restore=0
    while IFS= read -r commit; do
        [[ -z "$commit" ]] && continue
        if [[ -n "$current_prefix_status" ]] && [[ "$(get_commit_subject "$temp_worktree" "$commit")" == "$REWRITE_COMMIT_MESSAGE" ]]; then
            apply_rewrite_after_restore=1
            continue
        fi
        cherry_pick_commits+=("$commit")
    done <<< "$commits"

    for commit in "${cherry_pick_commits[@]}"; do
        echo "Cherry-picking $commit back into current worktree ..."
        ensure_git_identity "$REPO_ROOT"
        invoke_git -C "$REPO_ROOT" cherry-pick "$commit"
    done

    if [[ -n "$current_status" ]]; then
        echo "Restoring stashed local changes ..."
        if ! git -C "$REPO_ROOT" stash pop; then
            echo "Cherry-pick succeeded, but restoring stashed local changes failed. Resolve manually with git stash list / git stash pop." >&2
            exit 1
        fi
    fi

    if [[ "$apply_rewrite_after_restore" -eq 1 ]]; then
        echo "Applying CuTe namespace rewrite in current worktree after restoring local changes ..."
        python "$REWRITE_SCRIPT" "$REPO_ROOT/$PREFIX"
        echo "Namespace rewrite was applied in the current worktree without creating an extra commit because local changes already exist under $PREFIX."
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --init)
            INIT=1
            shift
            ;;
        --upstream-repo)
            UPSTREAM_REPO="$2"
            shift 2
            ;;
        --upstream-prefix)
            UPSTREAM_PREFIX="$2"
            shift 2
            ;;
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --temp-branch)
            TEMP_BRANCH="$2"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --skip-fetch)
            SKIP_FETCH=1
            shift
            ;;
        --keep-temp-branch)
            KEEP_TEMP_BRANCH=1
            shift
            ;;
        --no-temporary-worktree)
            NO_TEMPORARY_WORKTREE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
REWRITE_SCRIPT="$REPO_ROOT/scripts/rewrite_cute_namespace.py"
trap cleanup_worktree EXIT

if is_git_remote_spec "$UPSTREAM_REPO"; then
    CACHE_REPO="$REPO_ROOT/$CACHE_DIR"
    if [[ ! -d "$CACHE_REPO/.git" ]]; then
        echo "Cloning upstream repo into cache at $CACHE_DIR ..."
        invoke_git clone --origin origin "$UPSTREAM_REPO" "$CACHE_REPO"
    fi
    UPSTREAM_REPO_FOR_SPLIT="$CACHE_REPO"
    CURRENT_ORIGIN="$(git_output -C "$UPSTREAM_REPO_FOR_SPLIT" remote get-url origin)"
    if [[ "$CURRENT_ORIGIN" != "$UPSTREAM_REPO" ]]; then
        echo "Updating cached upstream origin URL ..."
        invoke_git -C "$UPSTREAM_REPO_FOR_SPLIT" remote set-url origin "$UPSTREAM_REPO"
    fi
else
    UPSTREAM_REPO_FOR_SPLIT="$(cd "$UPSTREAM_REPO" && pwd)"
fi

SYNC_START_HEAD="$(git_output -C "$REPO_ROOT" rev-parse HEAD)"
SYNC_END_HEAD="$SYNC_START_HEAD"

CURRENT_STATUS="$(dirty_status "$REPO_ROOT")"
if [[ -n "$CURRENT_STATUS" && "$NO_TEMPORARY_WORKTREE" -eq 0 ]]; then
    invoke_temporary_worktree_sync
elif [[ -n "$CURRENT_STATUS" ]]; then
    echo "Superproject has uncommitted changes and --no-temporary-worktree was set." >&2
    echo "$CURRENT_STATUS" >&2
    exit 1
else
    invoke_core_sync "$REPO_ROOT"
fi

echo "Done."
echo "Upstream source: $UPSTREAM_REPO"
echo "Upstream cache used for subtree split: $UPSTREAM_REPO_FOR_SPLIT"
echo "Synced commit range: $SYNC_START_HEAD -> $SYNC_END_HEAD"
echo "Local edits inside $PREFIX stay in this repo and future upstream changes can be merged by rerunning this script without --init."