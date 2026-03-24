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

CUTLASS_REPO="$REPO_ROOT/csrc/cutlass"
TARGET_PATH="$REPO_ROOT/$PREFIX"

invoke_git rev-parse --show-toplevel >/dev/null
invoke_git -C "$UPSTREAM_REPO_FOR_SPLIT" rev-parse --show-toplevel >/dev/null

test_worktree_clean "$REPO_ROOT" "Superproject"
if [[ -d "$CUTLASS_REPO/.git" || -d "$CUTLASS_REPO" ]]; then
    if git -C "$CUTLASS_REPO" rev-parse --show-toplevel >/dev/null 2>&1; then
        test_worktree_clean "$CUTLASS_REPO" "csrc/cutlass submodule"
    fi
fi

if [[ "$SKIP_FETCH" -eq 0 ]]; then
    echo "Fetching latest upstream changes from origin..."
    invoke_git -C "$UPSTREAM_REPO_FOR_SPLIT" fetch origin
fi

echo "Splitting upstream history for $UPSTREAM_PREFIX ..."
SPLIT_COMMIT="$(git_output -C "$UPSTREAM_REPO_FOR_SPLIT" subtree split --prefix="$UPSTREAM_PREFIX" HEAD | tail -n 1 | tr -d '\r')"
invoke_git -C "$UPSTREAM_REPO_FOR_SPLIT" branch -f "$TEMP_BRANCH" "$SPLIT_COMMIT"

cleanup() {
    if [[ "$KEEP_TEMP_BRANCH" -eq 0 ]]; then
        git -C "$UPSTREAM_REPO_FOR_SPLIT" update-ref -d "refs/heads/$TEMP_BRANCH" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ "$INIT" -eq 1 ]]; then
    if [[ -e "$TARGET_PATH" ]]; then
        echo "$PREFIX already exists. Remove --init to do an update instead." >&2
        exit 1
    fi
    echo "Adding subtree into $PREFIX ..."
    invoke_git subtree add --prefix="$PREFIX" "$UPSTREAM_REPO_FOR_SPLIT" "$TEMP_BRANCH"
else
    if [[ ! -e "$TARGET_PATH" ]]; then
        echo "$PREFIX does not exist yet. Run this script once with --init first." >&2
        exit 1
    fi
    echo "Pulling upstream updates into $PREFIX ..."
    invoke_git subtree pull --prefix="$PREFIX" "$UPSTREAM_REPO_FOR_SPLIT" "$TEMP_BRANCH"
fi

echo "Done."
echo "Upstream source: $UPSTREAM_REPO"
echo "Upstream cache used for subtree split: $UPSTREAM_REPO_FOR_SPLIT"
echo "Local edits inside $PREFIX stay in this repo and future upstream changes can be merged by rerunning this script without --init."