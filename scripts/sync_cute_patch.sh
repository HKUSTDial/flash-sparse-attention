#!/usr/bin/env bash
# Generate per-file git patches for the flash_attn/cute subdirectory from a
# given commit hash (or the hash stored in .hash) up to the latest main HEAD.
# Each changed file gets its own .patch file under the output directory.
#
# Usage:
#   sync_cute_patch.sh [options]
#
# Options:
#   --from <commit>    Starting commit hash (exclusive). If omitted, reads from .hash file.
#   --cache-dir <dir>  Local clone cache path (default: .ref_repo/flash-attention).
#   --skip-fetch       Skip fetching latest upstream changes.
#   --outdir <path>    Output directory for per-file patches (default: patches/cute_<from>..<to>/).
#   -h, --help         Show this help message.

set -euo pipefail

UPSTREAM_REPO="https://github.com/Dao-AILab/flash-attention.git"
UPSTREAM_PREFIX="flash_attn/cute"
CACHE_DIR=".ref_repo/flash-attention"
HASH_FILE="flash_sparse_attn/ops/cute/.hash"
FROM_COMMIT=""
OUTPUT_DIR=""
SKIP_FETCH=0

usage() {
    cat <<'EOF'
Usage: sync_cute_patch.sh [options]

Generate per-file cute-subdirectory git patches from a base commit to upstream main HEAD.

Options:
  --from <commit>    Starting commit hash (exclusive). Defaults to value in .hash file.
  --cache-dir <dir>  Local clone cache path (default: .ref_repo/flash-attention).
  --skip-fetch       Skip fetching latest upstream changes.
  --outdir <path>    Output directory for per-file patches.
  -h, --help         Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            FROM_COMMIT="$2"
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
        --outdir)
            OUTPUT_DIR="$2"
            shift 2
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

# Resolve repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HASH_FILE_ABS="$REPO_ROOT/$HASH_FILE"

# Determine FROM_COMMIT
if [[ -z "$FROM_COMMIT" ]]; then
    if [[ -f "$HASH_FILE_ABS" && -s "$HASH_FILE_ABS" ]]; then
        FROM_COMMIT="$(tr -d '[:space:]' < "$HASH_FILE_ABS")"
        echo "Read base commit from $HASH_FILE: $FROM_COMMIT"
    else
        echo "Error: No --from commit specified and $HASH_FILE is empty or missing." >&2
        echo "Provide a starting commit with --from <hash>." >&2
        exit 1
    fi
fi

# Ensure upstream clone exists
CACHE_REPO="$REPO_ROOT/$CACHE_DIR"
if [[ ! -d "$CACHE_REPO/.git" ]]; then
    echo "Cloning upstream repo into cache at $CACHE_DIR ..."
    git clone --origin origin "$UPSTREAM_REPO" "$CACHE_REPO"
fi

# Fetch latest
if [[ "$SKIP_FETCH" -eq 0 ]]; then
    echo "Fetching latest upstream changes ..."
    git -C "$CACHE_REPO" fetch origin
fi

# Resolve main branch HEAD
MAIN_REF="$(git -C "$CACHE_REPO" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null || echo "origin/main")"
TO_COMMIT="$(git -C "$CACHE_REPO" rev-parse "$MAIN_REF")"
TO_SHORT="$(git -C "$CACHE_REPO" rev-parse --short "$MAIN_REF")"

echo "Generating patch: $FROM_COMMIT -> $TO_COMMIT"
echo "  (subdirectory: $UPSTREAM_PREFIX)"

# Validate FROM_COMMIT exists in upstream
if ! git -C "$CACHE_REPO" cat-file -e "$FROM_COMMIT^{commit}" 2>/dev/null; then
    echo "Error: FROM_COMMIT $FROM_COMMIT not found in upstream repo." >&2
    exit 1
fi

# Check if there are any changes
DIFF_STAT="$(git -C "$CACHE_REPO" diff --stat "$FROM_COMMIT" "$TO_COMMIT" -- "$UPSTREAM_PREFIX")"
if [[ -z "$DIFF_STAT" ]]; then
    echo "No changes in $UPSTREAM_PREFIX between $FROM_COMMIT and $TO_COMMIT."
    echo "Updating .hash to $TO_COMMIT anyway."
    printf '%s\n' "$TO_COMMIT" > "$HASH_FILE_ABS"
    exit 0
fi

echo "Changes detected:"
echo "$DIFF_STAT"
echo ""

# Determine output directory
FROM_SHORT="$(git -C "$CACHE_REPO" rev-parse --short "$FROM_COMMIT")"
PATCHES_SUBDIR="patches/cute_${FROM_SHORT}..${TO_SHORT}"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="flash_sparse_attn/ops/cute"
fi
# Resolve to absolute path
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
fi
PATCHES_PATH="$OUTPUT_DIR/$PATCHES_SUBDIR"

# Clean and create output directory
rm -rf "$PATCHES_PATH"
mkdir -p "$PATCHES_PATH"

# Get list of changed files in the cute subdirectory
CHANGED_FILES="$(git -C "$CACHE_REPO" diff --name-only "$FROM_COMMIT" "$TO_COMMIT" -- "$UPSTREAM_PREFIX")"

# Generate one patch per file
FILE_COUNT=0
while IFS= read -r filepath; do
    [[ -z "$filepath" ]] && continue

    # Derive patch filename: replace '/' with '__' to flatten path
    patch_name="${filepath//\//__}.patch"
    patch_path="$PATCHES_PATH/$patch_name"

    git -C "$CACHE_REPO" diff "$FROM_COMMIT" "$TO_COMMIT" -- "$filepath" > "$patch_path"
    FILE_COUNT=$((FILE_COUNT + 1))
done <<< "$CHANGED_FILES"

echo "Generated $FILE_COUNT patch files in: $PATCHES_PATH"

# Update .hash with the new TO_COMMIT (main HEAD)
printf '%s\n' "$TO_COMMIT" > "$HASH_FILE_ABS"
echo "Updated $HASH_FILE -> $TO_COMMIT"

echo ""
echo "Done."
echo "  From: $FROM_COMMIT"
echo "  To:   $TO_COMMIT"
echo "  Patches: $PATCHES_PATH/ ($FILE_COUNT files)"
echo ""
echo "Next run will use $TO_COMMIT as the base commit (read from $HASH_FILE)."
