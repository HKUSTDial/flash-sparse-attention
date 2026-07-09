#!/usr/bin/env bash
# Upload all PNG assets to a GitHub Release, then update README references.
#
# Usage:
#   ./scripts/upload_assets_to_release.sh [RELEASE_TAG]
#
# Defaults to vX.Y.Z read from flash_sparse_attn/__init__.py if no tag is provided.
# Requires: GitHub CLI, authenticated.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Read version from flash_sparse_attn/__init__.py
VERSION=$(sed -n 's/^__version__ = "\([^"]*\)"/\1/p' "$REPO_ROOT/flash_sparse_attn/__init__.py")
if [[ -z "$VERSION" ]]; then
    echo "Error: Could not read __version__ from flash_sparse_attn/__init__.py"
    exit 1
fi
RELEASE_TAG="${1:-v${VERSION}}"
REPO="HKUSTDial/flash-sparse-attention"
BASE_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}"

echo "==> Collecting PNG files from assets/..."
FILES=()
# Upload logo from root assets/
if [[ -f assets/logo.png ]]; then
    FILES+=("assets/logo.png")
fi
# Upload benchmark images from assets/fsa/
while IFS= read -r -d '' f; do
    FILES+=("$f")
done < <(find assets/fsa -type f -name "*.png" -print0 | sort -z)

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Error: No PNG files found in assets/"
    exit 1
fi

echo "    Found ${#FILES[@]} files to upload."

# Create release
echo "==> Creating release '${RELEASE_TAG}' (if not exists)..."
if gh release view "$RELEASE_TAG" --repo "$REPO" &>/dev/null; then
    echo "    Release '${RELEASE_TAG}' already exists, will upload to it."
else
    gh release create "$RELEASE_TAG" \
        --repo "$REPO" \
        --title "Benchmark Assets (${RELEASE_TAG})" \
        --notes "Static image assets for README benchmark charts. Do not delete this release." \
        --latest=false
    echo "    Release created."
fi

# Upload files
echo "==> Uploading ${#FILES[@]} files to release '${RELEASE_TAG}'..."
for f in "${FILES[@]}"; do
    name=$(basename "$f")
    # Delete existing asset if present (ignore errors if not found)
    gh release delete-asset "$RELEASE_TAG" "$name" --repo "$REPO" -y 2>/dev/null || true
    gh release upload "$RELEASE_TAG" --repo "$REPO" "$f"
    echo "    Uploaded: $name"
done
echo "    Upload complete."

# --- Update READMEs ---
echo "==> Updating README.md and README_zh.md..."

for readme in README.md README_zh.md; do
    if [[ ! -f "$readme" ]]; then
        echo "    Skipping $readme (not found)"
        continue
    fi

    # Pass 1: assets/fsa/filename.png -> BASE_URL/filename.png
    sed -i '' -E 's|(\.\/)?assets/fsa/([^)"]+\.png)|'"${BASE_URL}"'/\2|g' "$readme"

    # Pass 2: assets/filename.png -> BASE_URL/filename.png (no subfolder)
    sed -i '' -E 's|(\.\/)?assets/([^/)"]+\.png)|'"${BASE_URL}"'/\2|g' "$readme"

    echo "    Updated $readme"
done

echo ""
echo "==> Done! Summary:"
echo "    - Release: https://github.com/${REPO}/releases/tag/${RELEASE_TAG}"
echo "    - README images now point to: ${BASE_URL}/..."
echo ""
echo "Next steps:"
echo "  1. Verify the README renders correctly on GitHub."
echo "  2. Optionally remove assets/ from git history:"
echo "     git rm -r assets/"
echo "     git commit -m 'chore: remove assets from repo'"
echo "  3. To fully purge from history:"
echo "     git filter-repo --path assets/ --invert-paths"
echo "     git push --force-with-lease"
