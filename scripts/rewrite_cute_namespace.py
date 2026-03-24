from __future__ import annotations

import argparse
from pathlib import Path


def rewrite_python_files(target_dir: Path) -> int:
    changed = 0
    for path in sorted(target_dir.rglob("*.py")):
        original = path.read_text(encoding="utf-8")
        updated = original.replace("flash_attn.cute", "flash_sparse_attn.ops.cute")
        if path.name == "__init__.py":
            updated = updated.replace('version("fa4")', 'version("flash-sparse-attn")')
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite vendored CuTe imports to the local package namespace.")
    parser.add_argument(
        "target_dir",
        nargs="?",
        default="flash_sparse_attn/ops/cute",
        help="Directory containing the vendored CuTe sources.",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    if not target_dir.exists():
        raise SystemExit(f"Target directory does not exist: {target_dir}")

    changed = rewrite_python_files(target_dir)
    print(f"Rewrote CuTe namespace in {changed} file(s) under {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())