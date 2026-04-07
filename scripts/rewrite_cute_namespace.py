from __future__ import annotations

import argparse
from pathlib import Path


LOCAL_NAMESPACE = "flash_sparse_attn.ops.cute"
UPSTREAM_NAMESPACE = "flash_attn.cute"
LOCAL_PACKAGE_NAME = "flash-sparse-attn"
UPSTREAM_PACKAGE_NAME = "fa4"


def rewrite_python_files(
    target_dir: Path,
    *,
    source_namespace: str,
    destination_namespace: str,
    source_package_name: str,
    destination_package_name: str,
) -> int:
    changed = 0
    for path in sorted(target_dir.rglob("*.py")):
        original = path.read_text(encoding="utf-8")
        updated = original.replace(source_namespace, destination_namespace)
        if path.name == "__init__.py":
            updated = updated.replace(
                f'version("{source_package_name}")',
                f'version("{destination_package_name}")',
            )
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite vendored CuTe imports to the local package namespace.")
    parser.add_argument(
        "--direction",
        choices=("local", "upstream"),
        default="local",
        help="Rewrite imports to the local namespace or back to the upstream namespace.",
    )
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

    if args.direction == "local":
        source_namespace = UPSTREAM_NAMESPACE
        destination_namespace = LOCAL_NAMESPACE
        source_package_name = UPSTREAM_PACKAGE_NAME
        destination_package_name = LOCAL_PACKAGE_NAME
    else:
        source_namespace = LOCAL_NAMESPACE
        destination_namespace = UPSTREAM_NAMESPACE
        source_package_name = LOCAL_PACKAGE_NAME
        destination_package_name = UPSTREAM_PACKAGE_NAME

    changed = rewrite_python_files(
        target_dir,
        source_namespace=source_namespace,
        destination_namespace=destination_namespace,
        source_package_name=source_package_name,
        destination_package_name=destination_package_name,
    )
    print(
        f"Rewrote CuTe namespace from {source_namespace} to {destination_namespace} "
        f"in {changed} file(s) under {target_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())