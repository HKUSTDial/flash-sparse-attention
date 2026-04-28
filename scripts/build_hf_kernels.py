"""
Build HuggingFace Kernel Hub compatible package from flash_sparse_attn Triton kernels.

Usage:
    python scripts/build_hf_kernels.py [--output-dir DIR] [--repo-id ID] [--version N] [--clean] [--dry-run]
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
TRITON_SRC = REPO_ROOT / "flash_sparse_attn" / "ops" / "triton"
PKG_NAME = "flash-sparse-attention"
PKG_IMPORT_NAME = "flash_sparse_attention"


PUBLIC_FUNCTIONS = [
    "flash_dense_attn_func",
    "flash_dense_attn_with_kvcache_func",
    "flash_dense_attn_varlen_func",
    "flash_dense_attn_varlen_with_kvcache_func",
    "flash_sparse_attn_func",
    "flash_sparse_attn_with_kvcache_func",
    "flash_sparse_attn_varlen_func",
    "flash_sparse_attn_varlen_with_kvcache_func",
    "flash_gated_attn_func",
    "flash_gated_attn_with_kvcache_func",
    "flash_gated_attn_varlen_func",
    "flash_gated_attn_varlen_with_kvcache_func",
]


KERNEL_FILES = [
    "interface.py",
    "utils.py",
    "activations.py",
    "assert_inputs.py",
    "block_info.py",
    "cache_utils.py",
    "seqlen_info.py",
    "mask.py",
    "launch_grid.py",
    "launch_template.py",
    "flash_fwd_combine.py",
    "flash_dec_combine.py",
    "flash_bwd_preprocess.py",
    "flash_bwd_postprocess.py",
    "flash_dense_fwd.py",
    "flash_dense_bwd.py",
    "flash_dense_dec.py",
    "flash_sparse_fwd.py",
    "flash_sparse_bwd.py",
    "flash_sparse_dec.py",
    "flash_gated_fwd.py",
    "flash_gated_bwd.py",
    "flash_gated_dec.py",
]


def rewrite_imports(source: str) -> str:
    """Rewrite absolute triton imports to relative imports."""
    # Pattern 1: from flash_sparse_attn.ops.triton.module import ...
    #          → from .module import ...
    source = re.sub(
        r"from flash_sparse_attn\.ops\.triton\.(\w+) import",
        r"from .\1 import",
        source,
    )
    # Pattern 2: from flash_sparse_attn.ops.triton import mod1, mod2
    #          → from . import mod1, mod2
    source = re.sub(
        r"from flash_sparse_attn\.ops\.triton import",
        "from . import",
        source,
    )
    return source


def generate_init() -> str:
    funcs = "\n".join(f"    {f}," for f in PUBLIC_FUNCTIONS)
    all_list = "\n".join(f'    "{f}",' for f in PUBLIC_FUNCTIONS)
    return f"""from .interface import (
{funcs}
)

__all__ = [
{all_list}
]
"""


def generate_build_toml(repo_id: str, version: int) -> str:
    return f"""[general]
name = "{PKG_NAME}"
version = {version}
license = "BSD-3-Clause"
backends = ["cuda"]

[general.hub]
repo-id = "{repo_id}"

[torch-noarch]

[kernel]
"""


def generate_flake_nix() -> str:
    return """{
  description = "Flash Sparse Attention Triton Kernels";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
      # triton.autotune fails in GPU-less build sandbox
      doGetKernelCheck = false;
    };
}
"""


def generate_card(repo_id: str) -> str:
    funcs_list = "\n".join(f"- `{f}`" for f in PUBLIC_FUNCTIONS)
    return f"""---
library_name: kernels
license: bsd-3-clause
---

# {PKG_NAME}

Flash Sparse Attention Triton kernels — dense, sparse, and gated attention
with forward, backward, and decode paths.

## Usage

```python
from kernels import get_kernel

fsa = get_kernel("{repo_id}", version=1)

# Dense forward
out = fsa.flash_dense_attn_func(q, k, v, is_causal=True)

# Decode with KV cache
out = fsa.flash_dense_attn_with_kvcache_func(q, k, v)

# Sparse attention
out = fsa.flash_sparse_attn_func(q, k, v, is_causal=True, threshold=0.01)

# Gated attention
out = fsa.flash_gated_attn_func(q, k, v, alpha, delta, is_causal=True)
```

## Available functions

{funcs_list}

## Source

Originally from [HKUSTDial/flash-sparse-attention](https://github.com/HKUSTDial/flash-sparse-attention).
"""


def generate_tests() -> str:
    return f"""import pytest
import torch


@pytest.mark.kernels_ci
def test_dense_forward():
    from {PKG_IMPORT_NAME} import flash_dense_attn_func

    B, S, H, D = 2, 128, 8, 64
    q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    out = flash_dense_attn_func(q, k, v, is_causal=True)
    assert out.shape == (B, S, H, D)
    assert not torch.isnan(out).any()


@pytest.mark.kernels_ci
def test_sparse_forward():
    from {PKG_IMPORT_NAME} import flash_sparse_attn_func

    B, S, H, D = 2, 128, 8, 64
    q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    out = flash_sparse_attn_func(q, k, v, is_causal=True, softmax_threshold=0.0)
    assert out.shape == (B, S, H, D)


@pytest.mark.kernels_ci
def test_dense_decode():
    from {PKG_IMPORT_NAME} import flash_dense_attn_with_kvcache_func

    B, H, D, S_kv = 2, 8, 64, 256
    q = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
    k_cache = torch.randn(B, S_kv, H, D, dtype=torch.float16, device="cuda")
    v_cache = torch.randn(B, S_kv, H, D, dtype=torch.float16, device="cuda")
    out = flash_dense_attn_with_kvcache_func(q, k_cache, v_cache, is_causal=False)
    assert out.shape == (B, H, D)
"""


def check_no_absolute_imports(pkg_dir: Path) -> list[str]:
    errors = []
    pattern = re.compile(r"flash_sparse_attn\.ops\.triton")
    for f in pkg_dir.glob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if pattern.search(line) and not line.strip().startswith("#"):
                errors.append(f"  {f.name}:{i}: {line.strip()}")
    return errors


def check_all_files_present(pkg_dir: Path) -> list[str]:
    missing = []
    for fname in KERNEL_FILES:
        if not (pkg_dir / fname).exists():
            missing.append(fname)
    return missing


def check_init_exports(pkg_dir: Path) -> list[str]:
    init_text = (pkg_dir / "__init__.py").read_text()
    missing = [f for f in PUBLIC_FUNCTIONS if f not in init_text]
    return missing


def run_checks(out_dir: Path) -> bool:
    pkg_dir = out_dir / "torch-ext" / PKG_IMPORT_NAME
    ok = True

    missing_files = check_all_files_present(pkg_dir)
    if missing_files:
        print(f"FAIL  Missing files: {missing_files}")
        ok = False
    else:
        print(f"OK    All {len(KERNEL_FILES)} kernel files present")

    abs_imports = check_no_absolute_imports(pkg_dir)
    if abs_imports:
        print(f"FAIL  Absolute imports remain ({len(abs_imports)} occurrences):")
        for line in abs_imports:
            print(line)
        ok = False
    else:
        print("OK    No absolute imports (all relative)")

    missing_exports = check_init_exports(pkg_dir)
    if missing_exports:
        print(f"FAIL  __init__.py missing exports: {missing_exports}")
        ok = False
    else:
        print(f"OK    __init__.py exports all {len(PUBLIC_FUNCTIONS)} public functions")

    for fname in ["build.toml", "flake.nix", "CARD.md"]:
        if not (out_dir / fname).exists():
            print(f"FAIL  Missing {fname}")
            ok = False
        else:
            print(f"OK    {fname} present")

    return ok


def build(
    out_dir: Path, repo_id: str, version: int, clean: bool, dry_run: bool
) -> None:
    if dry_run:
        print("Dry-run mode: checking source only, no files written.")
        src_missing = [f for f in KERNEL_FILES if not (TRITON_SRC / f).exists()]
        if src_missing:
            print(f"FAIL  Source files missing: {src_missing}")
            sys.exit(1)
        print(f"OK    All {len(KERNEL_FILES)} source files found in {TRITON_SRC}")
        return

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"Cleaned {out_dir}")

    pkg_dir = out_dir / "torch-ext" / PKG_IMPORT_NAME
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tests").mkdir(exist_ok=True)

    # Copy + rewrite kernel files
    copied = 0
    for fname in KERNEL_FILES:
        src = TRITON_SRC / fname
        if not src.exists():
            print(f"WARNING: {fname} not found in source, skipping")
            continue
        content = rewrite_imports(src.read_text())
        (pkg_dir / fname).write_text(content)
        copied += 1
    print(f"Copied and rewrote imports for {copied} files")

    # Generate __init__.py
    (pkg_dir / "__init__.py").write_text(generate_init())

    # Generate config files
    (out_dir / "build.toml").write_text(generate_build_toml(repo_id, version))
    (out_dir / "flake.nix").write_text(generate_flake_nix())
    (out_dir / "CARD.md").write_text(generate_card(repo_id))

    # Generate tests
    (out_dir / "tests" / "__init__.py").write_text("")
    (out_dir / "tests" / "test_flash_attn.py").write_text(generate_tests())

    print(f"\nGenerated structure in {out_dir}/\n")

    # Run checks
    print("Running compliance checks...")
    passed = run_checks(out_dir)

    if passed:
        print(f"""
All checks passed.

To publish to HuggingFace Hub:

  1. Install kernel-builder (if not already):
       curl -fsSL https://raw.githubusercontent.com/huggingface/kernels/main/install.sh | bash

  2. Login to HuggingFace:
       hf auth login

  3. Build the kernel package:
       cd {out_dir}
       export NIX_BUILD_CORES=1
       export NIX_CONFIG="max-jobs = 1
       extra-substituters = https://huggingface.cachix.org
       extra-trusted-public-keys = huggingface.cachix.org-1:ynTPbLS0W8ofXd9fDjk1KvoFky9K2jhxe6r4nXAkc/o=
       "
       kernel-builder build-and-copy -L

  4. Upload to Hub:
       kernel-builder upload --repo-type model
""")
    else:
        print("\nSome checks failed. Fix the issues above before publishing.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HuggingFace Kernel Hub package")
    parser.add_argument(
        "--output-dir", default="huggingface_kernels", help="Output directory"
    )
    parser.add_argument(
        "--repo-id", default="JingzeShi/flash-sparse-attention", help="Hub repo ID"
    )
    parser.add_argument("--version", type=int, default=1, help="Kernel version")
    parser.add_argument(
        "--clean", action="store_true", help="Remove output dir before building"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check source only, no output"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    build(out_dir, args.repo_id, args.version, args.clean, args.dry_run)


if __name__ == "__main__":
    main()
