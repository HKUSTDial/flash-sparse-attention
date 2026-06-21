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


# HuggingFace kernel-builder supported backends
HF_SUPPORTED_BACKENDS = ["cuda", "rocm", "metal", "cpu", "xpu"]

# Mapping from FlagGems vendor names to HF kernel-builder backends.
VENDOR_TO_BACKEND = {
    "nvidia": "cuda",       # NVIDIA GPU
    "amd": "rocm",          # AMD GPU
    "hygon": "rocm",        # Hygon DCU
    "intel": "xpu",         # Intel GPU
    "kunlunxin": "xpu",     # KunlunXin
    "metax": "cuda",        # MetaX GPU
    "iluvatar": "cuda",     # Iluvatar CoreX
    "cambricon": "cuda",    # Cambricon MLU
    "mthreads": "cuda",     # Moore Threads MUSA
    "apple": "metal",       # Apple Silicon
    "arm": "cpu",           # ARM CPU
    "spacemit": "cpu",      # SpaceMIT RISC-V
}

# Extended backends beyond HF kernel-builder defaults.
# These may require custom Triton backends or vendor-specific toolchains.
EXTENDED_BACKENDS = {
    "ascend": {
        "device_name": "npu",
        "vendor": "Huawei Ascend",
        "triton_extra": "ascend",
        "dispatch_key": "PrivateUse1",
    },
    "musa": {
        "device_name": "musa",
        "vendor": "Moore Threads",
        "triton_extra": None,
        "dispatch_key": "MUSA",
    },
    "mlu": {
        "device_name": "mlu",
        "vendor": "Cambricon",
        "triton_extra": None,
        "dispatch_key": "MLU",
    },
    "gcu": {
        "device_name": "gcu",
        "vendor": "Enflame",
        "triton_extra": None,
        "dispatch_key": "GCU",
    },
    "npu": {
        "device_name": "npu",
        "vendor": "Huawei Ascend",
        "triton_extra": "ascend",
        "dispatch_key": "PrivateUse1",
    },
}


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


KERNEL_FILES_EXCLUDE = {"__init__.py"}


def get_kernel_files() -> list[str]:
    """Dynamically discover all .py files under ops/triton (excluding __init__.py).

    This avoids hardcoding file lists that go stale as the kernel set evolves.
    Files are sorted for deterministic output.
    """
    if not TRITON_SRC.is_dir():
        return []
    return sorted(
        f.name
        for f in TRITON_SRC.glob("*.py")
        if f.name not in KERNEL_FILES_EXCLUDE
    )


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


def resolve_backends(backends: list[str] | None) -> list[str]:
    """Resolve backend list, validating against HF kernel-builder supported set.

    If backends is None, defaults to ["cuda"].
    Accepts both HF backend names (cuda, rocm, ...) and FlagGems vendor names
    (nvidia, amd, hygon, ...) which are mapped to HF backends automatically.
    """
    if not backends:
        return ["cuda"]

    resolved = []
    for b in backends:
        b_lower = b.lower().strip()
        # Direct HF backend name
        if b_lower in HF_SUPPORTED_BACKENDS:
            if b_lower not in resolved:
                resolved.append(b_lower)
        # FlagGems vendor name → HF backend
        elif b_lower in VENDOR_TO_BACKEND:
            mapped = VENDOR_TO_BACKEND[b_lower]
            if mapped not in resolved:
                resolved.append(mapped)
        else:
            print(f"WARNING: Unknown backend '{b}', skipping. "
                  f"Supported: {HF_SUPPORTED_BACKENDS} or vendor names: "
                  f"{list(VENDOR_TO_BACKEND.keys())}")
    return resolved if resolved else ["cuda"]


def generate_build_toml(repo_id: str, version: int, backends: list[str] | None = None) -> str:
    backend_list = resolve_backends(backends)
    backends_str = ", ".join(f'"{b}"' for b in backend_list)
    return f"""[general]
name = "{PKG_NAME}"
version = {version}
license = "BSD-3-Clause"
backends = [{backends_str}]

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


def generate_card(repo_id: str, backends: list[str] | None = None) -> str:
    funcs_list = "\n".join(f"- `{f}`" for f in PUBLIC_FUNCTIONS)
    resolved = resolve_backends(backends)
    backends_list = "\n".join(f"- `{b}`" for b in resolved)
    return f"""---
library_name: kernels
license: bsd-3-clause
---

# {PKG_NAME}

Flash Sparse Attention Triton kernels — dense, sparse, and gated attention
with forward, backward, and decode paths.

## Supported backends

{backends_list}

## Usage

```python
from kernels import get_kernel

fsa = get_kernel("{repo_id}", version=1, trust_remote_code=True)

# Dense forward
out = fsa.flash_dense_attn_func(q, k, v, is_causal=True)

# Decode with KV cache
out = fsa.flash_dense_attn_with_kvcache_func(q, k, v)

# Sparse attention
out = fsa.flash_sparse_attn_func(q, k, v, is_causal=True, softmax_threshold=0.01)

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
def test_dense_decode():
    from {PKG_IMPORT_NAME} import flash_dense_attn_with_kvcache_func

    B, H, D, S_kv = 2, 8, 64, 256
    q = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, S_kv, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, S_kv, H, D, dtype=torch.float16, device="cuda")
    out = flash_dense_attn_with_kvcache_func(q, k, v)
    assert out.shape == (B, H, D)


@pytest.mark.kernels_ci
def test_sparse_forward():
    from {PKG_IMPORT_NAME} import flash_sparse_attn_func

    B, S, H, D = 2, 128, 8, 64
    q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    out = flash_sparse_attn_func(q, k, v, is_causal=True, softmax_threshold=0.01)
    assert out.shape == (B, S, H, D)
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
    kernel_files = get_kernel_files()
    for fname in kernel_files:
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
        print(f"OK    All {len(get_kernel_files())} kernel files present")

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
    out_dir: Path, repo_id: str, version: int, clean: bool, dry_run: bool,
    backends: list[str] | None = None,
) -> None:
    if dry_run:
        print("Dry-run mode: checking source only, no files written.")
        kernel_files = get_kernel_files()
        if not kernel_files:
            print(f"FAIL  No kernel files found in {TRITON_SRC}")
            sys.exit(1)
        print(f"OK    All {len(kernel_files)} source files found in {TRITON_SRC}")
        resolved = resolve_backends(backends)
        print(f"OK    Backends: {resolved}")
        return

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"Cleaned {out_dir}")

    pkg_dir = out_dir / "torch-ext" / PKG_IMPORT_NAME
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tests").mkdir(exist_ok=True)

    # Copy + rewrite kernel files
    kernel_files = get_kernel_files()
    copied = 0
    for fname in kernel_files:
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
    resolved = resolve_backends(backends)
    (out_dir / "build.toml").write_text(generate_build_toml(repo_id, version, resolved))
    (out_dir / "flake.nix").write_text(generate_flake_nix())
    (out_dir / "CARD.md").write_text(generate_card(repo_id, resolved))

    # Generate tests
    (out_dir / "tests" / "__init__.py").write_text("")
    (out_dir / "tests" / "test_flash_attn.py").write_text(generate_tests())

    print(f"\nGenerated structure in {out_dir}/")
    print(f"Backends: {resolved}\n")

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
        "--repo-id", default="JingzeShi/flash-sparse-attn", help="Hub repo ID"
    )
    parser.add_argument("--version", type=int, default=1, help="Kernel version")
    parser.add_argument(
        "--clean", action="store_true", help="Remove output dir before building"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Check source only, no output"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help=(
            "Target backends for kernel-builder. "
            f"HF backends: {HF_SUPPORTED_BACKENDS}. "
            f"Also accepts vendor names: {list(VENDOR_TO_BACKEND.keys())}. "
            "Default: cuda"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    build(out_dir, args.repo_id, args.version, args.clean, args.dry_run, args.backends)


if __name__ == "__main__":
    main()
