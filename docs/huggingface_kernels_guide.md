# HuggingFace Kernel Hub Build Guide

Package and publish Flash Sparse Attention Triton kernels to [HuggingFace Kernel Hub](https://huggingface.co/docs/kernels/index).

## Prerequisites

- [Nix](https://nixos.org/download.html)
- kernel-builder:
  ```bash
  curl -fsSL https://raw.githubusercontent.com/huggingface/kernels/main/install.sh | bash
  ```
- HuggingFace authentication:
  ```bash
  hf auth login
  ```

## Build

### Generate kernel package

```bash
python scripts/build_hf_kernels.py --clean
cd huggingface_kernels
```

### Configure Nix environment

```bash
export NIX_BUILD_CORES=1
export NIX_CONFIG="max-jobs = 1
extra-substituters = https://huggingface.cachix.org
extra-trusted-public-keys = huggingface.cachix.org-1:ynTPbLS0W8ofXd9fDjk1KvoFky9K2jhxe6r4nXAkc/o=
"
```

### Build and Upload

```bash
kernel-builder build-and-upload
```

## Usage

```python
from kernels import get_kernel

fsa = get_kernel("JingzeShi/flash-sparse-attention", version=1, trust_remote_code=True)

# Dense forward
out = fsa.flash_dense_attn_func(q, k, v, is_causal=True)

# Sparse attention
out = fsa.flash_sparse_attn_func(q, k, v, is_causal=True, softmax_threshold=0.01)

# Gated attention
out = fsa.flash_gated_attn_func(q, k, v, alpha, delta, is_causal=True)

# Decode with KV cache
out = fsa.flash_dense_attn_with_kvcache_func(q, k, v)
```
