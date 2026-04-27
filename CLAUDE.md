# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flash-Sparse-Attention is a high-performance trainable sparse attention library built on Triton (and optionally CUTLASS/CUTE). It provides dense, sparse (softmax-thresholded), and gated attention kernels with forward, backward, and decode (KV-cache) paths. Requires NVIDIA GPU with compute capability >= 8.0.

## Commands

### Install
```bash
pip install -e ".[dev]"          # editable install with test deps
pip install -e ".[dev,cute]"     # also install CUTE/CUTLASS deps
```

### Tests
Tests require CUDA. Run from the repo root:
```bash
python -m pytest tests                              # all tests
python -m pytest tests/test_dense_base_forward.py    # single file
python -m pytest tests/test_dense_base_forward.py -s -q  # with stdout, quiet
python -m pytest tests/test_dense_base_forward.py::test_dense_base_forward_correctness  # single test
```
Tests import `test_utils` from the `tests/` directory (not a package — pytest adds it to sys.path). No conftest.py exists.

### Lint & Format
```bash
make style                # ruff check --fix + ruff format (whole codebase)
make fixup                # same but only on files changed vs main
make quality              # lint + format check + tests (CI gate)
```

### Benchmarks
```bash
python tests/benchmark_forward.py
python tests/benchmark_backward.py
python tests/benchmark_decode.py
```

## Architecture

### Kernel Organization (`flash_sparse_attn/ops/triton/`)

Three kernel families, each with three operation files:
- `flash_dense_{fwd,bwd,dec}.py` — standard dense attention
- `flash_sparse_{fwd,bwd,dec}.py` — sparse attention (softmax threshold)
- `flash_gated_{fwd,bwd,dec}.py` — gated attention (alpha/delta gate inputs)

Each `_dec.py` file contains multiple kernel variants:
- SM80 kernel (Ampere+, standard path)
- SM90 kernel (Hopper, FP8 support via `torch.float8_e5m2`)
- Split-KV kernels for long-sequence decode
- Both base (padded batch) and varlen (variable-length via `cu_seqlens`) dispatch functions

Shared utilities: `block_info.py`, `cache_utils.py`, `assert_inputs.py`, `activations.py`, `flash_fwd_combine.py`, `flash_dec_combine.py`, `utils.py`.

### Interface Layer (`flash_sparse_attn/ops/triton/interface.py`)

Single file exposing 12 public functions (the full public API):
- `flash_{dense,sparse,gated}_attn_func` — training forward+backward (autograd Functions)
- `flash_{dense,sparse,gated}_attn_with_kvcache_func` — decode with KV cache
- `flash_{dense,sparse,gated}_attn_varlen_func` — varlen training
- `flash_{dense,sparse,gated}_attn_varlen_with_kvcache_func` — varlen decode

All 12 are re-exported from `flash_sparse_attn/__init__.py`.

### Tensor Layouts

- Training kernels: `(batch, seqlen, num_heads, head_dim)` — "bshd"
- Decode kernels: query is `(batch, num_heads, head_dim)` (seqlen_q=1 squeezed), KV is `(batch, seqlen_k, num_kv_heads, head_dim)`
- SDPA-style benchmarks use `(batch, num_heads, seqlen, head_dim)` — "bhsd"

### CUTE Backend (`flash_sparse_attn/ops/cute/`)

Alternative CUTLASS-based implementation (requires `[cute]` extras). Separate interface at `ops/cute/interface.py`. Not part of the main Triton kernel path.

### Test Structure (`tests/`)

Test files follow the pattern `test_{dense,sparse,gated}_{base,varlen}_{forward,backward,decode}.py`. All heavy lifting is in `tests/test_utils.py` which provides:
- `run_forward_base_case`, `run_backward_base_case`, `run_decode_case` (and varlen variants)
- Reference implementations using PyTorch SDPA
- `BenchmarkConfig` / `BenchmarkResult` dataclasses
- FP8 quantization helper `_quantize_per_tensor_fp8`

### FP8 Support

FP8 decode is available on SM90+ (Hopper). Detection: `torch.float8_e5m2` dtype triggers SM90 kernel dispatch. Scales (`query_scale`, `key_scale`, `value_scale`) are passed through the interface functions. Output is always `bfloat16` when input is FP8.
