"""
Requires: triton-kernels == 3.6
    git clone https://github.com/triton-lang/kernels.git && cd kernels
    git checkout v3.6.0
    pip install -e .
"""

import torch
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig


def _flash_proj_backward(
    dout: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seqlen, out_features = dout.shape
    in_features = input.shape[-1]

    dy = dout.view(batch_size * seqlen, out_features)
    x = input.view(batch_size * seqlen, in_features)

    precision_config = PrecisionConfig(out_dtype=dy.dtype)

    dx = matmul_ogs(dy, weight, None, precision_config=precision_config)
    dw = matmul_ogs(dy.T, x, None, precision_config=precision_config)

    return dx.view(batch_size, seqlen, in_features), dw
