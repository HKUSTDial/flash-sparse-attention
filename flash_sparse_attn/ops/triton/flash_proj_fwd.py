"""
Requires: triton-kernels == 3.6
    git clone https://github.com/triton-lang/kernels.git && cd kernels
    git checkout v3.6.0
    pip install -e .
"""

import torch
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FlexCtx
from triton_kernels.numerics import InFlexData, OutFlexData


def _flash_proj_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.float8_e5m2,
    prev_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seqlen, in_features = input.shape
    out_features = weight.shape[0]
    device = input.device

    x = input.view(batch_size * seqlen, in_features)

    if prev_scale is None:
        expected_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    else:
        expected_scale = prev_scale.view(1).float()

    actual_scale = torch.zeros(1, dtype=torch.float32, device=device)

    out_flex = OutFlexData(
        dtype=out_dtype,
        expected_scale=expected_scale,
        actual_scale=actual_scale,
        checksum_scale=None,
    )

    precision_config = PrecisionConfig(
        flex_ctx=FlexCtx(
            lhs_data=InFlexData(),
            rhs_data=InFlexData(),
            out_data=out_flex,
        ),
        out_dtype=out_dtype,
    )

    out = matmul_ogs(x, weight.T, None, precision_config=precision_config)

    descale = precision_config.flex_ctx.out_data.actual_scale.clone()

    return out.view(batch_size, seqlen, out_features), descale
