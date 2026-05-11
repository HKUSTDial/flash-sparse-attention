import pytest
import torch

from flash_sparse_attn.ops.triton.cache_utils import get_device_arch
from test_utils import run_decode_base_case, set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_is_hopper = (
    torch.cuda.is_available() and get_device_arch(torch.device("cuda")) // 10 >= 9
)


def _dtypes():
    dtypes = [torch.float16, torch.bfloat16]
    if _is_hopper:
        dtypes.append(torch.float8_e5m2)
    return dtypes


@pytest.mark.parametrize("dtype", _dtypes(), ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("use_output_buffers", [False, True])
def test_dense_base_decode_correctness(
    dtype: torch.dtype, use_output_buffers: bool
) -> None:
    set_seed(0)
    run_decode_base_case(
        kind="dense",
        batch_size=2,
        seqlen_k=128,
        num_heads_q=8,
        num_heads_kv=4,
        head_dim=64,
        use_output_buffers=use_output_buffers,
        dtype=dtype,
    )


@pytest.mark.skipif(not _is_hopper, reason="Auto-quant requires Hopper (SM90+)")
@pytest.mark.parametrize("use_output_buffers", [False, True])
def test_dense_base_decode_auto_quant(use_output_buffers: bool) -> None:
    set_seed(0)
    run_decode_base_case(
        kind="dense",
        batch_size=2,
        seqlen_k=128,
        num_heads_q=8,
        num_heads_kv=4,
        head_dim=64,
        use_output_buffers=use_output_buffers,
        dtype=torch.bfloat16,
        is_quant=True,
    )
