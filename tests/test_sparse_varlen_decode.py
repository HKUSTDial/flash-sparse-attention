import pytest
import torch

from test_utils import run_decode_varlen_case, set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("use_output_buffers", [False, True])
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize(
    "lens_k,num_heads_q,num_heads_kv,head_dim",
    [
        ([53, 97, 129], 8, 4, 64),
        ([1024, 2048], 32, 2, 64),
        ([2048, 4096], 16, 4, 128),
    ],
)
def test_sparse_varlen_decode_correctness(
    use_output_buffers: bool,
    is_local: bool,
    lens_k: list,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> None:
    set_seed(0)
    run_decode_varlen_case(
        kind="sparse",
        lens_k=lens_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        is_local=is_local,
        use_output_buffers=use_output_buffers,
    )
