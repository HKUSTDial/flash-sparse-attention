import pytest
import torch

from test_utils import run_backward_varlen_case, set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize(
    "is_causal,is_local",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("is_split_kv", [False, True])
@pytest.mark.parametrize("is_split_qo", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_sparse_varlen_backward_correctness(
    is_causal: bool, is_local: bool, is_split_kv: bool, is_split_qo: bool, head_dim: int
) -> None:
    set_seed(0)
    run_backward_varlen_case(
        kind="sparse",
        lens_q=[1024, 2048, 4096],
        lens_k=[1024, 2048, 4096],
        num_heads_q=32,
        num_heads_kv=2,
        head_dim=head_dim,
        is_causal=is_causal,
        is_local=is_local,
        is_split_kv=is_split_kv,
        is_split_qo=is_split_qo,
    )
