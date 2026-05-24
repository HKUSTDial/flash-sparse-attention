import pytest
import torch

from test_utils import run_backward_varlen_case, set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize(
    "is_causal,is_local",
    [(False, False), (True, False), (False, True)],
)
def test_dense_varlen_backward_correctness(is_causal: bool, is_local: bool) -> None:
    set_seed(0)
    run_backward_varlen_case(
        kind="dense",
        lens_q=[1024, 2048, 4096],
        lens_k=[1024, 2048, 4096],
        num_heads_q=32,
        num_heads_kv=2,
        head_dim=64,
        is_causal=is_causal,
        is_local=is_local,
    )
