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
def test_gated_varlen_backward_correctness(is_causal: bool, is_local: bool) -> None:
    set_seed(0)
    run_backward_varlen_case(
        kind="gated",
        lens_q=[19, 27, 41],
        lens_k=[25, 31, 43],
        num_heads_q=8,
        num_heads_kv=4,
        head_dim=64,
        is_causal=is_causal,
        is_local=is_local,
        is_logsigmoid_gate=True,
    )
