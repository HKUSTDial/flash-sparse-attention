import pytest
import torch

from test_utils import run_forward_base_case, set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("is_causal", [False, True])
def test_dense_base_forward_correctness(is_causal: bool) -> None:
    set_seed(0)
    run_forward_base_case(
        kind="dense",
        batch_size=2,
        seqlen_q=96,
        seqlen_k=128,
        num_heads_q=8,
        num_heads_kv=4,
        head_dim=64,
        is_causal=is_causal,
    )
