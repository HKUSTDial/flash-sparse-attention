import pytest
import torch

from test_utils import run_backward_base_case, set_seed

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
@pytest.mark.parametrize("is_logsigmoid_gate", [True, False])
def test_gated_base_backward_correctness(
    is_causal: bool,
    is_local: bool,
    is_split_kv: bool,
    is_split_qo: bool,
    head_dim: int,
    is_logsigmoid_gate: bool,
) -> None:
    set_seed(0)
    run_backward_base_case(
        kind="gated",
        batch_size=2,
        seqlen_q=1024,
        seqlen_k=1024,
        num_heads_q=32,
        num_heads_kv=2,
        head_dim=head_dim,
        is_causal=is_causal,
        is_local=is_local,
        is_logsigmoid_gate=is_logsigmoid_gate,
        is_split_kv=is_split_kv,
        is_split_qo=is_split_qo,
    )
