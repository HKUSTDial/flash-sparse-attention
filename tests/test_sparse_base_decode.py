import pytest
import torch

from test_utils import run_decode_base_case, set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("use_output_buffers", [False, True])
def test_sparse_base_decode_correctness(use_output_buffers: bool) -> None:
    set_seed(0)
    run_decode_base_case(
        kind="sparse",
        batch_size=2,
        seqlen_k=128,
        num_heads_q=8,
        num_heads_kv=4,
        head_dim=64,
        use_output_buffers=use_output_buffers,
    )
