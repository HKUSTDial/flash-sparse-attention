import pytest
import torch

from test_utils import run_decode_varlen_case, set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("use_output_buffers", [False, True])
def test_dense_varlen_decode_correctness(use_output_buffers: bool) -> None:
    set_seed(0)
    run_decode_varlen_case(
        kind="dense",
        lens_k=[53, 97, 129],
        num_heads_q=8,
        num_heads_kv=4,
        head_dim=64,
        use_output_buffers=use_output_buffers,
    )
