import pytest
import torch

from test_utils import run_decode_varlen_case, set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("use_output_buffers", [False, True])
@pytest.mark.parametrize("is_local", [False, True])
def test_gated_varlen_decode_correctness(
    use_output_buffers: bool,
    is_local: bool,
) -> None:
    set_seed(0)
    run_decode_varlen_case(
        kind="gated",
        lens_k=[1024, 2048, 4096],
        num_heads_q=32,
        num_heads_kv=2,
        head_dim=64,
        is_local=is_local,
        is_logsigmoid_gate=True,
        use_output_buffers=use_output_buffers,
    )
