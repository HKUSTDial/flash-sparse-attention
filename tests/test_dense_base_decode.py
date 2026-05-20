import pytest
import torch

from test_utils import run_decode_base_case, set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("use_output_buffers", [False, True])
@pytest.mark.parametrize("is_local", [False, True])
def test_dense_base_decode_correctness(
    use_output_buffers: bool,
    is_local: bool,
) -> None:
    set_seed(0)
    run_decode_base_case(
        kind="dense",
        batch_size=2,
        seqlen_k=1024,
        num_heads_q=32,
        num_heads_kv=2,
        head_dim=64,
        is_local=is_local,
        use_output_buffers=use_output_buffers,
    )
