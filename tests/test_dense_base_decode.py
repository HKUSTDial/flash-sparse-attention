import pytest
import torch

from test_utils import run_decode_base_case, set_seed

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("use_output_buffers", [False, True])
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize(
    "seqlen_k,num_heads_q,num_heads_kv,head_dim",
    [
        (1024, 32, 2, 64),
        (2048, 8, 8, 64),
        (4096, 16, 4, 128),
    ],
)
def test_dense_base_decode_correctness(
    use_output_buffers: bool,
    is_local: bool,
    seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> None:
    set_seed(0)
    run_decode_base_case(
        kind="dense",
        batch_size=2,
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        is_local=is_local,
        use_output_buffers=use_output_buffers,
    )
