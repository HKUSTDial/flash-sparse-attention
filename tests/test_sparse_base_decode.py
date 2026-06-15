import pytest

from test_utils import supported_device_is_available, run_decode_base_case, set_seed

pytestmark = pytest.mark.skipif(
    not supported_device_is_available(), reason="A supported device is required"
)


@pytest.mark.parametrize("use_output_buffers", [False, True])
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_quant", [False, True])
def test_sparse_base_decode_correctness(
    use_output_buffers: bool,
    is_local: bool,
    head_dim: int,
    is_quant: bool,
) -> None:
    set_seed(0)
    run_decode_base_case(
        kind="sparse",
        batch_size=2,
        seqlen_k=1024,
        num_heads_q=32,
        num_heads_kv=2,
        head_dim=head_dim,
        is_local=is_local,
        use_output_buffers=use_output_buffers,
        is_quant=is_quant,
    )
