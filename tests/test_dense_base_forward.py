import pytest

from test_utils import supported_device_is_available, run_forward_base_case, set_seed

pytestmark = pytest.mark.skipif(
    not supported_device_is_available(), reason="A supported device is required"
)


@pytest.mark.parametrize(
    "is_causal,is_local",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("is_split_kv", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("pack_gqa", [False, True])
def test_dense_base_forward_correctness(
    is_causal: bool, is_local: bool, is_split_kv: bool, head_dim: int, pack_gqa: bool
) -> None:
    set_seed(0)
    run_forward_base_case(
        kind="dense",
        batch_size=2,
        seqlen_q=1024,
        seqlen_k=1024,
        num_heads_q=32,
        num_heads_kv=2,
        head_dim=head_dim,
        is_causal=is_causal,
        is_local=is_local,
        is_split_kv=is_split_kv,
        pack_gqa=pack_gqa,
    )
