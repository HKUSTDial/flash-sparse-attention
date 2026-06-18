import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate
from triton.runtime.jit import constexpr_function

from flash_sparse_attn.ops.triton import mask
from flash_sparse_attn.ops.triton import activations
from flash_sparse_attn.ops.triton import seqlen_info, block_info


@aggregate
class AttnFwdGridIndex:
    m_block: tl.tensor
    batch_idx: tl.tensor
    head_idx: tl.tensor
    head_kv_idx: tl.tensor
    split_idx: tl.tensor

    @constexpr_function
    def __init__(self, m_block, batch_idx, head_idx, head_kv_idx, split_idx):
        self.m_block = m_block
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx

    @staticmethod
    @triton.jit
    def create(
        num_splits,
        QHEAD_PER_KVHEAD: tl.constexpr,
        IS_SPLIT_KV: tl.constexpr,
        PACK_GQA: tl.constexpr,
    ):
        m_block = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_split_idx = tl.program_id(2)
        if IS_SPLIT_KV:
            batch_idx = batch_split_idx // num_splits
            split_idx = batch_split_idx - batch_idx * num_splits
        else:
            batch_idx = batch_split_idx
            split_idx = 0
        if PACK_GQA:
            head_kv_idx = head_idx
        else:
            head_kv_idx = head_idx // QHEAD_PER_KVHEAD
        return AttnFwdGridIndex(m_block, batch_idx, head_idx, head_kv_idx, split_idx)

    @triton.jit
    def load_window_sizes(self, window_sizes, stride_wh, IS_LOCAL: tl.constexpr):
        if IS_LOCAL:
            window_size_left = tl.load(window_sizes + self.head_kv_idx * stride_wh)
            window_size_right = tl.load(window_sizes + self.head_kv_idx * stride_wh + 1)
        else:
            window_size_left = 0
            window_size_right = 0
        return window_size_left, window_size_right


@aggregate
class AttnBwdGridIndex:
    n_block: tl.tensor
    batch_idx: tl.tensor
    head_idx: tl.tensor
    head_kv_idx: tl.tensor
    split_idx: tl.tensor

    @constexpr_function
    def __init__(self, n_block, batch_idx, head_idx, head_kv_idx, split_idx):
        self.n_block = n_block
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx

    @staticmethod
    @triton.jit
    def create(
        num_splits,
        QHEAD_PER_KVHEAD: tl.constexpr,
        IS_SPLIT_QO: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        head_idx = tl.program_id(1)
        batch_split_idx = tl.program_id(2)
        if IS_SPLIT_QO:
            batch_idx = batch_split_idx // num_splits
            split_idx = batch_split_idx - batch_idx * num_splits
        else:
            batch_idx = batch_split_idx
            split_idx = 0
        head_kv_idx = head_idx // QHEAD_PER_KVHEAD
        return AttnBwdGridIndex(n_block, batch_idx, head_idx, head_kv_idx, split_idx)

    @triton.jit
    def load_window_sizes(self, window_sizes, stride_wh, IS_LOCAL: tl.constexpr):
        if IS_LOCAL:
            window_size_left = tl.load(window_sizes + self.head_kv_idx * stride_wh)
            window_size_right = tl.load(window_sizes + self.head_kv_idx * stride_wh + 1)
        else:
            window_size_left = 0
            window_size_right = 0
        return window_size_left, window_size_right


@aggregate
class AttnDecGridIndex:
    batch_idx: tl.tensor
    head_idx: tl.tensor
    head_kv_idx: tl.tensor
    split_idx: tl.tensor

    @constexpr_function
    def __init__(self, batch_idx, head_idx, head_kv_idx, split_idx):
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx

    @staticmethod
    @triton.jit
    def create(
        num_splits,
    ):
        head_idx = tl.program_id(0)
        batch_split_idx = tl.program_id(1)
        batch_idx = batch_split_idx // num_splits
        split_idx = batch_split_idx - batch_idx * num_splits
        head_kv_idx = head_idx
        return AttnDecGridIndex(batch_idx, head_idx, head_kv_idx, split_idx)

    @triton.jit
    def load_window_sizes(self, window_sizes, stride_wh, IS_LOCAL: tl.constexpr):
        if IS_LOCAL:
            window_size_left = tl.load(window_sizes + self.head_kv_idx * stride_wh)
            window_size_right = tl.load(window_sizes + self.head_kv_idx * stride_wh + 1)
        else:
            window_size_left = 0
            window_size_right = 0
        return window_size_left, window_size_right


@aggregate
class AttnFwdConfig:
    softmax_scale_log2: tl.tensor
    softmax_threshold_log2: tl.tensor
    gate_threshold_log2: tl.tensor
    value_scale: tl.tensor
    m_block: tl.tensor
    actual_seqlen_q: tl.tensor
    actual_seqlen_k: tl.tensor
    offset_q: tl.tensor
    offset_k: tl.tensor
    padded_offset_q: tl.tensor
    padded_offset_k: tl.tensor
    window_size_left: tl.tensor
    window_size_right: tl.tensor
    head_dim: tl.tensor
    PACK_GQA: tl.constexpr
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr
    TILE_M: tl.constexpr
    TILE_N: tl.constexpr
    TILE_K: tl.constexpr
    IS_LOGSIGMOID_GATE: tl.constexpr

    @constexpr_function
    def __init__(
        self,
        softmax_scale_log2,
        softmax_threshold_log2,
        gate_threshold_log2,
        value_scale,
        m_block,
        actual_seqlen_q,
        actual_seqlen_k,
        offset_q,
        offset_k,
        padded_offset_q,
        padded_offset_k,
        window_size_left,
        window_size_right,
        head_dim,
        PACK_GQA,
        QHEAD_PER_KVHEAD_PACKGQA,
        TILE_M,
        TILE_N,
        TILE_K,
        IS_LOGSIGMOID_GATE,
    ):
        self.softmax_scale_log2 = softmax_scale_log2
        self.softmax_threshold_log2 = softmax_threshold_log2
        self.gate_threshold_log2 = gate_threshold_log2
        self.value_scale = value_scale
        self.m_block = m_block
        self.actual_seqlen_q = actual_seqlen_q
        self.actual_seqlen_k = actual_seqlen_k
        self.offset_q = offset_q
        self.offset_k = offset_k
        self.padded_offset_q = padded_offset_q
        self.padded_offset_k = padded_offset_k
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.head_dim = head_dim
        self.PACK_GQA = tl.constexpr(PACK_GQA)
        self.QHEAD_PER_KVHEAD_PACKGQA = tl.constexpr(QHEAD_PER_KVHEAD_PACKGQA)
        self.TILE_M = tl.constexpr(TILE_M)
        self.TILE_N = tl.constexpr(TILE_N)
        self.TILE_K = tl.constexpr(TILE_K)
        self.IS_LOGSIGMOID_GATE = tl.constexpr(IS_LOGSIGMOID_GATE)

    @staticmethod
    @triton.jit
    def create(
        softmax_scale=0.0,
        softmax_threshold=0.0,
        gate_threshold=0.0,
        query_scale=None,
        key_scale=None,
        value_scale=None,
        m_block=0,
        batch_idx=0,
        window_size_left=0,
        window_size_right=0,
        head_dim=0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        seqlen_q=0,
        seqlen_k=0,
        PACK_GQA: tl.constexpr = False,
        QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr = 1,
        TILE_M: tl.constexpr = 64,
        TILE_N: tl.constexpr = 64,
        TILE_K: tl.constexpr = 64,
        IS_CAUSAL: tl.constexpr = False,
        IS_LOGSIGMOID_GATE: tl.constexpr = False,
        IS_ADAPT_GATE: tl.constexpr = False,
        HAS_CU_SEQLENS_Q: tl.constexpr = False,
        HAS_CU_SEQLENS_K: tl.constexpr = False,
        HAS_SEQUSED_Q: tl.constexpr = False,
        HAS_SEQUSED_K: tl.constexpr = False,
    ):
        # Get seqlen info for this batch
        (
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            actual_seqlen_q,
            actual_seqlen_k,
        ) = seqlen_info.get_seqlen_info_qk(
            batch_idx=batch_idx,
            seqlen_q_static=seqlen_q,
            seqlen_k_static=seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
            HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
            HAS_SEQUSED_Q=HAS_SEQUSED_Q,
            HAS_SEQUSED_K=HAS_SEQUSED_K,
        )
        # Load query scale
        q_scale = tl.load(query_scale)
        # Load key scale
        k_scale = tl.load(key_scale)
        # Load value scale
        v_scale = tl.load(value_scale)
        LOG2E: tl.constexpr = 1.44269504089
        softmax_scale_log2 = softmax_scale * LOG2E * q_scale * k_scale
        softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
            softmax_threshold,
            m_block,
            actual_seqlen_q,
            actual_seqlen_k,
            IS_CAUSAL,
            TILE_M,
            QHEAD_PER_KVHEAD_PACKGQA,
        )
        gate_threshold_log2 = seqlen_info.get_gate_threshold(
            gate_threshold,
            m_block,
            actual_seqlen_q,
            actual_seqlen_k,
            IS_CAUSAL,
            TILE_M,
            QHEAD_PER_KVHEAD_PACKGQA,
            IS_ADAPT_GATE,
        )
        return AttnFwdConfig(
            softmax_scale_log2,
            softmax_threshold_log2,
            gate_threshold_log2,
            v_scale,
            m_block,
            actual_seqlen_q,
            actual_seqlen_k,
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            window_size_left,
            window_size_right,
            head_dim,
            PACK_GQA,
            QHEAD_PER_KVHEAD_PACKGQA,
            TILE_M,
            TILE_N,
            TILE_K,
            IS_LOGSIGMOID_GATE,
        )

    @triton.jit
    def get_offs_m(self):
        return self.m_block * self.TILE_M + tl.arange(0, self.TILE_M)

    @triton.jit
    def get_offs_k(self):
        return tl.arange(0, self.TILE_K)


@aggregate
class AttnBwdConfig:
    softmax_scale_log2: tl.tensor
    softmax_threshold: tl.tensor
    gate_threshold: tl.tensor
    query_scale: tl.tensor
    key_scale: tl.tensor
    value_scale: tl.tensor
    n_block: tl.tensor
    actual_seqlen_q: tl.tensor
    actual_seqlen_k: tl.tensor
    offset_q: tl.tensor
    offset_k: tl.tensor
    padded_offset_q: tl.tensor
    padded_offset_k: tl.tensor
    window_size_left: tl.tensor
    window_size_right: tl.tensor
    head_dim: tl.tensor
    QHEAD_PER_KVHEAD: tl.constexpr
    TILE_M: tl.constexpr
    TILE_N: tl.constexpr
    TILE_K: tl.constexpr
    IS_CAUSAL: tl.constexpr
    IS_LOGSIGMOID_GATE: tl.constexpr
    IS_ADAPT_GATE: tl.constexpr

    @constexpr_function
    def __init__(
        self,
        softmax_scale_log2,
        softmax_threshold,
        gate_threshold,
        query_scale,
        key_scale,
        value_scale,
        n_block,
        actual_seqlen_q,
        actual_seqlen_k,
        offset_q,
        offset_k,
        padded_offset_q,
        padded_offset_k,
        window_size_left,
        window_size_right,
        head_dim,
        QHEAD_PER_KVHEAD,
        TILE_M,
        TILE_N,
        TILE_K,
        IS_CAUSAL,
        IS_LOGSIGMOID_GATE,
        IS_ADAPT_GATE,
    ):
        self.softmax_scale_log2 = softmax_scale_log2
        self.softmax_threshold = softmax_threshold
        self.gate_threshold = gate_threshold
        self.query_scale = query_scale
        self.key_scale = key_scale
        self.value_scale = value_scale
        self.n_block = n_block
        self.actual_seqlen_q = actual_seqlen_q
        self.actual_seqlen_k = actual_seqlen_k
        self.offset_q = offset_q
        self.offset_k = offset_k
        self.padded_offset_q = padded_offset_q
        self.padded_offset_k = padded_offset_k
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.head_dim = head_dim
        self.QHEAD_PER_KVHEAD = tl.constexpr(QHEAD_PER_KVHEAD)
        self.TILE_M = tl.constexpr(TILE_M)
        self.TILE_N = tl.constexpr(TILE_N)
        self.TILE_K = tl.constexpr(TILE_K)
        self.IS_CAUSAL = tl.constexpr(IS_CAUSAL)
        self.IS_LOGSIGMOID_GATE = tl.constexpr(IS_LOGSIGMOID_GATE)
        self.IS_ADAPT_GATE = tl.constexpr(IS_ADAPT_GATE)

    @staticmethod
    @triton.jit
    def create(
        softmax_scale=0.0,
        softmax_threshold=0.0,
        gate_threshold=0.0,
        query_scale=None,
        key_scale=None,
        value_scale=None,
        n_block=0,
        batch_idx=0,
        window_size_left=0,
        window_size_right=0,
        head_dim=0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        seqlen_q=0,
        seqlen_k=0,
        QHEAD_PER_KVHEAD: tl.constexpr = 1,
        TILE_M: tl.constexpr = 64,
        TILE_N: tl.constexpr = 64,
        TILE_K: tl.constexpr = 64,
        IS_CAUSAL: tl.constexpr = False,
        IS_LOGSIGMOID_GATE: tl.constexpr = False,
        IS_ADAPT_GATE: tl.constexpr = False,
        HAS_CU_SEQLENS_Q: tl.constexpr = False,
        HAS_CU_SEQLENS_K: tl.constexpr = False,
        HAS_SEQUSED_Q: tl.constexpr = False,
        HAS_SEQUSED_K: tl.constexpr = False,
    ):
        # Get seqlen info for this batch
        (
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            actual_seqlen_q,
            actual_seqlen_k,
        ) = seqlen_info.get_seqlen_info_qk(
            batch_idx=batch_idx,
            seqlen_q_static=seqlen_q,
            seqlen_k_static=seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
            HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
            HAS_SEQUSED_Q=HAS_SEQUSED_Q,
            HAS_SEQUSED_K=HAS_SEQUSED_K,
        )
        # Load query scale
        q_scale = tl.load(query_scale)
        # Load key scale
        k_scale = tl.load(key_scale)
        # Load value scale
        v_scale = tl.load(value_scale)
        LOG2E: tl.constexpr = 1.44269504089
        softmax_scale_log2 = softmax_scale * LOG2E
        softmax_threshold = tl.full((), softmax_threshold, tl.float32)
        gate_threshold = tl.full((), gate_threshold, tl.float32)
        return AttnBwdConfig(
            softmax_scale_log2,
            softmax_threshold,
            gate_threshold,
            q_scale,
            k_scale,
            v_scale,
            n_block,
            actual_seqlen_q,
            actual_seqlen_k,
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            window_size_left,
            window_size_right,
            head_dim,
            QHEAD_PER_KVHEAD,
            TILE_M,
            TILE_N,
            TILE_K,
            IS_CAUSAL,
            IS_LOGSIGMOID_GATE,
            IS_ADAPT_GATE,
        )

    @triton.jit
    def get_softmax_threshold_log2(
        self,
        m_block,
    ):
        return seqlen_info.get_softmax_threshold(
            self.softmax_threshold,
            m_block,
            self.actual_seqlen_q,
            self.actual_seqlen_k,
            self.IS_CAUSAL,
            self.TILE_M,
            QHEAD_PER_KVHEAD_PACKGQA=1,
        )

    @triton.jit
    def get_gate_threshold_log2(
        self,
        m_block,
    ):
        return seqlen_info.get_gate_threshold(
            self.gate_threshold,
            m_block,
            self.actual_seqlen_q,
            self.actual_seqlen_k,
            self.IS_CAUSAL,
            self.TILE_M,
            QHEAD_PER_KVHEAD_PACKGQA=1,
            IS_ADAPT_GATE=self.IS_ADAPT_GATE,
        )


@aggregate
class AttnDecConfig:
    softmax_scale_log2: tl.tensor
    softmax_threshold_log2: tl.tensor
    gate_threshold_log2: tl.tensor
    value_scale: tl.tensor
    m_block: tl.tensor
    actual_seqlen_q: tl.tensor
    actual_seqlen_k: tl.tensor
    offset_q: tl.tensor
    offset_k: tl.tensor
    padded_offset_q: tl.tensor
    padded_offset_k: tl.tensor
    window_size_left: tl.tensor
    window_size_right: tl.tensor
    head_dim: tl.tensor
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr
    TILE_M: tl.constexpr
    TILE_N: tl.constexpr
    TILE_K: tl.constexpr
    IS_LOGSIGMOID_GATE: tl.constexpr

    @constexpr_function
    def __init__(
        self,
        softmax_scale_log2,
        softmax_threshold_log2,
        gate_threshold_log2,
        value_scale,
        m_block,
        actual_seqlen_q,
        actual_seqlen_k,
        offset_q,
        offset_k,
        padded_offset_q,
        padded_offset_k,
        window_size_left,
        window_size_right,
        head_dim,
        QHEAD_PER_KVHEAD_PACKGQA,
        TILE_M,
        TILE_N,
        TILE_K,
        IS_LOGSIGMOID_GATE,
    ):
        self.softmax_scale_log2 = softmax_scale_log2
        self.softmax_threshold_log2 = softmax_threshold_log2
        self.gate_threshold_log2 = gate_threshold_log2
        self.value_scale = value_scale
        self.m_block = m_block
        self.actual_seqlen_q = actual_seqlen_q
        self.actual_seqlen_k = actual_seqlen_k
        self.offset_q = offset_q
        self.offset_k = offset_k
        self.padded_offset_q = padded_offset_q
        self.padded_offset_k = padded_offset_k
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.head_dim = head_dim
        self.QHEAD_PER_KVHEAD_PACKGQA = tl.constexpr(QHEAD_PER_KVHEAD_PACKGQA)
        self.TILE_M = tl.constexpr(TILE_M)
        self.TILE_N = tl.constexpr(TILE_N)
        self.TILE_K = tl.constexpr(TILE_K)
        self.IS_LOGSIGMOID_GATE = tl.constexpr(IS_LOGSIGMOID_GATE)

    @staticmethod
    @triton.jit
    def create(
        softmax_scale=0.0,
        softmax_threshold=0.0,
        gate_threshold=0.0,
        query_scale=None,
        key_scale=None,
        value_scale=None,
        batch_idx=0,
        window_size_left=0,
        window_size_right=0,
        head_dim=0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        seqlen_q=0,
        seqlen_k=0,
        QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr = 1,
        TILE_M: tl.constexpr = 16,
        TILE_N: tl.constexpr = 64,
        TILE_K: tl.constexpr = 64,
        IS_LOGSIGMOID_GATE: tl.constexpr = False,
        HAS_CU_SEQLENS_Q: tl.constexpr = False,
        HAS_CU_SEQLENS_K: tl.constexpr = False,
        HAS_SEQUSED_Q: tl.constexpr = False,
        HAS_SEQUSED_K: tl.constexpr = False,
    ):
        # Get seqlen info for this batch
        (
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            actual_seqlen_q,
            actual_seqlen_k,
        ) = seqlen_info.get_seqlen_info_qk(
            batch_idx=batch_idx,
            seqlen_q_static=seqlen_q,
            seqlen_k_static=seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            HAS_CU_SEQLENS_Q=HAS_CU_SEQLENS_Q,
            HAS_CU_SEQLENS_K=HAS_CU_SEQLENS_K,
            HAS_SEQUSED_Q=HAS_SEQUSED_Q,
            HAS_SEQUSED_K=HAS_SEQUSED_K,
        )
        # Load query scale
        q_scale = tl.load(query_scale)
        # Load key scale
        k_scale = tl.load(key_scale)
        # Load value scale
        v_scale = tl.load(value_scale)
        LOG2E: tl.constexpr = 1.44269504089
        softmax_scale_log2 = softmax_scale * LOG2E * q_scale * k_scale
        softmax_threshold_log2 = seqlen_info.get_softmax_threshold(
            softmax_threshold,
            0,
            actual_seqlen_q,
            actual_seqlen_k,
            False,
            TILE_M,
            QHEAD_PER_KVHEAD_PACKGQA,
        )
        gate_threshold_log2 = seqlen_info.get_gate_threshold(
            gate_threshold,
            0,
            actual_seqlen_q,
            actual_seqlen_k,
            False,
            TILE_M,
            QHEAD_PER_KVHEAD_PACKGQA,
            False,
        )
        return AttnDecConfig(
            softmax_scale_log2,
            softmax_threshold_log2,
            gate_threshold_log2,
            v_scale,
            tl.full((), 0, tl.int32),
            actual_seqlen_q,
            actual_seqlen_k,
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            window_size_left,
            window_size_right,
            head_dim,
            QHEAD_PER_KVHEAD_PACKGQA,
            TILE_M,
            TILE_N,
            TILE_K,
            IS_LOGSIGMOID_GATE,
        )


@aggregate
class AttnFwdBlockScheduler:
    n_block_min: tl.tensor
    n_block_max: tl.tensor
    n_block_max_no_mask: tl.tensor
    n_block_window_min: tl.tensor
    n_block_window_max: tl.tensor
    n_block_window_min_no_mask: tl.tensor
    n_block_window_max_no_mask: tl.tensor

    @constexpr_function
    def __init__(
        self,
        n_block_min,
        n_block_max,
        n_block_max_no_mask,
        n_block_window_min,
        n_block_window_max,
        n_block_window_min_no_mask,
        n_block_window_max_no_mask,
    ):
        self.n_block_min = n_block_min
        self.n_block_max = n_block_max
        self.n_block_max_no_mask = n_block_max_no_mask
        self.n_block_window_min = n_block_window_min
        self.n_block_window_max = n_block_window_max
        self.n_block_window_min_no_mask = n_block_window_min_no_mask
        self.n_block_window_max_no_mask = n_block_window_max_no_mask

    @triton.jit
    def is_empty(self):
        return self.n_block_max <= self.n_block_min

    @staticmethod
    @triton.jit
    def create(
        config: AttnFwdConfig,
        split_idx,
        num_splits,
        IS_CAUSAL: tl.constexpr,
        IS_LOCAL: tl.constexpr,
        IS_SPLIT_KV: tl.constexpr,
    ):
        # Compute causal n_block range for this m_block
        n_block_min, n_block_max, n_block_window_min, n_block_window_max = (
            block_info.get_n_block_min_max(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                m_block=config.m_block,
                split_idx=split_idx,
                num_splits=num_splits,
                window_size_left=config.window_size_left,
                window_size_right=config.window_size_right,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_CAUSAL=IS_CAUSAL,
                IS_LOCAL=IS_LOCAL,
                IS_SPLIT_KV=IS_SPLIT_KV,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        )
        n_block_max_no_mask = block_info.get_n_block_min_causal_local_mask(
            seqlen_q=config.actual_seqlen_q,
            seqlen_k=config.actual_seqlen_k,
            m_block=config.m_block,
            n_block_min=n_block_min,
            window_size_right=0,
            TILE_N=config.TILE_N,
            TILE_M=config.TILE_M,
            IS_LOCAL=False,
            QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
        )

        # Clamp to split's range so the no-mask loop stays within bounds
        if IS_SPLIT_KV:
            n_block_max_no_mask = tl.minimum(n_block_max_no_mask, n_block_max)

        if IS_LOCAL:
            # Compute local n_block range for this m_block
            n_block_window_min = tl.maximum(n_block_window_min, n_block_min)
            n_block_window_max = tl.minimum(n_block_window_max, n_block_max_no_mask)
            n_block_window_max_no_mask = block_info.get_n_block_min_causal_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                m_block=config.m_block,
                n_block_min=n_block_window_min,
                window_size_right=config.window_size_right,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_LOCAL=True,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
            n_block_window_min_no_mask = block_info.get_n_block_min_before_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                m_block=config.m_block,
                n_block_min=n_block_window_min,
                window_size_left=config.window_size_left,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_LOCAL=True,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
            n_block_window_min_no_mask = tl.minimum(
                n_block_window_min_no_mask, n_block_window_max_no_mask
            )

            # Clamp window no-mask boundaries to the window's range
            if IS_SPLIT_KV:
                n_block_window_max_no_mask = tl.maximum(
                    tl.minimum(n_block_window_max_no_mask, n_block_window_max),
                    n_block_window_min,
                )
                n_block_window_min_no_mask = tl.maximum(
                    tl.minimum(n_block_window_min_no_mask, n_block_window_max),
                    n_block_window_min,
                )
        else:
            n_block_window_min_no_mask = 0
            n_block_window_max_no_mask = 0

        return AttnFwdBlockScheduler(
            n_block_min,
            n_block_max,
            n_block_max_no_mask,
            n_block_window_min,
            n_block_window_max,
            n_block_window_min_no_mask,
            n_block_window_max_no_mask,
        )


@aggregate
class AttnBwdBlockScheduler:
    m_block_min: tl.tensor
    m_block_max: tl.tensor
    m_block_min_no_mask: tl.tensor
    m_block_window_min: tl.tensor
    m_block_window_max: tl.tensor
    m_block_window_min_no_mask: tl.tensor
    m_block_window_max_no_mask: tl.tensor

    @constexpr_function
    def __init__(
        self,
        m_block_min,
        m_block_max,
        m_block_min_no_mask,
        m_block_window_min,
        m_block_window_max,
        m_block_window_min_no_mask,
        m_block_window_max_no_mask,
    ):
        self.m_block_min = m_block_min
        self.m_block_max = m_block_max
        self.m_block_min_no_mask = m_block_min_no_mask
        self.m_block_window_min = m_block_window_min
        self.m_block_window_max = m_block_window_max
        self.m_block_window_min_no_mask = m_block_window_min_no_mask
        self.m_block_window_max_no_mask = m_block_window_max_no_mask

    @triton.jit
    def is_empty(self):
        return self.m_block_max <= self.m_block_min

    @staticmethod
    @triton.jit
    def create(
        config: AttnBwdConfig,
        split_idx,
        num_splits,
        IS_CAUSAL: tl.constexpr,
        IS_LOCAL: tl.constexpr,
        IS_SPLIT_QO: tl.constexpr,
    ):
        # Compute causal m_block range for this n_block
        m_block_min, m_block_max, m_block_window_min, m_block_window_max = (
            block_info.get_m_block_min_max(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                n_block=config.n_block,
                split_idx=split_idx,
                num_splits=num_splits,
                window_size_left=config.window_size_left,
                window_size_right=config.window_size_right,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_CAUSAL=IS_CAUSAL,
                IS_LOCAL=IS_LOCAL,
                IS_SPLIT_QO=IS_SPLIT_QO,
            )
        )
        m_block_min_no_mask = block_info.get_m_block_min_causal_local_mask(
            seqlen_q=config.actual_seqlen_q,
            seqlen_k=config.actual_seqlen_k,
            n_block=config.n_block,
            m_block_min=m_block_min,
            window_size_right=0,
            TILE_N=config.TILE_N,
            TILE_M=config.TILE_M,
            IS_CAUSAL=IS_CAUSAL or IS_LOCAL,
            IS_LOCAL=False,
        )

        # Clamp to split's range so the no-mask loop stays within bounds
        if IS_SPLIT_QO:
            m_block_min_no_mask = tl.minimum(m_block_min_no_mask, m_block_max)

        if IS_LOCAL:
            # Compute local m_block range for this n_block
            m_block_window_min = tl.maximum(m_block_window_min, m_block_min_no_mask)
            m_block_window_max = tl.minimum(m_block_window_max, m_block_max)
            m_block_window_min_no_mask = block_info.get_m_block_min_causal_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                n_block=config.n_block,
                m_block_min=m_block_window_min,
                window_size_right=config.window_size_right,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_CAUSAL=False,
                IS_LOCAL=True,
            )
            m_block_window_min_no_mask = tl.maximum(
                m_block_window_min_no_mask, m_block_window_min
            )
            m_block_window_max_no_mask = block_info.get_m_block_max_before_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                n_block=config.n_block,
                m_block_max=m_block_window_max,
                window_size_left=config.window_size_left,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_LOCAL=True,
            )
            m_block_window_max_no_mask = tl.maximum(
                m_block_window_max_no_mask, m_block_window_min_no_mask
            )
        else:
            m_block_window_min_no_mask = 0
            m_block_window_max_no_mask = 0

        return AttnBwdBlockScheduler(
            m_block_min,
            m_block_max,
            m_block_min_no_mask,
            m_block_window_min,
            m_block_window_max,
            m_block_window_min_no_mask,
            m_block_window_max_no_mask,
        )


@aggregate
class AttnDecBlockScheduler:
    n_block_min: tl.tensor
    n_block_max: tl.tensor
    n_block_max_no_mask: tl.tensor
    n_block_window_min: tl.tensor
    n_block_window_max: tl.tensor
    n_block_window_min_no_mask: tl.tensor
    n_block_window_max_no_mask: tl.tensor

    @constexpr_function
    def __init__(
        self,
        n_block_min,
        n_block_max,
        n_block_max_no_mask,
        n_block_window_min,
        n_block_window_max,
        n_block_window_min_no_mask,
        n_block_window_max_no_mask,
    ):
        self.n_block_min = n_block_min
        self.n_block_max = n_block_max
        self.n_block_max_no_mask = n_block_max_no_mask
        self.n_block_window_min = n_block_window_min
        self.n_block_window_max = n_block_window_max
        self.n_block_window_min_no_mask = n_block_window_min_no_mask
        self.n_block_window_max_no_mask = n_block_window_max_no_mask

    @triton.jit
    def is_empty(self):
        return self.n_block_max <= self.n_block_min

    @staticmethod
    @triton.jit
    def create(
        config: AttnDecConfig,
        split_idx,
        num_splits,
        IS_LOCAL: tl.constexpr,
    ):
        # Compute non-causal n_block range for this m_block
        n_block_min, n_block_max, n_block_window_min, n_block_window_max = (
            block_info.get_n_block_min_max(
                seqlen_q=1,
                seqlen_k=config.actual_seqlen_k,
                m_block=0,
                split_idx=split_idx,
                num_splits=num_splits,
                window_size_left=config.window_size_left,
                window_size_right=config.window_size_right,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_CAUSAL=False,
                IS_LOCAL=IS_LOCAL,
                IS_SPLIT_KV=True,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        )
        n_block_max_no_mask = block_info.get_n_block_min_causal_local_mask(
            seqlen_q=1,
            seqlen_k=config.actual_seqlen_k,
            m_block=0,
            n_block_min=n_block_min,
            window_size_right=0,
            TILE_N=config.TILE_N,
            TILE_M=config.TILE_M,
            IS_LOCAL=False,
            QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
        )

        # Clamp to split's range so the no-mask loop stays within bounds
        n_block_max_no_mask = tl.minimum(n_block_max_no_mask, n_block_max)

        if IS_LOCAL:
            # Compute local n_block range for this m_block
            n_block_window_min = tl.maximum(n_block_window_min, n_block_min)
            n_block_window_max = tl.minimum(n_block_window_max, n_block_max_no_mask)
            n_block_window_max_no_mask = block_info.get_n_block_min_causal_local_mask(
                seqlen_q=1,
                seqlen_k=config.actual_seqlen_k,
                m_block=0,
                n_block_min=n_block_window_min,
                window_size_right=config.window_size_right,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_LOCAL=True,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
            n_block_window_min_no_mask = block_info.get_n_block_min_before_local_mask(
                seqlen_q=1,
                seqlen_k=config.actual_seqlen_k,
                m_block=0,
                n_block_min=n_block_window_min,
                window_size_left=config.window_size_left,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_LOCAL=True,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
            n_block_window_min_no_mask = tl.minimum(
                n_block_window_min_no_mask, n_block_window_max_no_mask
            )

            # Clamp window no-mask boundaries to the window's range
            n_block_window_max_no_mask = tl.maximum(
                tl.minimum(n_block_window_max_no_mask, n_block_window_max),
                n_block_window_min,
            )
            n_block_window_min_no_mask = tl.maximum(
                tl.minimum(n_block_window_min_no_mask, n_block_window_max),
                n_block_window_min,
            )
        else:
            n_block_window_min_no_mask = 0
            n_block_window_max_no_mask = 0

        return AttnDecBlockScheduler(
            n_block_min,
            n_block_max,
            n_block_max_no_mask,
            n_block_window_min,
            n_block_window_max,
            n_block_window_min_no_mask,
            n_block_window_max_no_mask,
        )


@aggregate
class AttnFwdPointerScheduler:
    q_base: tl.tensor
    k_base: tl.tensor
    v_base: tl.tensor
    a_base: tl.tensor
    d_base: tl.tensor
    out_base: tl.tensor
    lse_base: tl.tensor
    head_idx: tl.tensor
    stride_qh: tl.tensor
    stride_qm: tl.tensor
    stride_kn: tl.tensor
    stride_vn: tl.tensor
    stride_ah: tl.tensor
    stride_oh: tl.tensor
    stride_om: tl.tensor
    stride_lh: tl.tensor

    @constexpr_function
    def __init__(
        self,
        q_base,
        k_base,
        v_base,
        a_base,
        d_base,
        out_base,
        lse_base,
        head_idx,
        stride_qh,
        stride_qm,
        stride_kn,
        stride_vn,
        stride_ah,
        stride_oh,
        stride_om,
        stride_lh,
    ):
        self.q_base = q_base
        self.k_base = k_base
        self.v_base = v_base
        self.a_base = a_base
        self.d_base = d_base
        self.out_base = out_base
        self.lse_base = lse_base
        self.head_idx = head_idx
        self.stride_qh = stride_qh
        self.stride_qm = stride_qm
        self.stride_kn = stride_kn
        self.stride_vn = stride_vn
        self.stride_ah = stride_ah
        self.stride_oh = stride_oh
        self.stride_om = stride_om
        self.stride_lh = stride_lh

    @staticmethod
    @triton.jit
    def create(
        config: AttnFwdConfig,
        Q=None,
        K=None,
        V=None,
        A=None,
        D=None,
        Out=None,
        Lse=None,
        batch_idx=0,
        head_idx=0,
        head_kv_idx=0,
        split_idx=0,
        stride_qb=0,
        stride_qh=0,
        stride_qm=0,
        stride_kb=0,
        stride_kh=0,
        stride_kn=0,
        stride_vb=0,
        stride_vh=0,
        stride_vn=0,
        stride_ab=0,
        stride_ah=0,
        stride_am=0,
        stride_db=0,
        stride_dh=0,
        stride_dn=0,
        stride_ob=0,
        stride_oh=0,
        stride_om=0,
        stride_os=0,
        stride_lb=0,
        stride_lh=0,
        stride_ls=0,
        IS_SPLIT_KV: tl.constexpr = False,
        IS_GATED: tl.constexpr = False,
        HAS_CU_SEQLENS_Q: tl.constexpr = False,
        HAS_CU_SEQLENS_K: tl.constexpr = False,
    ):
        # Initialize base pointers
        q_base = seqlen_info.offset_batch_Q(
            Q + head_idx * stride_qh if not config.PACK_GQA else Q,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_qb,
            stride_qm,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        k_base = seqlen_info.offset_batch_K(
            K + head_kv_idx * stride_kh,
            batch_idx,
            config.offset_k,
            config.padded_offset_k,
            stride_kb,
            stride_kn,
            HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        v_base = seqlen_info.offset_batch_K(
            V + head_kv_idx * stride_vh,
            batch_idx,
            config.offset_k,
            config.padded_offset_k,
            stride_vb,
            stride_vn,
            HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        out_base = seqlen_info.offset_batch_Q(
            Out + head_idx * stride_oh if not config.PACK_GQA else Out,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_ob,
            stride_om,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        lse_base = seqlen_info.offset_batch_Q(
            Lse + head_idx * stride_lh if not config.PACK_GQA else Lse,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_lb,
            1,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )

        if IS_GATED:
            a_base = seqlen_info.offset_batch_Q(
                A + head_idx * stride_ah if not config.PACK_GQA else A,
                batch_idx,
                config.offset_q,
                config.padded_offset_q,
                stride_ab,
                stride_am,
                HAS_CU_SEQLENS_Q,
                USE_PADDED=False,
            )
            d_base = seqlen_info.offset_batch_K(
                D + head_kv_idx * stride_dh,
                batch_idx,
                config.offset_k,
                config.padded_offset_k,
                stride_db,
                stride_dn,
                HAS_CU_SEQLENS_K,
                USE_PADDED=False,
            )
        else:
            a_base = tl.full((), 0, tl.int64)
            d_base = tl.full((), 0, tl.int64)
            stride_ah = tl.full((), 0, tl.int64)

        # For split KV, offset output and LSE base pointers by split_idx
        if IS_SPLIT_KV:
            out_base += split_idx * stride_os
            lse_base += split_idx * stride_ls

        return AttnFwdPointerScheduler(
            q_base,
            k_base,
            v_base,
            a_base,
            d_base,
            out_base,
            lse_base,
            head_idx,
            stride_qh,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_ah,
            stride_oh,
            stride_om,
            stride_lh,
        )

    @triton.jit
    def make_q_ptrs(self, config: AttnFwdConfig):
        if config.PACK_GQA:
            return seqlen_info.make_pack_gqa_ptrs(
                self.q_base,
                config.m_block,
                self.head_idx,
                self.stride_qh,
                self.stride_qm,
                TILE_M=config.TILE_M,
                TILE_K=config.TILE_K,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        else:
            return tl.make_tensor_descriptor(
                self.q_base,
                shape=[config.actual_seqlen_q, config.head_dim],
                strides=[self.stride_qm, 1],
                block_shape=[config.TILE_M, config.TILE_K],
            )

    @triton.jit
    def make_k_ptrs(self, config: AttnFwdConfig):
        return tl.make_tensor_descriptor(
            self.k_base,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_kn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_v_ptrs(self, config: AttnFwdConfig):
        return tl.make_tensor_descriptor(
            self.v_base,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_vn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_a_ptrs(self, config: AttnFwdConfig):
        if config.PACK_GQA:
            return seqlen_info.make_pack_gqa_ptrs(
                self.a_base,
                config.m_block,
                self.head_idx,
                self.stride_ah,
                1,
                TILE_M=config.TILE_M,
                TILE_K=1,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        else:
            return tl.make_tensor_descriptor(
                self.a_base,
                shape=[config.actual_seqlen_q],
                strides=[1],
                block_shape=[config.TILE_M],
            )

    @triton.jit
    def make_d_ptrs(self, config: AttnFwdConfig):
        return tl.make_tensor_descriptor(
            self.d_base,
            shape=[config.actual_seqlen_k],
            strides=[1],
            block_shape=[config.TILE_N],
        )

    @triton.jit
    def make_out_ptrs(self, config: AttnFwdConfig):
        if config.PACK_GQA:
            return seqlen_info.make_pack_gqa_ptrs(
                self.out_base,
                config.m_block,
                self.head_idx,
                self.stride_oh,
                self.stride_om,
                TILE_M=config.TILE_M,
                TILE_K=config.TILE_K,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        else:
            return tl.make_tensor_descriptor(
                self.out_base,
                shape=[config.actual_seqlen_q, config.head_dim],
                strides=[self.stride_om, 1],
                block_shape=[config.TILE_M, config.TILE_K],
            )

    @triton.jit
    def make_lse_ptrs(self, config: AttnFwdConfig):
        if config.PACK_GQA:
            return seqlen_info.make_pack_gqa_ptrs(
                self.lse_base,
                config.m_block,
                self.head_idx,
                self.stride_lh,
                1,
                TILE_M=config.TILE_M,
                TILE_K=1,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        else:
            return tl.make_tensor_descriptor(
                self.lse_base,
                shape=[config.actual_seqlen_q],
                strides=[1],
                block_shape=[config.TILE_M],
            )

    @triton.jit
    def load_q(self, config: AttnFwdConfig, q_ptrs):
        if config.PACK_GQA:
            offs_m = config.get_offs_m()
            offs_k = config.get_offs_k()
            return tl.load(
                q_ptrs,
                mask=(
                    (offs_m // config.QHEAD_PER_KVHEAD_PACKGQA) < config.actual_seqlen_q
                )[:, None]
                & (offs_k < config.head_dim)[None, :],
                other=0.0,
                cache_modifier=".ca",
            )
        else:
            return q_ptrs.load([config.m_block * config.TILE_M, 0])

    @triton.jit
    def load_k(self, config: AttnFwdConfig, k_ptrs, n_block):
        return k_ptrs.load([n_block * config.TILE_N, 0])

    @triton.jit
    def load_v(self, config: AttnFwdConfig, v_ptrs, n_block):
        return v_ptrs.load([n_block * config.TILE_N, 0])

    @triton.jit
    def load_a(self, config: AttnFwdConfig, a_ptrs):
        if config.PACK_GQA:
            offs_m = config.get_offs_m()
            return tl.load(
                a_ptrs,
                mask=(offs_m // config.QHEAD_PER_KVHEAD_PACKGQA)
                < config.actual_seqlen_q,
                other=0.0,
            ).to(tl.float32)
        else:
            return a_ptrs.load([config.m_block * config.TILE_M]).to(tl.float32)

    @triton.jit
    def load_d(self, config: AttnFwdConfig, d_ptrs, n_block):
        return d_ptrs.load([n_block * config.TILE_N]).to(tl.float32)

    @triton.jit
    def store_out(
        self, config: AttnFwdConfig, out_ptrs, o_tile, IS_SPLIT_KV: tl.constexpr = False
    ):
        if not IS_SPLIT_KV:
            if config.PACK_GQA:
                o_tile = o_tile.to(out_ptrs.dtype.element_ty)
            else:
                o_tile = o_tile.to(out_ptrs.dtype)
        if config.PACK_GQA:
            offs_m = config.get_offs_m()
            offs_k = config.get_offs_k()
            tl.store(
                out_ptrs,
                o_tile,
                mask=(
                    (offs_m // config.QHEAD_PER_KVHEAD_PACKGQA) < config.actual_seqlen_q
                )[:, None]
                & (offs_k < config.head_dim)[None, :],
                cache_modifier=".wb",
            )
        else:
            out_ptrs.store([config.m_block * config.TILE_M, 0], o_tile)

    @triton.jit
    def store_lse(self, config: AttnFwdConfig, lse_ptrs, lse_tile):
        if config.PACK_GQA:
            offs_m = config.get_offs_m()
            tl.store(
                lse_ptrs,
                lse_tile,
                mask=(
                    (offs_m // config.QHEAD_PER_KVHEAD_PACKGQA) < config.actual_seqlen_q
                ),
                cache_modifier=".wb",
            )
        else:
            lse_ptrs.store([config.m_block * config.TILE_M], lse_tile)

    @triton.jit
    def store_empty(self, config: AttnFwdConfig, out_ptrs, lse_ptrs):
        lse_tile = tl.full((config.TILE_M,), float("-inf"), dtype=tl.float32)
        self.store_lse(config, lse_ptrs, lse_tile)
        o_tile = tl.zeros((config.TILE_M, config.TILE_K), dtype=tl.float32)
        self.store_out(config, out_ptrs, o_tile)


@aggregate
class AttnBwdPointerScheduler:
    q_base: tl.tensor
    k_base: tl.tensor
    v_base: tl.tensor
    a_base: tl.tensor
    d_base: tl.tensor
    do_base: tl.tensor
    lse_base: tl.tensor
    dpsum_base: tl.tensor
    dq_accum_base: tl.tensor
    dk_base: tl.tensor
    dv_base: tl.tensor
    da_base: tl.tensor
    dd_base: tl.tensor
    stride_qm: tl.tensor
    stride_kn: tl.tensor
    stride_vn: tl.tensor
    stride_dom: tl.tensor
    stride_dqam: tl.tensor
    stride_dkn: tl.tensor
    stride_dvn: tl.tensor

    @constexpr_function
    def __init__(
        self,
        q_base,
        k_base,
        v_base,
        a_base,
        d_base,
        do_base,
        lse_base,
        dpsum_base,
        dq_accum_base,
        dk_base,
        dv_base,
        da_base,
        dd_base,
        stride_qm,
        stride_kn,
        stride_vn,
        stride_dom,
        stride_dqam,
        stride_dkn,
        stride_dvn,
    ):
        self.q_base = q_base
        self.k_base = k_base
        self.v_base = v_base
        self.a_base = a_base
        self.d_base = d_base
        self.do_base = do_base
        self.lse_base = lse_base
        self.dpsum_base = dpsum_base
        self.dq_accum_base = dq_accum_base
        self.dk_base = dk_base
        self.dv_base = dv_base
        self.da_base = da_base
        self.dd_base = dd_base
        self.stride_qm = stride_qm
        self.stride_kn = stride_kn
        self.stride_vn = stride_vn
        self.stride_dom = stride_dom
        self.stride_dqam = stride_dqam
        self.stride_dkn = stride_dkn
        self.stride_dvn = stride_dvn

    @staticmethod
    @triton.jit
    def create(
        config: AttnBwdConfig,
        Q=None,
        K=None,
        V=None,
        A=None,
        D=None,
        dO=None,
        LSELog2=None,
        dPsum=None,
        dQaccum=None,
        dK=None,
        dV=None,
        dA=None,
        dD=None,
        batch_idx=0,
        head_idx=0,
        head_kv_idx=0,
        split_idx=0,
        stride_qb=0,
        stride_qh=0,
        stride_qm=0,
        stride_kb=0,
        stride_kh=0,
        stride_kn=0,
        stride_vb=0,
        stride_vh=0,
        stride_vn=0,
        stride_ab=0,
        stride_ah=0,
        stride_db=0,
        stride_dh=0,
        stride_dob=0,
        stride_doh=0,
        stride_dom=0,
        stride_lb=0,
        stride_lh=0,
        stride_pb=0,
        stride_ph=0,
        stride_dqab=0,
        stride_dqah=0,
        stride_dqam=0,
        stride_dkb=0,
        stride_dkh=0,
        stride_dkn=0,
        stride_dks=0,
        stride_dvb=0,
        stride_dvh=0,
        stride_dvn=0,
        stride_dvs=0,
        stride_dab=0,
        stride_dah=0,
        stride_ddb=0,
        stride_ddh=0,
        stride_dds=0,
        IS_SPLIT_QO: tl.constexpr = False,
        IS_GATED: tl.constexpr = False,
        HAS_CU_SEQLENS_Q: tl.constexpr = False,
        HAS_CU_SEQLENS_K: tl.constexpr = False,
    ):
        # Initialize base pointers
        q_base = seqlen_info.offset_batch_Q(
            Q + head_idx * stride_qh,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_qb,
            stride_qm,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        k_base = seqlen_info.offset_batch_K(
            K + head_kv_idx * stride_kh,
            batch_idx,
            config.offset_k,
            config.padded_offset_k,
            stride_kb,
            stride_kn,
            HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        v_base = seqlen_info.offset_batch_K(
            V + head_kv_idx * stride_vh,
            batch_idx,
            config.offset_k,
            config.padded_offset_k,
            stride_vb,
            stride_vn,
            HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        do_base = seqlen_info.offset_batch_Q(
            dO + head_idx * stride_doh,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_dob,
            stride_dom,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        lse_base = seqlen_info.offset_batch_Q(
            LSELog2 + head_idx * stride_lh,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_lb,
            1,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=True,
        )
        dpsum_base = seqlen_info.offset_batch_Q(
            dPsum + head_idx * stride_ph,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_pb,
            1,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=True,
        )
        dq_accum_base = seqlen_info.offset_batch_Q(
            dQaccum + head_idx * stride_dqah,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_dqab,
            stride_dqam,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=True,
        )
        dk_base = seqlen_info.offset_batch_K(
            dK + head_kv_idx * stride_dkh,
            batch_idx,
            config.offset_k,
            config.padded_offset_k,
            stride_dkb,
            stride_dkn,
            HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        dv_base = seqlen_info.offset_batch_K(
            dV + head_kv_idx * stride_dvh,
            batch_idx,
            config.offset_k,
            config.padded_offset_k,
            stride_dvb,
            stride_dvn,
            HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )

        if IS_GATED:
            a_base = seqlen_info.offset_batch_Q(
                A + head_idx * stride_ah,
                batch_idx,
                config.offset_q,
                config.padded_offset_q,
                stride_ab,
                1,
                HAS_CU_SEQLENS_Q,
                USE_PADDED=False,
            )
            d_base = seqlen_info.offset_batch_K(
                D + head_kv_idx * stride_dh,
                batch_idx,
                config.offset_k,
                config.padded_offset_k,
                stride_db,
                1,
                HAS_CU_SEQLENS_K,
                USE_PADDED=False,
            )
            da_base = seqlen_info.offset_batch_Q(
                dA + head_idx * stride_dah,
                batch_idx,
                config.offset_q,
                config.padded_offset_q,
                stride_dab,
                1,
                HAS_CU_SEQLENS_Q,
                USE_PADDED=True,
            )
            dd_base = seqlen_info.offset_batch_K(
                dD + head_kv_idx * stride_ddh,
                batch_idx,
                config.offset_k,
                config.padded_offset_k,
                stride_ddb,
                1,
                HAS_CU_SEQLENS_K,
                USE_PADDED=False,
            )
        else:
            a_base = tl.full((), 0, tl.int64)
            d_base = tl.full((), 0, tl.int64)
            da_base = tl.full((), 0, tl.int64)
            dd_base = tl.full((), 0, tl.int64)

        # For split QO, offset key and value gradients base pointers by split_idx
        if IS_SPLIT_QO:
            dk_base += split_idx * stride_dks
            dv_base += split_idx * stride_dvs
            if IS_GATED:
                dd_base += split_idx * stride_dds

        return AttnBwdPointerScheduler(
            q_base,
            k_base,
            v_base,
            a_base,
            d_base,
            do_base,
            lse_base,
            dpsum_base,
            dq_accum_base,
            dk_base,
            dv_base,
            da_base,
            dd_base,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_dom,
            stride_dqam,
            stride_dkn,
            stride_dvn,
        )

    @triton.jit
    def make_q_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.q_base,
            shape=[config.actual_seqlen_q, config.head_dim],
            strides=[self.stride_qm, 1],
            block_shape=[config.TILE_M, config.TILE_K],
        )

    @triton.jit
    def make_k_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.k_base,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_kn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_v_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.v_base,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_vn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_a_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.a_base,
            shape=[config.actual_seqlen_q],
            strides=[1],
            block_shape=[config.TILE_M],
        )

    @triton.jit
    def make_d_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.d_base,
            shape=[config.actual_seqlen_k],
            strides=[1],
            block_shape=[config.TILE_N],
        )

    @triton.jit
    def make_do_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.do_base,
            shape=[config.actual_seqlen_q, config.head_dim],
            strides=[self.stride_dom, 1],
            block_shape=[config.TILE_M, config.TILE_K],
        )

    @triton.jit
    def make_lse_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.lse_base,
            shape=[config.actual_seqlen_q],
            strides=[1],
            block_shape=[config.TILE_M],
        )

    @triton.jit
    def make_dpsum_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.dpsum_base,
            shape=[config.actual_seqlen_q],
            strides=[1],
            block_shape=[config.TILE_M],
        )

    @triton.jit
    def make_dq_accum_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.dq_accum_base,
            shape=[config.actual_seqlen_q, config.head_dim],
            strides=[self.stride_dqam, 1],
            block_shape=[config.TILE_M, config.TILE_K],
        )

    @triton.jit
    def make_dk_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.dk_base,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_dkn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_dv_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.dv_base,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_dvn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_da_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.da_base,
            shape=[config.actual_seqlen_q],
            strides=[1],
            block_shape=[config.TILE_M],
        )

    @triton.jit
    def make_dd_ptrs(self, config: AttnBwdConfig):
        return tl.make_tensor_descriptor(
            self.dd_base,
            shape=[config.actual_seqlen_k],
            strides=[1],
            block_shape=[config.TILE_N],
        )

    @triton.jit
    def load_q(self, config: AttnBwdConfig, q_ptrs, m_block):
        return (q_ptrs.load([m_block * config.TILE_M, 0]) * config.query_scale).to(
            self.do_base.dtype.element_ty
        )

    @triton.jit
    def load_k(self, config: AttnBwdConfig, k_ptrs):
        return (k_ptrs.load([config.n_block * config.TILE_N, 0]) * config.key_scale).to(
            self.do_base.dtype.element_ty
        )

    @triton.jit
    def load_v(self, config: AttnBwdConfig, v_ptrs):
        return (
            v_ptrs.load([config.n_block * config.TILE_N, 0]) * config.value_scale
        ).to(self.do_base.dtype.element_ty)

    @triton.jit
    def load_a(self, config: AttnBwdConfig, a_ptrs, m_block):
        return a_ptrs.load([m_block * config.TILE_M]).to(tl.float32)

    @triton.jit
    def load_d(self, config: AttnBwdConfig, d_ptrs):
        return d_ptrs.load([config.n_block * config.TILE_N]).to(tl.float32)

    @triton.jit
    def load_do(self, config: AttnBwdConfig, do_ptrs, m_block):
        return do_ptrs.load([m_block * config.TILE_M, 0])

    @triton.jit
    def load_lse(self, config: AttnBwdConfig, lse_ptrs, m_block):
        return lse_ptrs.load([m_block * config.TILE_M])

    @triton.jit
    def load_dpsum(self, config: AttnBwdConfig, dpsum_ptrs, m_block):
        return dpsum_ptrs.load([m_block * config.TILE_M])

    @triton.jit
    def store_dq(self, config: AttnBwdConfig, dq_ptrs, m_block, dq_tile):
        dq_ptrs.atomic_add([m_block * config.TILE_M, 0], dq_tile)

    @triton.jit
    def store_dk(self, config: AttnBwdConfig, dk_ptrs, dk_tile):
        if config.QHEAD_PER_KVHEAD > 1:
            dk_ptrs.atomic_add([config.n_block * config.TILE_N, 0], dk_tile)
        else:
            dk_ptrs.store([config.n_block * config.TILE_N, 0], dk_tile)

    @triton.jit
    def store_dv(self, config: AttnBwdConfig, dv_ptrs, dv_tile):
        if config.QHEAD_PER_KVHEAD > 1:
            dv_ptrs.atomic_add([config.n_block * config.TILE_N, 0], dv_tile)
        else:
            dv_ptrs.store([config.n_block * config.TILE_N, 0], dv_tile)

    @triton.jit
    def store_da(self, config: AttnBwdConfig, da_ptrs, m_block, da_tile):
        da_ptrs.atomic_add([m_block * config.TILE_M], da_tile)

    @triton.jit
    def store_dd(self, config: AttnBwdConfig, dd_ptrs, dd_tile):
        if config.QHEAD_PER_KVHEAD > 1:
            dd_ptrs.atomic_add([config.n_block * config.TILE_N], dd_tile)
        else:
            dd_ptrs.store([config.n_block * config.TILE_N], dd_tile)


@aggregate
class AttnDecPointerScheduler:
    q_base: tl.tensor
    k_base: tl.tensor
    v_base: tl.tensor
    a_base: tl.tensor
    d_base: tl.tensor
    out_base: tl.tensor
    lse_base: tl.tensor
    stride_qh: tl.tensor
    stride_kn: tl.tensor
    stride_vn: tl.tensor
    stride_oh: tl.tensor

    @constexpr_function
    def __init__(
        self,
        q_base,
        k_base,
        v_base,
        a_base,
        d_base,
        out_base,
        lse_base,
        stride_qh,
        stride_kn,
        stride_vn,
        stride_oh,
    ):
        self.q_base = q_base
        self.k_base = k_base
        self.v_base = v_base
        self.a_base = a_base
        self.d_base = d_base
        self.out_base = out_base
        self.lse_base = lse_base
        self.stride_qh = stride_qh
        self.stride_kn = stride_kn
        self.stride_vn = stride_vn
        self.stride_oh = stride_oh

    @staticmethod
    @triton.jit
    def create(
        config: AttnDecConfig,
        Q=None,
        K=None,
        V=None,
        A=None,
        D=None,
        Out=None,
        Lse=None,
        batch_idx=0,
        head_idx=0,
        head_kv_idx=0,
        split_idx=0,
        stride_qb=0,
        stride_qh=0,
        stride_qm=0,
        stride_kb=0,
        stride_kh=0,
        stride_kn=0,
        stride_vb=0,
        stride_vh=0,
        stride_vn=0,
        stride_ab=0,
        stride_ah=0,
        stride_db=0,
        stride_dh=0,
        stride_ob=0,
        stride_oh=0,
        stride_om=0,
        stride_os=0,
        stride_lb=0,
        stride_lh=0,
        stride_lm=0,
        stride_ls=0,
        IS_GATED: tl.constexpr = False,
        HAS_CU_SEQLENS_Q: tl.constexpr = False,
        HAS_CU_SEQLENS_K: tl.constexpr = False,
    ):
        # Initialize base pointers
        q_base = seqlen_info.offset_batch_Q(
            Q + head_idx * config.QHEAD_PER_KVHEAD_PACKGQA * stride_qh,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_qb,
            stride_qm,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        k_base = seqlen_info.offset_batch_K(
            K + head_kv_idx * stride_kh,
            batch_idx,
            config.offset_k,
            config.padded_offset_k,
            stride_kb,
            stride_kn,
            HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        v_base = seqlen_info.offset_batch_K(
            V + head_kv_idx * stride_vh,
            batch_idx,
            config.offset_k,
            config.padded_offset_k,
            stride_vb,
            stride_vn,
            HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        out_base = seqlen_info.offset_batch_Q(
            Out + head_idx * config.QHEAD_PER_KVHEAD_PACKGQA * stride_oh,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_ob,
            stride_om,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        lse_base = seqlen_info.offset_batch_Q(
            Lse + head_idx * config.QHEAD_PER_KVHEAD_PACKGQA * stride_lh,
            batch_idx,
            config.offset_q,
            config.padded_offset_q,
            stride_lb,
            stride_lm,
            HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )

        if IS_GATED:
            a_base = seqlen_info.offset_batch_Q(
                A + head_idx * config.QHEAD_PER_KVHEAD_PACKGQA * stride_ah,
                batch_idx,
                config.offset_q,
                config.padded_offset_q,
                stride_ab,
                1,
                HAS_CU_SEQLENS_Q,
                USE_PADDED=False,
            )
            d_base = seqlen_info.offset_batch_K(
                D + head_kv_idx * stride_dh,
                batch_idx,
                config.offset_k,
                config.padded_offset_k,
                stride_db,
                1,
                HAS_CU_SEQLENS_K,
                USE_PADDED=False,
            )
        else:
            a_base = tl.full((), 0, tl.int64)
            d_base = tl.full((), 0, tl.int64)

        # For split KV, offset output and LSE base pointers by split_idx
        out_base += split_idx * stride_os
        lse_base += split_idx * stride_ls

        return AttnDecPointerScheduler(
            q_base,
            k_base,
            v_base,
            a_base,
            d_base,
            out_base,
            lse_base,
            stride_qh,
            stride_kn,
            stride_vn,
            stride_oh,
        )

    @triton.jit
    def make_q_ptrs(self, config: AttnDecConfig):
        return tl.make_tensor_descriptor(
            self.q_base,
            shape=[config.actual_seqlen_q, config.head_dim],
            strides=[self.stride_qh, 1],
            block_shape=[config.TILE_M, config.TILE_K],
        )

    @triton.jit
    def make_k_ptrs(self, config: AttnDecConfig):
        return tl.make_tensor_descriptor(
            self.k_base,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_kn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_v_ptrs(self, config: AttnDecConfig):
        return tl.make_tensor_descriptor(
            self.v_base,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_vn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_a_ptrs(self, config: AttnDecConfig):
        return tl.make_tensor_descriptor(
            self.a_base,
            shape=[config.actual_seqlen_q],
            strides=[1],
            block_shape=[config.TILE_M],
        )

    @triton.jit
    def make_d_ptrs(self, config: AttnDecConfig):
        return tl.make_tensor_descriptor(
            self.d_base,
            shape=[config.actual_seqlen_k],
            strides=[1],
            block_shape=[config.TILE_N],
        )

    @triton.jit
    def make_out_ptrs(self, config: AttnDecConfig):
        return tl.make_tensor_descriptor(
            self.out_base,
            shape=[config.actual_seqlen_q, config.head_dim],
            strides=[self.stride_oh, 1],
            block_shape=[config.TILE_M, config.TILE_K],
        )

    @triton.jit
    def make_lse_ptrs(self, config: AttnDecConfig):
        return tl.make_tensor_descriptor(
            self.lse_base,
            shape=[config.actual_seqlen_q],
            strides=[1],
            block_shape=[config.TILE_M],
        )

    @triton.jit
    def load_q(self, config: AttnDecConfig, q_ptrs):
        return q_ptrs.load([0, 0])

    @triton.jit
    def load_k(self, config: AttnDecConfig, k_ptrs, n_block):
        return k_ptrs.load([n_block * config.TILE_N, 0])

    @triton.jit
    def load_v(self, config: AttnDecConfig, v_ptrs, n_block):
        return v_ptrs.load([n_block * config.TILE_N, 0])

    @triton.jit
    def load_a(self, config: AttnDecConfig, a_ptrs):
        return a_ptrs.load([0]).to(tl.float32)

    @triton.jit
    def load_d(self, config: AttnDecConfig, d_ptrs, n_block):
        return d_ptrs.load([n_block * config.TILE_N]).to(tl.float32)

    @triton.jit
    def store_out(self, config: AttnDecConfig, out_ptrs, o_tile):
        out_ptrs.store([0, 0], o_tile)

    @triton.jit
    def store_lse(self, config: AttnDecConfig, lse_ptrs, lse_tile):
        lse_ptrs.store([0], lse_tile)

    @triton.jit
    def store_empty(self, config: AttnDecConfig, out_ptrs, lse_ptrs, Out):
        lse_tile = tl.full((config.TILE_M,), float("-inf"), dtype=tl.float32)
        self.store_lse(config, lse_ptrs, lse_tile)
        o_tile = tl.zeros((config.TILE_M, config.TILE_K), dtype=Out.dtype.element_ty)
        self.store_out(config, out_ptrs, o_tile)


@aggregate
class AttnMaskScheduler:
    fixed_block: tl.tensor
    actual_seqlen_q: tl.tensor
    actual_seqlen_k: tl.tensor
    window_size_left: tl.tensor
    window_size_right: tl.tensor
    TILE_M: tl.constexpr
    TILE_N: tl.constexpr
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr
    SWAP_AB: tl.constexpr

    @constexpr_function
    def __init__(
        self,
        fixed_block,
        actual_seqlen_q,
        actual_seqlen_k,
        window_size_left,
        window_size_right,
        TILE_M,
        TILE_N,
        QHEAD_PER_KVHEAD_PACKGQA,
        SWAP_AB,
    ):
        self.fixed_block = fixed_block
        self.actual_seqlen_q = actual_seqlen_q
        self.actual_seqlen_k = actual_seqlen_k
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.TILE_M = tl.constexpr(TILE_M)
        self.TILE_N = tl.constexpr(TILE_N)
        self.QHEAD_PER_KVHEAD_PACKGQA = tl.constexpr(QHEAD_PER_KVHEAD_PACKGQA)
        self.SWAP_AB = tl.constexpr(SWAP_AB)

    @staticmethod
    @triton.jit
    def create(
        config,
        SWAP_AB: tl.constexpr = False,
    ):
        if not SWAP_AB:
            return AttnMaskScheduler(
                config.m_block,
                config.actual_seqlen_q,
                config.actual_seqlen_k,
                config.window_size_left,
                config.window_size_right,
                config.TILE_M,
                config.TILE_N,
                config.QHEAD_PER_KVHEAD_PACKGQA,
                False,
            )
        else:
            return AttnMaskScheduler(
                config.n_block,
                config.actual_seqlen_q,
                config.actual_seqlen_k,
                config.window_size_left,
                config.window_size_right,
                config.TILE_M,
                config.TILE_N,
                1,
                True,
            )

    @triton.jit
    def apply_mask(
        self,
        acc_s,
        iter_block,
        MASK_CAUSAL: tl.constexpr = False,
        MASK_LOCAL: tl.constexpr = False,
    ):
        if not self.SWAP_AB:
            m_block = self.fixed_block
            n_block = iter_block
        else:
            m_block = iter_block
            n_block = self.fixed_block
        return mask.apply_mask(
            acc_s=acc_s,
            m_block=m_block,
            n_block=n_block,
            seqlen_q=self.actual_seqlen_q,
            seqlen_k=self.actual_seqlen_k,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            MASK_SEQLEN=True,
            MASK_CAUSAL=MASK_CAUSAL,
            MASK_LOCAL=MASK_LOCAL,
            TILE_M=self.TILE_M,
            TILE_N=self.TILE_N,
            QHEAD_PER_KVHEAD_PACKGQA=self.QHEAD_PER_KVHEAD_PACKGQA,
            SWAP_AB=self.SWAP_AB,
        )


@aggregate
class SoftmaxScheduler:
    softmax_scale_log2: tl.tensor
    value_scale: tl.tensor

    @constexpr_function
    def __init__(
        self,
        softmax_scale_log2,
        value_scale,
    ):
        self.softmax_scale_log2 = softmax_scale_log2
        self.value_scale = value_scale

    @staticmethod
    @triton.jit
    def create(config):
        return SoftmaxScheduler(
            config.softmax_scale_log2,
            config.value_scale,
        )

    @triton.jit
    def online_softmax(
        self,
        acc_s,
        row_max,
        row_sum,
        CHECK_INF: tl.constexpr = False,
        RESCALE_THRESHOLD: tl.constexpr = 0.0,
    ):
        return activations.online_softmax(
            acc_s=acc_s,
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=self.softmax_scale_log2,
            CHECK_INF=CHECK_INF,
            RESCALE_THRESHOLD=RESCALE_THRESHOLD,
        )

    @triton.jit
    def online_sparse_softmax(
        self,
        acc_s,
        row_max,
        row_sum,
        softmax_threshold_log2,
        CHECK_INF: tl.constexpr = False,
    ):
        return activations.online_sparse_softmax(
            acc_s=acc_s,
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=self.softmax_scale_log2,
            softmax_threshold_log2=softmax_threshold_log2,
            CHECK_INF=CHECK_INF,
        )

    @triton.jit
    def rescale_o(
        self,
        acc_o,
        row_scale,
        LAZY_RESCALE: tl.constexpr = False,
    ):
        return activations.rescale_o(
            acc_o=acc_o,
            row_scale=row_scale,
            LAZY_RESCALE=LAZY_RESCALE,
        )

    @triton.jit
    def finalize(
        self,
        row_max,
        row_sum,
        IS_LOG2: tl.constexpr = False,
        CHECK_NAN: tl.constexpr = True,
    ):
        return activations.finalize(
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=self.softmax_scale_log2,
            final_scale=self.value_scale,
            IS_LOG2=IS_LOG2,
            CHECK_NAN=CHECK_NAN,
        )

    @triton.jit
    def log_sigmoid(
        self,
        acc_s,
        FASTMATH: tl.constexpr = False,
    ):
        return activations.log_sigmoid(x=acc_s, FASTMATH=FASTMATH)

    @triton.jit
    def online_gate(
        self,
        a_max,
        a_min,
        d_max,
        d_min,
        gate_max,
        gate_threshold_log2,
    ):
        return activations.online_gate(
            a_max=a_max,
            a_min=a_min,
            d_max=d_max,
            d_min=d_min,
            gate_max=gate_max,
            scale_log2=self.softmax_scale_log2,
            gate_threshold_log2=gate_threshold_log2,
        )
