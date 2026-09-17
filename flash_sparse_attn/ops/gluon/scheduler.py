# Copyright (c) 2026, Jingze Shi.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from flash_sparse_attn.ops.gluon.seqlen_info import (
    get_seqlen_info_qk,
    get_softmax_threshold,
    offset_batch_Q,
    offset_batch_K,
    make_ptrs,
    make_pack_gqa_ptrs,
)
from flash_sparse_attn.ops.gluon.block_info import (
    get_n_block_min_max,
    get_m_block_min_max,
    get_n_block_max_no_causal_local_mask,
    get_m_block_min_no_causal_local_mask,
    get_n_block_min_before_local_mask,
    get_m_block_max_before_local_mask,
)
from flash_sparse_attn.ops.gluon.mask import apply_mask
from flash_sparse_attn.ops.gluon.softmax import (
    online_softmax,
    online_sparse_softmax,
    rescale_o,
    finalize,
)


@aggregate
class AttnFwdGridIndex:
    m_block: gl.tensor
    batch_idx: gl.tensor
    head_idx: gl.tensor
    head_kv_idx: gl.tensor
    split_idx: gl.tensor

    @gluon.constexpr_function
    def __init__(self, m_block, batch_idx, head_idx, head_kv_idx, split_idx):
        self.m_block = m_block
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx

    @staticmethod
    @gluon.jit
    def create(
        NUM_SPLITS: gl.constexpr,
        QHEAD_PER_KVHEAD: gl.constexpr,
        IS_SPLIT_KV: gl.constexpr,
        PACK_GQA: gl.constexpr,
    ) -> "AttnFwdGridIndex":
        """
        Decode the forward program IDs into attention grid indices.

        :param NUM_SPLITS: number of KV splits
        :type NUM_SPLITS: int
        :param QHEAD_PER_KVHEAD: ratio of query heads to key/value heads
        :type QHEAD_PER_KVHEAD: int
        :param IS_SPLIT_KV: boolean flag indicating if the KV range is split
        :type IS_SPLIT_KV: bool
        :param PACK_GQA: boolean flag indicating if packed GQA is enabled
        :type PACK_GQA: bool

        :return: forward grid indices for the current program
        """
        m_block = gl.program_id(0)
        head_idx = gl.program_id(1)
        batch_split_idx = gl.program_id(2)
        if IS_SPLIT_KV:
            batch_idx = batch_split_idx // NUM_SPLITS
            split_idx = batch_split_idx - batch_idx * NUM_SPLITS
        else:
            batch_idx = batch_split_idx
            split_idx = gl.to_tensor(0)
        if PACK_GQA:
            head_kv_idx = head_idx
        else:
            head_kv_idx = head_idx // QHEAD_PER_KVHEAD
        return AttnFwdGridIndex(
            m_block,
            batch_idx,
            head_idx,
            head_kv_idx,
            split_idx,
        )

    @gluon.jit
    def load_window_sizes(
        self,
        window_sizes: gl.tensor,
        stride_wh: gl.tensor,
        IS_LOCAL: gl.constexpr,
    ) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
        """
        Load the local-attention window sizes for the current KV head.

        :param window_sizes: base pointer to per-head window sizes
        :type window_sizes: tensor
        :param stride_wh: stride between window-size records for adjacent heads
        :type stride_wh: tensor
        :param IS_LOCAL: boolean flag indicating if local attention is enabled
        :type IS_LOCAL: bool

        :return window_size_sink: prefix-sink token count
        :return window_size_left: distant local band token count
        :return window_size_right: gap token count after the near-diagonal window
        :return window_size_near: near-diagonal local token count
        """
        if IS_LOCAL:
            window_size_sink = gl.load(window_sizes + self.head_kv_idx * stride_wh)
            window_size_left = gl.load(window_sizes + self.head_kv_idx * stride_wh + 1)
            window_size_right = gl.load(window_sizes + self.head_kv_idx * stride_wh + 2)
            window_size_near = gl.load(window_sizes + self.head_kv_idx * stride_wh + 3)
        else:
            window_size_sink = gl.to_tensor(0)
            window_size_left = gl.to_tensor(0)
            window_size_right = gl.to_tensor(0)
            window_size_near = gl.to_tensor(0)
        return window_size_sink, window_size_left, window_size_right, window_size_near


@aggregate
class AttnBwdGridIndex:
    n_block: gl.tensor
    batch_idx: gl.tensor
    head_idx: gl.tensor
    head_kv_idx: gl.tensor
    split_idx: gl.tensor

    @gluon.constexpr_function
    def __init__(self, n_block, batch_idx, head_idx, head_kv_idx, split_idx):
        self.n_block = n_block
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx

    @staticmethod
    @gluon.jit
    def create(
        NUM_SPLITS: gl.constexpr,
        QHEAD_PER_KVHEAD: gl.constexpr,
        IS_SPLIT_QO: gl.constexpr,
    ) -> "AttnBwdGridIndex":
        """
        Decode the backward program IDs into attention grid indices.

        :param NUM_SPLITS: number of QO splits
        :type NUM_SPLITS: int
        :param QHEAD_PER_KVHEAD: ratio of query heads to key/value heads
        :type QHEAD_PER_KVHEAD: int
        :param IS_SPLIT_QO: boolean flag indicating if the QO range is split
        :type IS_SPLIT_QO: bool

        :return: backward grid indices for the current program
        """
        n_block = gl.program_id(0)
        head_idx = gl.program_id(1)
        batch_split_idx = gl.program_id(2)
        if IS_SPLIT_QO:
            batch_idx = batch_split_idx // NUM_SPLITS
            split_idx = batch_split_idx - batch_idx * NUM_SPLITS
        else:
            batch_idx = batch_split_idx
            split_idx = gl.to_tensor(0)
        head_kv_idx = head_idx // QHEAD_PER_KVHEAD
        return AttnBwdGridIndex(
            n_block,
            batch_idx,
            head_idx,
            head_kv_idx,
            split_idx,
        )

    @gluon.jit
    def load_window_sizes(
        self,
        window_sizes: gl.tensor,
        stride_wh: gl.tensor,
        IS_LOCAL: gl.constexpr,
    ) -> tuple[gl.tensor, gl.tensor, gl.tensor, gl.tensor]:
        """
        Load the local-attention window sizes for the current KV head.

        :param window_sizes: base pointer to per-head window sizes
        :type window_sizes: tensor
        :param stride_wh: stride between window-size records for adjacent heads
        :type stride_wh: tensor
        :param IS_LOCAL: boolean flag indicating if local attention is enabled
        :type IS_LOCAL: bool

        :return window_size_sink: prefix-sink token count
        :return window_size_left: distant local band token count
        :return window_size_right: gap token count after the near-diagonal window
        :return window_size_near: near-diagonal local token count
        """
        if IS_LOCAL:
            window_size_sink = gl.load(window_sizes + self.head_kv_idx * stride_wh)
            window_size_left = gl.load(window_sizes + self.head_kv_idx * stride_wh + 1)
            window_size_right = gl.load(window_sizes + self.head_kv_idx * stride_wh + 2)
            window_size_near = gl.load(window_sizes + self.head_kv_idx * stride_wh + 3)
        else:
            window_size_sink = gl.to_tensor(0)
            window_size_left = gl.to_tensor(0)
            window_size_right = gl.to_tensor(0)
            window_size_near = gl.to_tensor(0)
        return window_size_sink, window_size_left, window_size_right, window_size_near


@aggregate
class AttnFwdConfig:
    batch_idx: gl.tensor
    head_idx: gl.tensor
    head_kv_idx: gl.tensor
    split_idx: gl.tensor
    m_block: gl.tensor
    softmax_scale_log2: gl.tensor
    softmax_threshold_log2: gl.tensor
    value_scale: gl.tensor
    actual_seqlen_q: gl.tensor
    actual_seqlen_k: gl.tensor
    offset_q: gl.tensor
    offset_k: gl.tensor
    padded_offset_q: gl.tensor
    padded_offset_k: gl.tensor
    window_size_sink: gl.tensor
    window_size_left: gl.tensor
    window_size_right: gl.tensor
    window_size_near: gl.tensor
    head_dim: gl.tensor
    PACK_GQA: gl.constexpr
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr
    NUM_SPLITS: gl.constexpr
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    TILE_K: gl.constexpr
    IS_CAUSAL: gl.constexpr
    IS_LOCAL: gl.constexpr
    IS_SPLIT_KV: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        batch_idx,
        head_idx,
        head_kv_idx,
        split_idx,
        m_block,
        softmax_scale_log2,
        softmax_threshold_log2,
        value_scale,
        actual_seqlen_q,
        actual_seqlen_k,
        offset_q,
        offset_k,
        padded_offset_q,
        padded_offset_k,
        window_size_sink,
        window_size_left,
        window_size_right,
        window_size_near,
        head_dim,
        PACK_GQA,
        QHEAD_PER_KVHEAD_PACKGQA,
        NUM_SPLITS,
        TILE_M,
        TILE_N,
        TILE_K,
        IS_CAUSAL,
        IS_LOCAL,
        IS_SPLIT_KV,
    ):
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx
        self.m_block = m_block
        self.softmax_scale_log2 = softmax_scale_log2
        self.softmax_threshold_log2 = softmax_threshold_log2
        self.value_scale = value_scale
        self.actual_seqlen_q = actual_seqlen_q
        self.actual_seqlen_k = actual_seqlen_k
        self.offset_q = offset_q
        self.offset_k = offset_k
        self.padded_offset_q = padded_offset_q
        self.padded_offset_k = padded_offset_k
        self.window_size_sink = window_size_sink
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.window_size_near = window_size_near
        self.head_dim = head_dim
        self.PACK_GQA = gl.constexpr(PACK_GQA)
        self.QHEAD_PER_KVHEAD_PACKGQA = gl.constexpr(QHEAD_PER_KVHEAD_PACKGQA)
        self.NUM_SPLITS = gl.constexpr(NUM_SPLITS)
        self.TILE_M = gl.constexpr(TILE_M)
        self.TILE_N = gl.constexpr(TILE_N)
        self.TILE_K = gl.constexpr(TILE_K)
        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.IS_LOCAL = gl.constexpr(IS_LOCAL)
        self.IS_SPLIT_KV = gl.constexpr(IS_SPLIT_KV)

    @staticmethod
    @gluon.jit
    def create(
        batch_idx,
        head_idx,
        head_kv_idx,
        split_idx,
        m_block,
        row_offsets,
        softmax_scale=0.0,
        softmax_threshold=0.0,
        query_scale=None,
        key_scale=None,
        value_scale=None,
        window_size_sink=0,
        window_size_left=0,
        window_size_right=0,
        window_size_near=0,
        head_dim=0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        seqlen_q=0,
        seqlen_k=0,
        PACK_GQA: gl.constexpr = False,
        QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr = 1,
        NUM_SPLITS: gl.constexpr = 1,
        TILE_M: gl.constexpr = 64,
        TILE_N: gl.constexpr = 64,
        TILE_K: gl.constexpr = 64,
        IS_CAUSAL: gl.constexpr = False,
        IS_LOCAL: gl.constexpr = False,
        IS_SPLIT_KV: gl.constexpr = False,
        IS_QUANT: gl.constexpr = False,
        HAS_CU_SEQLENS_Q: gl.constexpr = False,
        HAS_CU_SEQLENS_K: gl.constexpr = False,
        HAS_SEQUSED_Q: gl.constexpr = False,
        HAS_SEQUSED_K: gl.constexpr = False,
    ):
        # Get seqlen info for this batch
        (
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            actual_seqlen_q,
            actual_seqlen_k,
        ) = get_seqlen_info_qk(
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
        if IS_QUANT:
            q_scale = gl.load(query_scale)
            k_scale = gl.load(key_scale)
            v_scale = gl.load(value_scale)
        else:
            q_scale = gl.to_tensor(1.0)
            k_scale = gl.to_tensor(1.0)
            v_scale = gl.to_tensor(1.0)
        LOG2E: gl.constexpr = 1.44269504089
        softmax_scale_log2 = softmax_scale * LOG2E * q_scale * k_scale
        softmax_threshold_log2 = get_softmax_threshold(
            softmax_threshold=softmax_threshold,
            m_block=m_block,
            seqlen_q=actual_seqlen_q,
            seqlen_k=actual_seqlen_k,
            row_offsets=row_offsets,
            IS_CAUSAL=IS_CAUSAL,
            QHEAD_PER_KVHEAD_PACKGQA=QHEAD_PER_KVHEAD_PACKGQA,
        )
        return AttnFwdConfig(
            gl.to_tensor(batch_idx),
            gl.to_tensor(head_idx),
            gl.to_tensor(head_kv_idx),
            gl.to_tensor(split_idx),
            gl.to_tensor(m_block),
            gl.to_tensor(softmax_scale_log2),
            gl.to_tensor(softmax_threshold_log2),
            gl.to_tensor(v_scale),
            gl.to_tensor(actual_seqlen_q),
            gl.to_tensor(actual_seqlen_k),
            gl.to_tensor(offset_q),
            gl.to_tensor(offset_k),
            gl.to_tensor(padded_offset_q),
            gl.to_tensor(padded_offset_k),
            gl.to_tensor(window_size_sink),
            gl.to_tensor(window_size_left),
            gl.to_tensor(window_size_right),
            gl.to_tensor(window_size_near),
            gl.to_tensor(head_dim),
            PACK_GQA,
            QHEAD_PER_KVHEAD_PACKGQA,
            NUM_SPLITS,
            TILE_M,
            TILE_N,
            TILE_K,
            IS_CAUSAL,
            IS_LOCAL,
            IS_SPLIT_KV,
        )

    @gluon.jit
    def get_offs_m(self, layout: gl.constexpr):
        return self.m_block * self.TILE_M + gl.arange(0, self.TILE_M, layout)

    @gluon.jit
    def get_offs_n(self, n_block, layout: gl.constexpr):
        return n_block * self.TILE_N + gl.arange(0, self.TILE_N, layout)

    @gluon.jit
    def get_offs_k(self, layout: gl.constexpr):
        return gl.arange(0, self.TILE_K, layout)


@aggregate
class AttnBwdConfig:
    batch_idx: gl.tensor
    head_idx: gl.tensor
    head_kv_idx: gl.tensor
    split_idx: gl.tensor
    n_block: gl.tensor
    row_offsets: gl.tensor
    softmax_scale_log2: gl.tensor
    softmax_threshold: gl.tensor
    query_scale: gl.tensor
    key_scale: gl.tensor
    value_scale: gl.tensor
    actual_seqlen_q: gl.tensor
    actual_seqlen_k: gl.tensor
    offset_q: gl.tensor
    offset_k: gl.tensor
    padded_offset_q: gl.tensor
    padded_offset_k: gl.tensor
    window_size_sink: gl.tensor
    window_size_left: gl.tensor
    window_size_right: gl.tensor
    window_size_near: gl.tensor
    head_dim: gl.tensor
    QHEAD_PER_KVHEAD: gl.constexpr
    NUM_SPLITS: gl.constexpr
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    TILE_K: gl.constexpr
    IS_CAUSAL: gl.constexpr
    IS_LOCAL: gl.constexpr
    IS_SPLIT_QO: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        batch_idx,
        head_idx,
        head_kv_idx,
        split_idx,
        n_block,
        row_offsets,
        softmax_scale_log2,
        softmax_threshold,
        query_scale,
        key_scale,
        value_scale,
        actual_seqlen_q,
        actual_seqlen_k,
        offset_q,
        offset_k,
        padded_offset_q,
        padded_offset_k,
        window_size_sink,
        window_size_left,
        window_size_right,
        window_size_near,
        head_dim,
        QHEAD_PER_KVHEAD,
        NUM_SPLITS,
        TILE_M,
        TILE_N,
        TILE_K,
        IS_CAUSAL,
        IS_LOCAL,
        IS_SPLIT_QO,
    ):
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx
        self.n_block = n_block
        self.row_offsets = row_offsets
        self.softmax_scale_log2 = softmax_scale_log2
        self.softmax_threshold = softmax_threshold
        self.query_scale = query_scale
        self.key_scale = key_scale
        self.value_scale = value_scale
        self.actual_seqlen_q = actual_seqlen_q
        self.actual_seqlen_k = actual_seqlen_k
        self.offset_q = offset_q
        self.offset_k = offset_k
        self.padded_offset_q = padded_offset_q
        self.padded_offset_k = padded_offset_k
        self.window_size_sink = window_size_sink
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.window_size_near = window_size_near
        self.head_dim = head_dim
        self.QHEAD_PER_KVHEAD = gl.constexpr(QHEAD_PER_KVHEAD)
        self.NUM_SPLITS = gl.constexpr(NUM_SPLITS)
        self.TILE_M = gl.constexpr(TILE_M)
        self.TILE_N = gl.constexpr(TILE_N)
        self.TILE_K = gl.constexpr(TILE_K)
        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.IS_LOCAL = gl.constexpr(IS_LOCAL)
        self.IS_SPLIT_QO = gl.constexpr(IS_SPLIT_QO)

    @staticmethod
    @gluon.jit
    def create(
        batch_idx,
        head_idx,
        head_kv_idx,
        split_idx,
        n_block,
        row_offsets,
        softmax_scale=0.0,
        softmax_threshold=0.0,
        query_scale=None,
        key_scale=None,
        value_scale=None,
        window_size_sink=0,
        window_size_left=0,
        window_size_right=0,
        window_size_near=0,
        head_dim=0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        seqlen_q=0,
        seqlen_k=0,
        QHEAD_PER_KVHEAD: gl.constexpr = 1,
        NUM_SPLITS: gl.constexpr = 1,
        TILE_M: gl.constexpr = 64,
        TILE_N: gl.constexpr = 64,
        TILE_K: gl.constexpr = 64,
        IS_CAUSAL: gl.constexpr = False,
        IS_LOCAL: gl.constexpr = False,
        IS_SPLIT_QO: gl.constexpr = False,
        IS_QUANT: gl.constexpr = False,
        HAS_CU_SEQLENS_Q: gl.constexpr = False,
        HAS_CU_SEQLENS_K: gl.constexpr = False,
        HAS_SEQUSED_Q: gl.constexpr = False,
        HAS_SEQUSED_K: gl.constexpr = False,
    ):
        # Get seqlen info for this batch
        (
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            actual_seqlen_q,
            actual_seqlen_k,
        ) = get_seqlen_info_qk(
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
        if IS_QUANT:
            q_scale = gl.load(query_scale)
            k_scale = gl.load(key_scale)
            v_scale = gl.load(value_scale)
        else:
            q_scale = gl.to_tensor(1.0)
            k_scale = gl.to_tensor(1.0)
            v_scale = gl.to_tensor(1.0)
        LOG2E: gl.constexpr = 1.44269504089
        softmax_scale_log2 = softmax_scale * LOG2E
        return AttnBwdConfig(
            gl.to_tensor(batch_idx),
            gl.to_tensor(head_idx),
            gl.to_tensor(head_kv_idx),
            gl.to_tensor(split_idx),
            gl.to_tensor(n_block),
            gl.to_tensor(row_offsets),
            gl.to_tensor(softmax_scale_log2),
            gl.to_tensor(softmax_threshold),
            gl.to_tensor(q_scale),
            gl.to_tensor(k_scale),
            gl.to_tensor(v_scale),
            gl.to_tensor(actual_seqlen_q),
            gl.to_tensor(actual_seqlen_k),
            gl.to_tensor(offset_q),
            gl.to_tensor(offset_k),
            gl.to_tensor(padded_offset_q),
            gl.to_tensor(padded_offset_k),
            gl.to_tensor(window_size_sink),
            gl.to_tensor(window_size_left),
            gl.to_tensor(window_size_right),
            gl.to_tensor(window_size_near),
            gl.to_tensor(head_dim),
            QHEAD_PER_KVHEAD,
            NUM_SPLITS,
            TILE_M,
            TILE_N,
            TILE_K,
            IS_CAUSAL,
            IS_LOCAL,
            IS_SPLIT_QO,
        )

    @gluon.jit
    def get_softmax_threshold_log2(self, m_block):
        return get_softmax_threshold(
            softmax_threshold=self.softmax_threshold,
            m_block=m_block,
            seqlen_q=self.actual_seqlen_q,
            seqlen_k=self.actual_seqlen_k,
            row_offsets=self.row_offsets,
            IS_CAUSAL=self.IS_CAUSAL,
            QHEAD_PER_KVHEAD_PACKGQA=1,
        )

    @gluon.jit
    def get_offs_m(self, m_block, layout: gl.constexpr):
        return m_block * self.TILE_M + gl.arange(0, self.TILE_M, layout)

    @gluon.jit
    def get_offs_n(self, layout: gl.constexpr):
        return self.n_block * self.TILE_N + gl.arange(0, self.TILE_N, layout)

    @gluon.jit
    def get_offs_k(self, layout: gl.constexpr):
        return gl.arange(0, self.TILE_K, layout)


@aggregate
class AttnFwdBlockScheduler:
    n_block_min: gl.tensor
    n_block_max: gl.tensor
    n_block_max_no_mask: gl.tensor
    n_block_window_min: gl.tensor
    n_block_window_max: gl.tensor
    n_block_window_min_no_mask: gl.tensor
    n_block_window_max_no_mask: gl.tensor
    n_block_sink_min: gl.tensor
    n_block_sink_max: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        n_block_min,
        n_block_max,
        n_block_max_no_mask,
        n_block_window_min,
        n_block_window_max,
        n_block_window_min_no_mask,
        n_block_window_max_no_mask,
        n_block_sink_min,
        n_block_sink_max,
    ):
        self.n_block_min = n_block_min
        self.n_block_max = n_block_max
        self.n_block_max_no_mask = n_block_max_no_mask
        self.n_block_window_min = n_block_window_min
        self.n_block_window_max = n_block_window_max
        self.n_block_window_min_no_mask = n_block_window_min_no_mask
        self.n_block_window_max_no_mask = n_block_window_max_no_mask
        self.n_block_sink_min = n_block_sink_min
        self.n_block_sink_max = n_block_sink_max

    @gluon.jit
    def is_empty(self):
        return (
            (self.n_block_max <= self.n_block_min)
            & (self.n_block_window_max <= self.n_block_window_min)
            & (self.n_block_sink_max <= self.n_block_sink_min)
        )

    @staticmethod
    @gluon.jit
    def create(config: AttnFwdConfig):
        # Compute causal n_block range for this m_block
        (
            n_block_min,
            n_block_max,
            n_block_window_min,
            n_block_window_max,
            n_block_sink_min,
            n_block_sink_max,
        ) = get_n_block_min_max(
            seqlen_q=config.actual_seqlen_q,
            seqlen_k=config.actual_seqlen_k,
            m_block=config.m_block,
            split_idx=config.split_idx,
            window_size_sink=config.window_size_sink,
            window_size_left=config.window_size_left,
            window_size_right=config.window_size_right,
            window_size_near=config.window_size_near,
            NUM_SPLITS=config.NUM_SPLITS,
            TILE_N=config.TILE_N,
            TILE_M=config.TILE_M,
            IS_CAUSAL=config.IS_CAUSAL,
            IS_LOCAL=config.IS_LOCAL,
            IS_SPLIT_KV=config.IS_SPLIT_KV,
            QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
        )
        n_block_max_no_mask = get_n_block_max_no_causal_local_mask(
            seqlen_q=config.actual_seqlen_q,
            seqlen_k=config.actual_seqlen_k,
            m_block=config.m_block,
            n_block_min=n_block_min,
            window_size_right=0,
            window_size_near=0,
            TILE_N=config.TILE_N,
            TILE_M=config.TILE_M,
            IS_LOCAL=False,
            QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
        )

        # Clamp to split's range so the no-mask loop stays within bounds
        if config.IS_SPLIT_KV:
            n_block_max_no_mask = gl.minimum(n_block_max_no_mask, n_block_max)

        if config.IS_LOCAL:
            # Compute local n_block range for this m_block
            if config.IS_SPLIT_KV:
                n_block_max_no_mask = gl.where(
                    config.split_idx >= config.NUM_SPLITS - 1,
                    n_block_min,
                    n_block_max_no_mask,
                )
            else:
                n_block_max_no_mask = n_block_min
            n_block_window_max_no_mask = get_n_block_max_no_causal_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                m_block=config.m_block,
                n_block_min=n_block_window_min,
                window_size_right=config.window_size_right,
                window_size_near=config.window_size_near,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_LOCAL=True,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
            n_block_window_min_no_mask = get_n_block_min_before_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                m_block=config.m_block,
                n_block_min=n_block_window_min,
                window_size_left=config.window_size_left,
                window_size_right=config.window_size_right,
                window_size_near=config.window_size_near,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_LOCAL=True,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
            # Clamp window no-mask boundaries to the window's range
            n_block_window_max_no_mask = gl.maximum(
                gl.minimum(n_block_window_max_no_mask, n_block_window_max),
                n_block_window_min,
            )
            n_block_window_min_no_mask = gl.maximum(
                gl.minimum(n_block_window_min_no_mask, n_block_window_max_no_mask),
                n_block_window_min,
            )
        else:
            n_block_window_min = gl.to_tensor(0)
            n_block_window_max = gl.to_tensor(0)
            n_block_window_min_no_mask = gl.to_tensor(0)
            n_block_window_max_no_mask = gl.to_tensor(0)

        return AttnFwdBlockScheduler(
            n_block_min,
            n_block_max,
            n_block_max_no_mask,
            n_block_window_min,
            n_block_window_max,
            n_block_window_min_no_mask,
            n_block_window_max_no_mask,
            n_block_sink_min,
            n_block_sink_max,
        )


@aggregate
class AttnBwdBlockScheduler:
    m_block_min: gl.tensor
    m_block_max: gl.tensor
    m_block_min_no_mask: gl.tensor
    m_block_window_min: gl.tensor
    m_block_window_max: gl.tensor
    m_block_window_min_no_mask: gl.tensor
    m_block_window_max_no_mask: gl.tensor
    m_block_sink_min: gl.tensor
    m_block_sink_max: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        m_block_min,
        m_block_max,
        m_block_min_no_mask,
        m_block_window_min,
        m_block_window_max,
        m_block_window_min_no_mask,
        m_block_window_max_no_mask,
        m_block_sink_min,
        m_block_sink_max,
    ):
        self.m_block_min = m_block_min
        self.m_block_max = m_block_max
        self.m_block_min_no_mask = m_block_min_no_mask
        self.m_block_window_min = m_block_window_min
        self.m_block_window_max = m_block_window_max
        self.m_block_window_min_no_mask = m_block_window_min_no_mask
        self.m_block_window_max_no_mask = m_block_window_max_no_mask
        self.m_block_sink_min = m_block_sink_min
        self.m_block_sink_max = m_block_sink_max

    @gluon.jit
    def is_empty(self):
        return (
            (self.m_block_max <= self.m_block_min)
            & (self.m_block_window_max <= self.m_block_window_min)
            & (self.m_block_sink_max <= self.m_block_sink_min)
        )

    @staticmethod
    @gluon.jit
    def create(config: AttnBwdConfig):
        # Compute causal m_block range for this n_block
        (
            m_block_min,
            m_block_max,
            m_block_window_min,
            m_block_window_max,
            m_block_sink_min,
            m_block_sink_max,
        ) = get_m_block_min_max(
            seqlen_q=config.actual_seqlen_q,
            seqlen_k=config.actual_seqlen_k,
            n_block=config.n_block,
            split_idx=config.split_idx,
            window_size_sink=config.window_size_sink,
            window_size_left=config.window_size_left,
            window_size_right=config.window_size_right,
            window_size_near=config.window_size_near,
            NUM_SPLITS=config.NUM_SPLITS,
            TILE_N=config.TILE_N,
            TILE_M=config.TILE_M,
            IS_CAUSAL=config.IS_CAUSAL,
            IS_LOCAL=config.IS_LOCAL,
            IS_SPLIT_QO=config.IS_SPLIT_QO,
        )
        m_block_min_no_mask = get_m_block_min_no_causal_local_mask(
            seqlen_q=config.actual_seqlen_q,
            seqlen_k=config.actual_seqlen_k,
            n_block=config.n_block,
            m_block_min=m_block_min,
            window_size_right=0,
            window_size_near=0,
            TILE_N=config.TILE_N,
            TILE_M=config.TILE_M,
            IS_CAUSAL=config.IS_CAUSAL or config.IS_LOCAL,
            IS_LOCAL=False,
        )

        # Clamp to split's range so the no-mask loop stays within bounds
        if config.IS_SPLIT_QO:
            m_block_min_no_mask = gl.minimum(m_block_min_no_mask, m_block_max)

        if config.IS_LOCAL:
            # Compute local m_block range for this n_block
            m_block_min_no_mask = m_block_max
            m_block_window_min_no_mask = get_m_block_min_no_causal_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                n_block=config.n_block,
                m_block_min=m_block_window_min,
                window_size_right=config.window_size_right,
                window_size_near=config.window_size_near,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_CAUSAL=False,
                IS_LOCAL=True,
            )
            m_block_window_min_no_mask = gl.maximum(
                m_block_window_min_no_mask, m_block_window_min
            )
            m_block_window_max_no_mask = get_m_block_max_before_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                n_block=config.n_block,
                m_block_max=m_block_window_max,
                window_size_left=config.window_size_left,
                window_size_right=config.window_size_right,
                window_size_near=config.window_size_near,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_LOCAL=True,
            )
            # Clamp window no-mask boundaries to the window's range
            m_block_window_max_no_mask = gl.maximum(
                m_block_window_max_no_mask, m_block_window_min_no_mask
            )
            m_block_window_min_no_mask = gl.maximum(
                gl.minimum(m_block_window_min_no_mask, m_block_window_max),
                m_block_window_min,
            )
            m_block_window_max_no_mask = gl.maximum(
                gl.minimum(m_block_window_max_no_mask, m_block_window_max),
                m_block_window_min_no_mask,
            )
        else:
            m_block_window_min_no_mask = gl.to_tensor(0)
            m_block_window_max_no_mask = gl.to_tensor(0)
            m_block_sink_min = gl.to_tensor(0)
            m_block_sink_max = gl.to_tensor(0)

        return AttnBwdBlockScheduler(
            m_block_min,
            m_block_max,
            m_block_min_no_mask,
            m_block_window_min,
            m_block_window_max,
            m_block_window_min_no_mask,
            m_block_window_max_no_mask,
            m_block_sink_min,
            m_block_sink_max,
        )


@aggregate
class AttnFwdPointerScheduler:
    q_base: gl.tensor
    k_base: gl.tensor
    v_base: gl.tensor
    out_base: gl.tensor
    lse_base: gl.tensor
    stride_qh: gl.tensor
    stride_qm: gl.tensor
    stride_kb: gl.tensor
    stride_kn: gl.tensor
    stride_vb: gl.tensor
    stride_vn: gl.tensor
    stride_oh: gl.tensor
    stride_om: gl.tensor
    stride_lh: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        q_base,
        k_base,
        v_base,
        out_base,
        lse_base,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kn,
        stride_vb,
        stride_vn,
        stride_oh,
        stride_om,
        stride_lh,
    ):
        self.q_base = q_base
        self.k_base = k_base
        self.v_base = v_base
        self.out_base = out_base
        self.lse_base = lse_base
        self.stride_qh = stride_qh
        self.stride_qm = stride_qm
        self.stride_kb = stride_kb
        self.stride_kn = stride_kn
        self.stride_vb = stride_vb
        self.stride_vn = stride_vn
        self.stride_oh = stride_oh
        self.stride_om = stride_om
        self.stride_lh = stride_lh

    @staticmethod
    @gluon.jit
    def create(
        config: AttnFwdConfig,
        Q=None,
        K=None,
        V=None,
        Out=None,
        Lse=None,
        stride_qb=0,
        stride_qh=0,
        stride_qm=0,
        stride_kb=0,
        stride_kh=0,
        stride_kn=0,
        stride_vb=0,
        stride_vh=0,
        stride_vn=0,
        stride_ob=0,
        stride_oh=0,
        stride_om=0,
        stride_os=0,
        stride_lb=0,
        stride_lh=0,
        stride_ls=0,
        HAS_CU_SEQLENS_Q: gl.constexpr = False,
        HAS_CU_SEQLENS_K: gl.constexpr = False,
    ):
        # Initialize base pointers
        q_base = offset_batch_Q(
            base_ptr=Q + config.head_idx * stride_qh if not config.PACK_GQA else Q,
            batch_idx=config.batch_idx,
            offset=config.offset_q,
            padded_offset=config.padded_offset_q,
            stride_batch=stride_qb,
            stride_seq=stride_qm,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        k_base = offset_batch_K(
            base_ptr=K + config.head_kv_idx * stride_kh,
            batch_idx=config.batch_idx,
            offset=config.offset_k,
            padded_offset=config.padded_offset_k,
            stride_batch=stride_kb,
            stride_seq=stride_kn,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        v_base = offset_batch_K(
            base_ptr=V + config.head_kv_idx * stride_vh,
            batch_idx=config.batch_idx,
            offset=config.offset_k,
            padded_offset=config.padded_offset_k,
            stride_batch=stride_vb,
            stride_seq=stride_vn,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        out_base = offset_batch_Q(
            base_ptr=Out + config.head_idx * stride_oh if not config.PACK_GQA else Out,
            batch_idx=config.batch_idx,
            offset=config.offset_q,
            padded_offset=config.padded_offset_q,
            stride_batch=stride_ob,
            stride_seq=stride_om,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        lse_base = offset_batch_Q(
            base_ptr=Lse + config.head_idx * stride_lh if not config.PACK_GQA else Lse,
            batch_idx=config.batch_idx,
            offset=config.offset_q,
            padded_offset=config.padded_offset_q,
            stride_batch=stride_lb,
            stride_seq=1,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )

        # For split KV, offset output and LSE base pointers by split_idx
        if config.IS_SPLIT_KV:
            out_base += config.split_idx * stride_os
            lse_base += config.split_idx * stride_ls

        return AttnFwdPointerScheduler(
            q_base,
            k_base,
            v_base,
            out_base,
            lse_base,
            gl.to_tensor(stride_qh),
            gl.to_tensor(stride_qm),
            gl.to_tensor(stride_kb),
            gl.to_tensor(stride_kn),
            gl.to_tensor(stride_vb),
            gl.to_tensor(stride_vn),
            gl.to_tensor(stride_oh),
            gl.to_tensor(stride_om),
            gl.to_tensor(stride_lh),
        )

    @gluon.jit
    def make_q_ptrs(self, config: AttnFwdConfig, offs_m, offs_k):
        if config.PACK_GQA:
            return make_pack_gqa_ptrs(
                base_ptr=self.q_base,
                m_block=config.m_block,
                head_idx=config.head_idx,
                stride_head=self.stride_qh,
                stride_seq=self.stride_qm,
                offs_m=offs_m,
                offs_k=offs_k,
                TILE_K=config.TILE_K,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        return make_ptrs(
            base_ptr=self.q_base,
            mn_block=config.m_block,
            stride_seq=self.stride_qm,
            offs_mn=offs_m,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )

    @gluon.jit
    def make_k_ptrs(self, config: AttnFwdConfig, n_block, offs_n, offs_k):
        return make_ptrs(
            base_ptr=self.k_base,
            mn_block=n_block,
            stride_seq=self.stride_kn,
            offs_mn=offs_n,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )

    @gluon.jit
    def make_v_ptrs(self, config: AttnFwdConfig, n_block, offs_n, offs_k):
        return make_ptrs(
            base_ptr=self.v_base,
            mn_block=n_block,
            stride_seq=self.stride_vn,
            offs_mn=offs_n,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )

    @gluon.jit
    def make_out_ptrs(self, config: AttnFwdConfig, offs_m, offs_k):
        if config.PACK_GQA:
            return make_pack_gqa_ptrs(
                base_ptr=self.out_base,
                m_block=config.m_block,
                head_idx=config.head_idx,
                stride_head=self.stride_oh,
                stride_seq=self.stride_om,
                offs_m=offs_m,
                offs_k=offs_k,
                TILE_K=config.TILE_K,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        return make_ptrs(
            base_ptr=self.out_base,
            mn_block=config.m_block,
            stride_seq=self.stride_om,
            offs_mn=offs_m,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )

    @gluon.jit
    def make_lse_ptrs(self, config: AttnFwdConfig, offs_m):
        if config.PACK_GQA:
            return make_pack_gqa_ptrs(
                base_ptr=self.lse_base,
                m_block=config.m_block,
                head_idx=config.head_idx,
                stride_head=self.stride_lh,
                stride_seq=1,
                offs_m=offs_m,
                offs_k=offs_m,
                TILE_K=1,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        return self.lse_base + config.m_block * config.TILE_M + offs_m


@aggregate
class AttnBwdPointerScheduler:
    q_base: gl.tensor
    k_base: gl.tensor
    v_base: gl.tensor
    do_base: gl.tensor
    lse_base: gl.tensor
    dpsum_base: gl.tensor
    dq_accum_base: gl.tensor
    dk_base: gl.tensor
    dv_base: gl.tensor
    stride_qm: gl.tensor
    stride_kn: gl.tensor
    stride_vn: gl.tensor
    stride_dom: gl.tensor
    stride_dqam: gl.tensor
    stride_dkn: gl.tensor
    stride_dvn: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        q_base,
        k_base,
        v_base,
        do_base,
        lse_base,
        dpsum_base,
        dq_accum_base,
        dk_base,
        dv_base,
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
        self.do_base = do_base
        self.lse_base = lse_base
        self.dpsum_base = dpsum_base
        self.dq_accum_base = dq_accum_base
        self.dk_base = dk_base
        self.dv_base = dv_base
        self.stride_qm = stride_qm
        self.stride_kn = stride_kn
        self.stride_vn = stride_vn
        self.stride_dom = stride_dom
        self.stride_dqam = stride_dqam
        self.stride_dkn = stride_dkn
        self.stride_dvn = stride_dvn

    @staticmethod
    @gluon.jit
    def create(
        config: AttnBwdConfig,
        Q=None,
        K=None,
        V=None,
        dO=None,
        LSELog2=None,
        dPsum=None,
        dQaccum=None,
        dK=None,
        dV=None,
        stride_qb=0,
        stride_qh=0,
        stride_qm=0,
        stride_kb=0,
        stride_kh=0,
        stride_kn=0,
        stride_vb=0,
        stride_vh=0,
        stride_vn=0,
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
        HAS_CU_SEQLENS_Q: gl.constexpr = False,
        HAS_CU_SEQLENS_K: gl.constexpr = False,
    ):
        # Initialize base pointers
        q_base = offset_batch_Q(
            base_ptr=Q + config.head_idx * stride_qh,
            batch_idx=config.batch_idx,
            offset=config.offset_q,
            padded_offset=config.padded_offset_q,
            stride_batch=stride_qb,
            stride_seq=stride_qm,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        k_base = offset_batch_K(
            base_ptr=K + config.head_kv_idx * stride_kh,
            batch_idx=config.batch_idx,
            offset=config.offset_k,
            padded_offset=config.padded_offset_k,
            stride_batch=stride_kb,
            stride_seq=stride_kn,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        v_base = offset_batch_K(
            base_ptr=V + config.head_kv_idx * stride_vh,
            batch_idx=config.batch_idx,
            offset=config.offset_k,
            padded_offset=config.padded_offset_k,
            stride_batch=stride_vb,
            stride_seq=stride_vn,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        do_base = offset_batch_Q(
            base_ptr=dO + config.head_idx * stride_doh,
            batch_idx=config.batch_idx,
            offset=config.offset_q,
            padded_offset=config.padded_offset_q,
            stride_batch=stride_dob,
            stride_seq=stride_dom,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
            USE_PADDED=False,
        )
        lse_base = offset_batch_Q(
            base_ptr=LSELog2 + config.head_idx * stride_lh,
            batch_idx=config.batch_idx,
            offset=config.offset_q,
            padded_offset=config.padded_offset_q,
            stride_batch=stride_lb,
            stride_seq=1,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
            USE_PADDED=True,
        )
        dpsum_base = offset_batch_Q(
            base_ptr=dPsum + config.head_idx * stride_ph,
            batch_idx=config.batch_idx,
            offset=config.offset_q,
            padded_offset=config.padded_offset_q,
            stride_batch=stride_pb,
            stride_seq=1,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
            USE_PADDED=True,
        )
        dq_accum_base = offset_batch_Q(
            base_ptr=dQaccum + config.head_idx * stride_dqah,
            batch_idx=config.batch_idx,
            offset=config.offset_q,
            padded_offset=config.padded_offset_q,
            stride_batch=stride_dqab,
            stride_seq=stride_dqam,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_Q,
            USE_PADDED=True,
        )
        dk_base = offset_batch_K(
            base_ptr=dK + config.head_kv_idx * stride_dkh,
            batch_idx=config.batch_idx,
            offset=config.offset_k,
            padded_offset=config.padded_offset_k,
            stride_batch=stride_dkb,
            stride_seq=stride_dkn,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )
        dv_base = offset_batch_K(
            base_ptr=dV + config.head_kv_idx * stride_dvh,
            batch_idx=config.batch_idx,
            offset=config.offset_k,
            padded_offset=config.padded_offset_k,
            stride_batch=stride_dvb,
            stride_seq=stride_dvn,
            HAS_CU_SEQLENS=HAS_CU_SEQLENS_K,
            USE_PADDED=False,
        )

        # For split QO, offset key and value gradients base pointers by split_idx
        if config.IS_SPLIT_QO:
            dk_base += config.split_idx * stride_dks
            dv_base += config.split_idx * stride_dvs

        return AttnBwdPointerScheduler(
            q_base,
            k_base,
            v_base,
            do_base,
            lse_base,
            dpsum_base,
            dq_accum_base,
            dk_base,
            dv_base,
            gl.to_tensor(stride_qm),
            gl.to_tensor(stride_kn),
            gl.to_tensor(stride_vn),
            gl.to_tensor(stride_dom),
            gl.to_tensor(stride_dqam),
            gl.to_tensor(stride_dkn),
            gl.to_tensor(stride_dvn),
        )

    @gluon.jit
    def make_q_ptrs(self, config: AttnBwdConfig, m_block, offs_m, offs_k):
        return make_ptrs(
            base_ptr=self.q_base,
            mn_block=m_block,
            stride_seq=self.stride_qm,
            offs_mn=offs_m,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )

    @gluon.jit
    def make_k_ptrs(self, config: AttnBwdConfig, offs_n, offs_k):
        return make_ptrs(
            base_ptr=self.k_base,
            mn_block=config.n_block,
            stride_seq=self.stride_kn,
            offs_mn=offs_n,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )

    @gluon.jit
    def make_v_ptrs(self, config: AttnBwdConfig, offs_n, offs_k):
        return make_ptrs(
            base_ptr=self.v_base,
            mn_block=config.n_block,
            stride_seq=self.stride_vn,
            offs_mn=offs_n,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )

    @gluon.jit
    def make_do_ptrs(self, config: AttnBwdConfig, m_block, offs_m, offs_k):
        return make_ptrs(
            base_ptr=self.do_base,
            mn_block=m_block,
            stride_seq=self.stride_dom,
            offs_mn=offs_m,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )

    @gluon.jit
    def make_lse_ptrs(self, config: AttnBwdConfig):
        return self.lse_base

    @gluon.jit
    def make_dpsum_ptrs(self, config: AttnBwdConfig):
        return self.dpsum_base

    @gluon.jit
    def make_dq_accum_ptrs(self, config: AttnBwdConfig, m_block, offs_m, offs_k):
        return make_ptrs(
            base_ptr=self.dq_accum_base,
            mn_block=m_block,
            stride_seq=self.stride_dqam,
            offs_mn=offs_m,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )

    @gluon.jit
    def make_dk_ptrs(self, config: AttnBwdConfig, offs_n, offs_k):
        return make_ptrs(
            base_ptr=self.dk_base,
            mn_block=config.n_block,
            stride_seq=self.stride_dkn,
            offs_mn=offs_n,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )

    @gluon.jit
    def make_dv_ptrs(self, config: AttnBwdConfig, offs_n, offs_k):
        return make_ptrs(
            base_ptr=self.dv_base,
            mn_block=config.n_block,
            stride_seq=self.stride_dvn,
            offs_mn=offs_n,
            offs_k=offs_k,
            TILE_K=config.TILE_K,
            SWAP_AB=False,
        )


@aggregate
class AttnMaskScheduler:
    fixed_block: gl.tensor
    actual_seqlen_q: gl.tensor
    actual_seqlen_k: gl.tensor
    window_size_sink: gl.tensor
    window_size_left: gl.tensor
    window_size_right: gl.tensor
    window_size_near: gl.tensor
    TILE_M: gl.constexpr
    TILE_N: gl.constexpr
    QHEAD_PER_KVHEAD_PACKGQA: gl.constexpr
    SWAP_AB: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        fixed_block,
        actual_seqlen_q,
        actual_seqlen_k,
        window_size_sink,
        window_size_left,
        window_size_right,
        window_size_near,
        TILE_M,
        TILE_N,
        QHEAD_PER_KVHEAD_PACKGQA,
        SWAP_AB,
    ):
        self.fixed_block = fixed_block
        self.actual_seqlen_q = actual_seqlen_q
        self.actual_seqlen_k = actual_seqlen_k
        self.window_size_sink = window_size_sink
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.window_size_near = window_size_near
        self.TILE_M = gl.constexpr(TILE_M)
        self.TILE_N = gl.constexpr(TILE_N)
        self.QHEAD_PER_KVHEAD_PACKGQA = gl.constexpr(QHEAD_PER_KVHEAD_PACKGQA)
        self.SWAP_AB = gl.constexpr(SWAP_AB)

    @staticmethod
    @gluon.jit
    def create(
        config,
        SWAP_AB: gl.constexpr = False,
    ):
        if not SWAP_AB:
            return AttnMaskScheduler(
                config.m_block,
                config.actual_seqlen_q,
                config.actual_seqlen_k,
                config.window_size_sink,
                config.window_size_left,
                config.window_size_right,
                config.window_size_near,
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
                config.window_size_sink,
                config.window_size_left,
                config.window_size_right,
                config.window_size_near,
                config.TILE_M,
                config.TILE_N,
                1,
                True,
            )

    @gluon.jit
    def apply_mask(
        self,
        acc_s,
        iter_block,
        offs_m,
        offs_n,
        MASK_SEQLEN: gl.constexpr = True,
        MASK_CAUSAL: gl.constexpr = False,
        MASK_LOCAL: gl.constexpr = False,
        MASK_SINK: gl.constexpr = False,
    ):
        if not self.SWAP_AB:
            m_block = self.fixed_block
            n_block = iter_block
        else:
            m_block = iter_block
            n_block = self.fixed_block
        return apply_mask(
            acc_s=acc_s,
            m_block=m_block,
            n_block=n_block,
            offs_m=offs_m,
            offs_n=offs_n,
            seqlen_q=self.actual_seqlen_q,
            seqlen_k=self.actual_seqlen_k,
            window_size_sink=self.window_size_sink,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            window_size_near=self.window_size_near,
            MASK_SEQLEN=MASK_SEQLEN,
            MASK_CAUSAL=MASK_CAUSAL,
            MASK_LOCAL=MASK_LOCAL,
            MASK_SINK=MASK_SINK,
            TILE_M=self.TILE_M,
            TILE_N=self.TILE_N,
            QHEAD_PER_KVHEAD_PACKGQA=self.QHEAD_PER_KVHEAD_PACKGQA,
            SWAP_AB=self.SWAP_AB,
        )


@aggregate
class SoftmaxScheduler:
    softmax_scale_log2: gl.tensor
    value_scale: gl.tensor

    @gluon.constexpr_function
    def __init__(self, softmax_scale_log2, value_scale):
        self.softmax_scale_log2 = softmax_scale_log2
        self.value_scale = value_scale

    @staticmethod
    @gluon.jit
    def create(config):
        return SoftmaxScheduler(
            config.softmax_scale_log2,
            config.value_scale,
        )

    @gluon.jit
    def online_softmax(
        self,
        acc_s,
        row_max,
        row_sum,
        CHECK_INF: gl.constexpr = False,
    ):
        return online_softmax(
            acc_s=acc_s,
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=self.softmax_scale_log2,
            CHECK_INF=CHECK_INF,
        )

    @gluon.jit
    def online_sparse_softmax(
        self,
        acc_s,
        row_max,
        row_sum,
        softmax_threshold_log2,
        CHECK_INF: gl.constexpr = False,
    ):
        return online_sparse_softmax(
            acc_s=acc_s,
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=self.softmax_scale_log2,
            softmax_threshold_log2=softmax_threshold_log2,
            CHECK_INF=CHECK_INF,
        )

    @gluon.jit
    def rescale_o(
        self,
        acc_o,
        row_scale,
    ):
        return rescale_o(
            acc_o=acc_o,
            row_scale=row_scale,
        )

    @gluon.jit
    def finalize(
        self,
        row_max,
        row_sum,
        IS_LOG2: gl.constexpr = False,
        CHECK_NAN: gl.constexpr = True,
    ):
        return finalize(
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=self.softmax_scale_log2,
            final_scale=self.value_scale,
            IS_LOG2=IS_LOG2,
            CHECK_NAN=CHECK_NAN,
        )
