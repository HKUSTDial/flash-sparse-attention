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
        batch_idx: gl.tensor,
        head_idx: gl.tensor,
        head_kv_idx: gl.tensor,
        split_idx: gl.tensor,
        m_block: gl.tensor,
        row_offsets: gl.tensor,
        softmax_scale: gl.tensor = 0.0,
        softmax_threshold: gl.tensor = 0.0,
        query_scale: gl.tensor = None,
        key_scale: gl.tensor = None,
        value_scale: gl.tensor = None,
        window_size_sink: gl.tensor = 0,
        window_size_left: gl.tensor = 0,
        window_size_right: gl.tensor = 0,
        window_size_near: gl.tensor = 0,
        head_dim: gl.tensor = 0,
        cu_seqlens_q: gl.tensor = None,
        cu_seqlens_k: gl.tensor = None,
        seqused_q: gl.tensor = None,
        seqused_k: gl.tensor = None,
        seqlen_q: gl.tensor = 0,
        seqlen_k: gl.tensor = 0,
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
    ) -> "AttnFwdConfig":
        """
        Build the attention forward configuration for the current program.

        :param batch_idx: current batch index
        :type batch_idx: tensor
        :param head_idx: current query/output head index
        :type head_idx: tensor
        :param head_kv_idx: current key/value head index
        :type head_kv_idx: tensor
        :param split_idx: current KV split index
        :type split_idx: tensor
        :param m_block: current block index along the M dimension
        :type m_block: tensor
        :param row_offsets: row offsets within the M dimension
        :type row_offsets: tensor
        :param softmax_scale: scale applied to attention scores
        :type softmax_scale: tensor
        :param softmax_threshold: dimensionless multiple of uniform attention
        :type softmax_threshold: tensor
        :param query_scale: pointer to the query quantization scale
        :type query_scale: tensor
        :param key_scale: pointer to the key quantization scale
        :type key_scale: tensor
        :param value_scale: pointer to the value quantization scale
        :type value_scale: tensor
        :param window_size_sink: prefix-sink token count
        :type window_size_sink: tensor
        :param window_size_left: distant local band token count
        :type window_size_left: tensor
        :param window_size_right: gap token count after the near-diagonal window
        :type window_size_right: tensor
        :param window_size_near: near-diagonal local token count
        :type window_size_near: tensor
        :param head_dim: attention head dimension
        :type head_dim: tensor
        :param cu_seqlens_q: cumulative query sequence lengths
        :type cu_seqlens_q: tensor
        :param cu_seqlens_k: cumulative key sequence lengths
        :type cu_seqlens_k: tensor
        :param seqused_q: actual query sequence lengths
        :type seqused_q: tensor
        :param seqused_k: actual key sequence lengths
        :type seqused_k: tensor
        :param seqlen_q: static query sequence length
        :type seqlen_q: tensor
        :param seqlen_k: static key sequence length
        :type seqlen_k: tensor
        :param PACK_GQA: boolean flag indicating if packed GQA is enabled
        :type PACK_GQA: bool
        :param QHEAD_PER_KVHEAD_PACKGQA: packed query heads per KV head
        :type QHEAD_PER_KVHEAD_PACKGQA: int
        :param NUM_SPLITS: number of KV splits
        :type NUM_SPLITS: int
        :param TILE_M: tile size along the M dimension
        :type TILE_M: int
        :param TILE_N: tile size along the N dimension
        :type TILE_N: int
        :param TILE_K: tile size along the K dimension
        :type TILE_K: int
        :param IS_CAUSAL: boolean flag indicating if the attention is causal
        :type IS_CAUSAL: bool
        :param IS_LOCAL: boolean flag indicating if local attention is enabled
        :type IS_LOCAL: bool
        :param IS_SPLIT_KV: boolean flag indicating if the KV range is split
        :type IS_SPLIT_KV: bool
        :param IS_QUANT: boolean flag indicating if QKV quantization is enabled
        :type IS_QUANT: bool
        :param HAS_CU_SEQLENS_Q: boolean flag indicating if cu_seqlens_q is provided
        :type HAS_CU_SEQLENS_Q: bool
        :param HAS_CU_SEQLENS_K: boolean flag indicating if cu_seqlens_k is provided
        :type HAS_CU_SEQLENS_K: bool
        :param HAS_SEQUSED_Q: boolean flag indicating if seqused_q is provided
        :type HAS_SEQUSED_Q: bool
        :param HAS_SEQUSED_K: boolean flag indicating if seqused_k is provided
        :type HAS_SEQUSED_K: bool

        :return: attention forward configuration for the current program
        """
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
        batch_idx: gl.tensor,
        head_idx: gl.tensor,
        head_kv_idx: gl.tensor,
        split_idx: gl.tensor,
        n_block: gl.tensor,
        row_offsets: gl.tensor,
        softmax_scale: gl.tensor = 0.0,
        softmax_threshold: gl.tensor = 0.0,
        query_scale: gl.tensor = None,
        key_scale: gl.tensor = None,
        value_scale: gl.tensor = None,
        window_size_sink: gl.tensor = 0,
        window_size_left: gl.tensor = 0,
        window_size_right: gl.tensor = 0,
        window_size_near: gl.tensor = 0,
        head_dim: gl.tensor = 0,
        cu_seqlens_q: gl.tensor = None,
        cu_seqlens_k: gl.tensor = None,
        seqused_q: gl.tensor = None,
        seqused_k: gl.tensor = None,
        seqlen_q: gl.tensor = 0,
        seqlen_k: gl.tensor = 0,
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
    ) -> "AttnBwdConfig":
        """
        Build the attention backward configuration for the current program.

        :param batch_idx: current batch index
        :type batch_idx: tensor
        :param head_idx: current query/output head index
        :type head_idx: tensor
        :param head_kv_idx: current key/value head index
        :type head_kv_idx: tensor
        :param split_idx: current QO split index
        :type split_idx: tensor
        :param n_block: current block index along the N dimension
        :type n_block: tensor
        :param row_offsets: row offsets within the M dimension
        :type row_offsets: tensor
        :param softmax_scale: scale applied to attention scores
        :type softmax_scale: tensor
        :param softmax_threshold: dimensionless multiple of uniform attention
        :type softmax_threshold: tensor
        :param query_scale: pointer to the query quantization scale
        :type query_scale: tensor
        :param key_scale: pointer to the key quantization scale
        :type key_scale: tensor
        :param value_scale: pointer to the value quantization scale
        :type value_scale: tensor
        :param window_size_sink: prefix-sink token count
        :type window_size_sink: tensor
        :param window_size_left: distant local band token count
        :type window_size_left: tensor
        :param window_size_right: gap token count after the near-diagonal window
        :type window_size_right: tensor
        :param window_size_near: near-diagonal local token count
        :type window_size_near: tensor
        :param head_dim: attention head dimension
        :type head_dim: tensor
        :param cu_seqlens_q: cumulative query sequence lengths
        :type cu_seqlens_q: tensor
        :param cu_seqlens_k: cumulative key sequence lengths
        :type cu_seqlens_k: tensor
        :param seqused_q: actual query sequence lengths
        :type seqused_q: tensor
        :param seqused_k: actual key sequence lengths
        :type seqused_k: tensor
        :param seqlen_q: static query sequence length
        :type seqlen_q: tensor
        :param seqlen_k: static key sequence length
        :type seqlen_k: tensor
        :param QHEAD_PER_KVHEAD: ratio of query heads to key/value heads
        :type QHEAD_PER_KVHEAD: int
        :param NUM_SPLITS: number of QO splits
        :type NUM_SPLITS: int
        :param TILE_M: tile size along the M dimension
        :type TILE_M: int
        :param TILE_N: tile size along the N dimension
        :type TILE_N: int
        :param TILE_K: tile size along the K dimension
        :type TILE_K: int
        :param IS_CAUSAL: boolean flag indicating if the attention is causal
        :type IS_CAUSAL: bool
        :param IS_LOCAL: boolean flag indicating if local attention is enabled
        :type IS_LOCAL: bool
        :param IS_SPLIT_QO: boolean flag indicating if the QO range is split
        :type IS_SPLIT_QO: bool
        :param IS_QUANT: boolean flag indicating if QKV quantization is enabled
        :type IS_QUANT: bool
        :param HAS_CU_SEQLENS_Q: boolean flag indicating if cu_seqlens_q is provided
        :type HAS_CU_SEQLENS_Q: bool
        :param HAS_CU_SEQLENS_K: boolean flag indicating if cu_seqlens_k is provided
        :type HAS_CU_SEQLENS_K: bool
        :param HAS_SEQUSED_Q: boolean flag indicating if seqused_q is provided
        :type HAS_SEQUSED_Q: bool
        :param HAS_SEQUSED_K: boolean flag indicating if seqused_k is provided
        :type HAS_SEQUSED_K: bool

        :return: attention backward configuration for the current program
        """
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
    def get_softmax_threshold_log2(self, m_block: gl.tensor) -> gl.tensor:
        """
        Compute the log2-domain softmax threshold for an given block.

        :param m_block: current block index along the M dimension
        :type m_block: tensor

        :return: softmax threshold in log2-domain for the given block
        """
        return get_softmax_threshold(
            softmax_threshold=self.softmax_threshold,
            m_block=m_block,
            seqlen_q=self.actual_seqlen_q,
            seqlen_k=self.actual_seqlen_k,
            row_offsets=self.row_offsets,
            IS_CAUSAL=self.IS_CAUSAL,
            QHEAD_PER_KVHEAD_PACKGQA=1,
        )


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
    def is_empty(self) -> gl.tensor:
        """
        Check whether all forward key-block ranges are empty.

        :return: boolean tensor indicating whether there are no N blocks to process
        """
        return (
            (self.n_block_max <= self.n_block_min)
            & (self.n_block_window_max <= self.n_block_window_min)
            & (self.n_block_sink_max <= self.n_block_sink_min)
        )

    @staticmethod
    @gluon.jit
    def create(config: AttnFwdConfig) -> "AttnFwdBlockScheduler":
        """
        Compute the forward key-block ranges for the current query block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig

        :return: forward block scheduler containing diagonal, window, and sink ranges
        """
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
    def is_empty(self) -> gl.tensor:
        """
        Check whether all backward query-block ranges are empty.

        :return: boolean tensor indicating whether there are no M blocks to process
        """
        return (
            (self.m_block_max <= self.m_block_min)
            & (self.m_block_window_max <= self.m_block_window_min)
            & (self.m_block_sink_max <= self.m_block_sink_min)
        )

    @staticmethod
    @gluon.jit
    def create(config: AttnBwdConfig) -> "AttnBwdBlockScheduler":
        """
        Compute the backward query-block ranges for the current key block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: backward block scheduler containing diagonal, window, and sink ranges
        """
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
        Q: gl.tensor = None,
        K: gl.tensor = None,
        V: gl.tensor = None,
        Out: gl.tensor = None,
        Lse: gl.tensor = None,
        stride_qb: gl.tensor = 0,
        stride_qh: gl.tensor = 0,
        stride_qm: gl.tensor = 0,
        stride_kb: gl.tensor = 0,
        stride_kh: gl.tensor = 0,
        stride_kn: gl.tensor = 0,
        stride_vb: gl.tensor = 0,
        stride_vh: gl.tensor = 0,
        stride_vn: gl.tensor = 0,
        stride_ob: gl.tensor = 0,
        stride_oh: gl.tensor = 0,
        stride_om: gl.tensor = 0,
        stride_os: gl.tensor = 0,
        stride_lb: gl.tensor = 0,
        stride_lh: gl.tensor = 0,
        stride_ls: gl.tensor = 0,
        HAS_CU_SEQLENS_Q: gl.constexpr = False,
        HAS_CU_SEQLENS_K: gl.constexpr = False,
    ) -> "AttnFwdPointerScheduler":
        """
        Build forward base pointers.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param Q: query tensor base pointer
        :type Q: tensor
        :param K: key tensor base pointer
        :type K: tensor
        :param V: value tensor base pointer
        :type V: tensor
        :param Out: output tensor base pointer
        :type Out: tensor
        :param Lse: logsumexp tensor base pointer
        :type Lse: tensor
        :param stride_qb: query batch stride
        :type stride_qb: tensor
        :param stride_qh: query head stride
        :type stride_qh: tensor
        :param stride_qm: query sequence stride
        :type stride_qm: tensor
        :param stride_kb: key batch stride
        :type stride_kb: tensor
        :param stride_kh: key head stride
        :type stride_kh: tensor
        :param stride_kn: key sequence stride
        :type stride_kn: tensor
        :param stride_vb: value batch stride
        :type stride_vb: tensor
        :param stride_vh: value head stride
        :type stride_vh: tensor
        :param stride_vn: value sequence stride
        :type stride_vn: tensor
        :param stride_ob: output batch stride
        :type stride_ob: tensor
        :param stride_oh: output head stride
        :type stride_oh: tensor
        :param stride_om: output sequence stride
        :type stride_om: tensor
        :param stride_os: output split stride
        :type stride_os: tensor
        :param stride_lb: logsumexp batch stride
        :type stride_lb: tensor
        :param stride_lh: logsumexp head stride
        :type stride_lh: tensor
        :param stride_ls: logsumexp split stride
        :type stride_ls: tensor
        :param HAS_CU_SEQLENS_Q: boolean flag indicating if packed query sequences are used
        :type HAS_CU_SEQLENS_Q: bool
        :param HAS_CU_SEQLENS_K: boolean flag indicating if packed key sequences are used
        :type HAS_CU_SEQLENS_K: bool

        :return: forward pointer scheduler for the selected batch, head, and split
        """
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
    def make_q_ptrs(
        self,
        config: AttnFwdConfig,
        offs_m: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct query pointers for the current M block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param offs_m: offsets within the M dimension
        :type offs_m: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: query pointers of shape [TILE_M, TILE_K]
        """
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
    def make_k_ptrs(
        self,
        config: AttnFwdConfig,
        n_block: gl.tensor,
        offs_n: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct key pointers for an N block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param n_block: block index along the N dimension
        :type n_block: tensor
        :param offs_n: offsets within the N dimension
        :type offs_n: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: key pointers of shape [TILE_N, TILE_K]
        """
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
    def make_v_ptrs(
        self,
        config: AttnFwdConfig,
        n_block: gl.tensor,
        offs_n: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct value pointers for an N block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param n_block: block index along the N dimension
        :type n_block: tensor
        :param offs_n: offsets within the N dimension
        :type offs_n: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: value pointers of shape [TILE_N, TILE_K]
        """
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
    def make_out_ptrs(
        self,
        config: AttnFwdConfig,
        offs_m: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct output pointers for the current M block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param offs_m: offsets within the M dimension
        :type offs_m: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: output pointers of shape [TILE_M, TILE_K]
        """
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
    def make_lse_ptrs(
        self,
        config: AttnFwdConfig,
        offs_m: gl.tensor,
    ) -> gl.tensor:
        """
        Construct logsumexp pointers for the current M block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param offs_m: offsets within the M dimension
        :type offs_m: tensor

        :return: logsumexp pointers of shape [TILE_M]
        """
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
        Q: gl.tensor = None,
        K: gl.tensor = None,
        V: gl.tensor = None,
        dO: gl.tensor = None,
        LSELog2: gl.tensor = None,
        dPsum: gl.tensor = None,
        dQaccum: gl.tensor = None,
        dK: gl.tensor = None,
        dV: gl.tensor = None,
        stride_qb: gl.tensor = 0,
        stride_qh: gl.tensor = 0,
        stride_qm: gl.tensor = 0,
        stride_kb: gl.tensor = 0,
        stride_kh: gl.tensor = 0,
        stride_kn: gl.tensor = 0,
        stride_vb: gl.tensor = 0,
        stride_vh: gl.tensor = 0,
        stride_vn: gl.tensor = 0,
        stride_dob: gl.tensor = 0,
        stride_doh: gl.tensor = 0,
        stride_dom: gl.tensor = 0,
        stride_lb: gl.tensor = 0,
        stride_lh: gl.tensor = 0,
        stride_pb: gl.tensor = 0,
        stride_ph: gl.tensor = 0,
        stride_dqab: gl.tensor = 0,
        stride_dqah: gl.tensor = 0,
        stride_dqam: gl.tensor = 0,
        stride_dkb: gl.tensor = 0,
        stride_dkh: gl.tensor = 0,
        stride_dkn: gl.tensor = 0,
        stride_dks: gl.tensor = 0,
        stride_dvb: gl.tensor = 0,
        stride_dvh: gl.tensor = 0,
        stride_dvn: gl.tensor = 0,
        stride_dvs: gl.tensor = 0,
        HAS_CU_SEQLENS_Q: gl.constexpr = False,
        HAS_CU_SEQLENS_K: gl.constexpr = False,
    ) -> "AttnBwdPointerScheduler":
        """
        Build backward base pointers.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param Q: query tensor base pointer
        :type Q: tensor
        :param K: key tensor base pointer
        :type K: tensor
        :param V: value tensor base pointer
        :type V: tensor
        :param dO: output gradient tensor base pointer
        :type dO: tensor
        :param LSELog2: log2-domain logsumexp tensor base pointer
        :type LSELog2: tensor
        :param dPsum: softmax gradient sum tensor base pointer
        :type dPsum: tensor
        :param dQaccum: accumulated query gradient tensor base pointer
        :type dQaccum: tensor
        :param dK: key gradient tensor base pointer
        :type dK: tensor
        :param dV: value gradient tensor base pointer
        :type dV: tensor
        :param stride_qb: query batch stride
        :type stride_qb: tensor
        :param stride_qh: query head stride
        :type stride_qh: tensor
        :param stride_qm: query sequence stride
        :type stride_qm: tensor
        :param stride_kb: key batch stride
        :type stride_kb: tensor
        :param stride_kh: key head stride
        :type stride_kh: tensor
        :param stride_kn: key sequence stride
        :type stride_kn: tensor
        :param stride_vb: value batch stride
        :type stride_vb: tensor
        :param stride_vh: value head stride
        :type stride_vh: tensor
        :param stride_vn: value sequence stride
        :type stride_vn: tensor
        :param stride_dob: output gradient batch stride
        :type stride_dob: tensor
        :param stride_doh: output gradient head stride
        :type stride_doh: tensor
        :param stride_dom: output gradient sequence stride
        :type stride_dom: tensor
        :param stride_lb: logsumexp batch stride
        :type stride_lb: tensor
        :param stride_lh: logsumexp head stride
        :type stride_lh: tensor
        :param stride_pb: softmax gradient sum batch stride
        :type stride_pb: tensor
        :param stride_ph: softmax gradient sum head stride
        :type stride_ph: tensor
        :param stride_dqab: accumulated query gradient batch stride
        :type stride_dqab: tensor
        :param stride_dqah: accumulated query gradient head stride
        :type stride_dqah: tensor
        :param stride_dqam: accumulated query gradient sequence stride
        :type stride_dqam: tensor
        :param stride_dkb: key gradient batch stride
        :type stride_dkb: tensor
        :param stride_dkh: key gradient head stride
        :type stride_dkh: tensor
        :param stride_dkn: key gradient sequence stride
        :type stride_dkn: tensor
        :param stride_dks: key gradient split stride
        :type stride_dks: tensor
        :param stride_dvb: value gradient batch stride
        :type stride_dvb: tensor
        :param stride_dvh: value gradient head stride
        :type stride_dvh: tensor
        :param stride_dvn: value gradient sequence stride
        :type stride_dvn: tensor
        :param stride_dvs: value gradient split stride
        :type stride_dvs: tensor
        :param HAS_CU_SEQLENS_Q: boolean flag indicating if packed query sequences are used
        :type HAS_CU_SEQLENS_Q: bool
        :param HAS_CU_SEQLENS_K: boolean flag indicating if packed key sequences are used
        :type HAS_CU_SEQLENS_K: bool

        :return: backward pointer scheduler for the selected batch, head, and split
        """
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
    def make_q_ptrs(
        self,
        config: AttnBwdConfig,
        m_block: gl.tensor,
        offs_m: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct query pointers for an M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param m_block: block index along the M dimension
        :type m_block: tensor
        :param offs_m: offsets within the M dimension
        :type offs_m: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: query pointers of shape [TILE_M, TILE_K]
        """
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
    def make_k_ptrs(
        self,
        config: AttnBwdConfig,
        offs_n: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct key pointers for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param offs_n: offsets within the N dimension
        :type offs_n: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: key pointers of shape [TILE_N, TILE_K]
        """
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
    def make_v_ptrs(
        self,
        config: AttnBwdConfig,
        offs_n: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct value pointers for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param offs_n: offsets within the N dimension
        :type offs_n: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: value pointers of shape [TILE_N, TILE_K]
        """
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
    def make_do_ptrs(
        self,
        config: AttnBwdConfig,
        m_block: gl.tensor,
        offs_m: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct output gradient pointers for an M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param m_block: block index along the M dimension
        :type m_block: tensor
        :param offs_m: offsets within the M dimension
        :type offs_m: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: output gradient pointers of shape [TILE_M, TILE_K]
        """
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
    def make_lse_ptrs(self, config: AttnBwdConfig) -> gl.tensor:
        """
        Construct logsumexp pointers for an M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: logsumexp pointers of shape [TILE_M]
        """
        return self.lse_base

    @gluon.jit
    def make_dpsum_ptrs(self, config: AttnBwdConfig) -> gl.tensor:
        """
        Construct softmax gradient sum pointers for an M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: softmax gradient sum pointers of shape [TILE_M]
        """
        return self.dpsum_base

    @gluon.jit
    def make_dq_accum_ptrs(
        self,
        config: AttnBwdConfig,
        m_block: gl.tensor,
        offs_m: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct accumulated query gradient pointers for an M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param m_block: block index along the M dimension
        :type m_block: tensor
        :param offs_m: offsets within the M dimension
        :type offs_m: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: accumulated query gradient pointers of shape [TILE_M, TILE_K]
        """
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
    def make_dk_ptrs(
        self,
        config: AttnBwdConfig,
        offs_n: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct key gradient pointers for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param offs_n: offsets within the N dimension
        :type offs_n: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: key gradient pointers of shape [TILE_N, TILE_K]
        """
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
    def make_dv_ptrs(
        self,
        config: AttnBwdConfig,
        offs_n: gl.tensor,
        offs_k: gl.tensor,
    ) -> gl.tensor:
        """
        Construct value gradient pointers for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param offs_n: offsets within the N dimension
        :type offs_n: tensor
        :param offs_k: offsets within the K dimension
        :type offs_k: tensor

        :return: value gradient pointers of shape [TILE_N, TILE_K]
        """
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
