import triton
import triton.language as tl
from triton.language.core import _aggregate as aggregate

from flash_sparse_attn.ops.triton import (
    seqlen_info,
    block_info,
    mask,
    softmax,
)


@aggregate
class AttnFwdGridIndex:
    m_block: tl.tensor
    batch_idx: tl.tensor
    head_idx: tl.tensor
    head_kv_idx: tl.tensor
    split_idx: tl.tensor

    @triton.constexpr_function
    def __init__(self, m_block, batch_idx, head_idx, head_kv_idx, split_idx):
        self.m_block = m_block
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx

    @staticmethod
    @triton.jit
    def create(
        seqlen_k: tl.tensor = None,
        head_dim: tl.tensor = None,
        NUM_SPLITS: tl.constexpr = None,
        QHEAD_PER_KVHEAD: tl.constexpr = None,
        PACK_GQA: tl.constexpr = False,
        IS_SPLIT_KV: tl.constexpr = False,
        IS_LPT: tl.constexpr = False,
    ) -> "AttnFwdGridIndex":
        """
        Decode the forward program IDs into attention grid indices.

        :param seqlen_k: sequence length of the key
        :type seqlen_k: tensor
        :param head_dim: attention head dimension
        :type head_dim: tensor
        :param NUM_SPLITS: number of KV splits
        :type NUM_SPLITS: int
        :param QHEAD_PER_KVHEAD: ratio of query heads to key/value heads
        :type QHEAD_PER_KVHEAD: int
        :param PACK_GQA: boolean flag indicating if packed GQA is enabled
        :type PACK_GQA: bool
        :param IS_SPLIT_KV: boolean flag indicating if the KV range is split
        :type IS_SPLIT_KV: bool
        :param IS_LPT: reverse M blocks within head groups
        :type IS_LPT: bool

        :return: forward grid indices for the current program
        """
        if IS_LPT:
            tl.static_assert(
                not PACK_GQA and not IS_SPLIT_KV,
                "LPT does not support PackGQA or split KV",
            )
            tl.static_assert(
                seqlen_k is not None and head_dim is not None,
                "LPT requires seqlen_k and head_dim",
            )
            num_m_blocks = tl.num_programs(0)
            num_heads = tl.num_programs(1)
            num_batches = tl.num_programs(2)
            tile_idx = tl.program_id(0) + num_m_blocks * (
                tl.program_id(1) + num_heads * tl.program_id(2)
            )
            capacity = tl.maximum((50 * 1024 * 1024) // (seqlen_k * head_dim * 4), 1)
            log2_capacity = tl.inline_asm_elementwise(
                "bfind.u32 $0, $1;",
                constraints="=r,r",
                args=[capacity],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            swizzle = 1 << log2_capacity
            group_idx = tile_idx // (swizzle * num_m_blocks)
            within_group = tile_idx % (swizzle * num_m_blocks)
            group_heads = tl.where(
                group_idx < num_heads * num_batches // swizzle,
                swizzle,
                tl.maximum(num_heads * num_batches % swizzle, 1),
            )
            head_batch_idx = group_idx * swizzle + within_group % group_heads
            return AttnFwdGridIndex(
                num_m_blocks - 1 - within_group // group_heads,
                head_batch_idx // num_heads,
                head_batch_idx % num_heads,
                (head_batch_idx % num_heads) // QHEAD_PER_KVHEAD,
                tl.core.to_tensor(0),
            )
        else:
            m_block = tl.program_id(0)
            head_idx = tl.program_id(1)
            batch_split_idx = tl.program_id(2)
            if IS_SPLIT_KV:
                batch_idx = batch_split_idx // NUM_SPLITS
                split_idx = batch_split_idx - batch_idx * NUM_SPLITS
            else:
                batch_idx = batch_split_idx
                split_idx = tl.core.to_tensor(0)
            head_kv_idx = head_idx if PACK_GQA else head_idx // QHEAD_PER_KVHEAD
            return AttnFwdGridIndex(
                m_block,
                batch_idx,
                head_idx,
                head_kv_idx,
                split_idx,
            )

    @triton.jit
    def load_window_sizes(
        self,
        window_sizes: tl.tensor,
        stride_wh: tl.tensor,
        IS_LOCAL: tl.constexpr,
    ) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
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
            window_size_sink = tl.load(window_sizes + self.head_kv_idx * stride_wh)
            window_size_left = tl.load(window_sizes + self.head_kv_idx * stride_wh + 1)
            window_size_right = tl.load(window_sizes + self.head_kv_idx * stride_wh + 2)
            window_size_near = tl.load(window_sizes + self.head_kv_idx * stride_wh + 3)
        else:
            window_size_sink = tl.core.to_tensor(0)
            window_size_left = tl.core.to_tensor(0)
            window_size_right = tl.core.to_tensor(0)
            window_size_near = tl.core.to_tensor(0)
        return window_size_sink, window_size_left, window_size_right, window_size_near


@aggregate
class AttnBwdGridIndex:
    n_block: tl.tensor
    batch_idx: tl.tensor
    head_idx: tl.tensor
    head_kv_idx: tl.tensor
    split_idx: tl.tensor

    @triton.constexpr_function
    def __init__(self, n_block, batch_idx, head_idx, head_kv_idx, split_idx):
        self.n_block = n_block
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx

    @staticmethod
    @triton.jit
    def create(
        seqlen_q: tl.tensor = None,
        head_dim: tl.tensor = None,
        NUM_SPLITS: tl.constexpr = None,
        QHEAD_PER_KVHEAD: tl.constexpr = None,
        IS_SPLIT_QO: tl.constexpr = False,
        IS_SPT: tl.constexpr = False,
    ) -> "AttnBwdGridIndex":
        """
        Decode the backward program IDs into attention grid indices.

        :param seqlen_q: sequence length of the query
        :type seqlen_q: tensor
        :param head_dim: attention head dimension
        :type head_dim: tensor
        :param NUM_SPLITS: number of QO splits
        :type NUM_SPLITS: int
        :param QHEAD_PER_KVHEAD: ratio of query heads to key/value heads
        :type QHEAD_PER_KVHEAD: int
        :param IS_SPLIT_QO: boolean flag indicating if the QO range is split
        :type IS_SPLIT_QO: bool
        :param IS_SPT: reverse N blocks within head groups
        :type IS_SPT: bool

        :return: backward grid indices for the current program
        """
        if IS_SPT:
            tl.static_assert(
                not IS_SPLIT_QO,
                "SPT does not support split QO",
            )
            tl.static_assert(
                seqlen_q is not None and head_dim is not None,
                "SPT requires seqlen_q and head_dim",
            )
            num_n_blocks = tl.num_programs(0)
            num_heads = tl.num_programs(1)
            num_batches = tl.num_programs(2)
            tile_idx = tl.program_id(0) + num_n_blocks * (
                tl.program_id(1) + num_heads * tl.program_id(2)
            )
            capacity = tl.maximum(((50 * 1024 * 1024) // (seqlen_q * head_dim * 8)), 1)
            log2_capacity = tl.inline_asm_elementwise(
                "bfind.u32 $0, $1;",
                constraints="=r,r",
                args=[capacity],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            swizzle = 1 << log2_capacity
            group_idx = tile_idx // (swizzle * num_n_blocks)
            within_group = tile_idx % (swizzle * num_n_blocks)
            group_heads = tl.where(
                group_idx < num_heads * num_batches // swizzle,
                swizzle,
                tl.maximum(num_heads * num_batches % swizzle, 1),
            )
            head_batch_idx = group_idx * swizzle + within_group % group_heads
            return AttnBwdGridIndex(
                num_n_blocks - 1 - within_group // group_heads,
                head_batch_idx // num_heads,
                head_batch_idx % num_heads,
                (head_batch_idx % num_heads) // QHEAD_PER_KVHEAD,
                tl.core.to_tensor(0),
            )
        else:
            n_block = tl.program_id(0)
            head_idx = tl.program_id(1)
            batch_split_idx = tl.program_id(2)
            if IS_SPLIT_QO:
                batch_idx = batch_split_idx // NUM_SPLITS
                split_idx = batch_split_idx - batch_idx * NUM_SPLITS
            else:
                batch_idx = batch_split_idx
                split_idx = tl.core.to_tensor(0)
            head_kv_idx = head_idx // QHEAD_PER_KVHEAD
            return AttnBwdGridIndex(
                n_block,
                batch_idx,
                head_idx,
                head_kv_idx,
                split_idx,
            )

    @triton.jit
    def load_window_sizes(
        self,
        window_sizes: tl.tensor,
        stride_wh: tl.tensor,
        IS_LOCAL: tl.constexpr,
    ) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
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
            window_size_sink = tl.load(window_sizes + self.head_kv_idx * stride_wh)
            window_size_left = tl.load(window_sizes + self.head_kv_idx * stride_wh + 1)
            window_size_right = tl.load(window_sizes + self.head_kv_idx * stride_wh + 2)
            window_size_near = tl.load(window_sizes + self.head_kv_idx * stride_wh + 3)
        else:
            window_size_sink = tl.core.to_tensor(0)
            window_size_left = tl.core.to_tensor(0)
            window_size_right = tl.core.to_tensor(0)
            window_size_near = tl.core.to_tensor(0)
        return window_size_sink, window_size_left, window_size_right, window_size_near


@aggregate
class AttnFwdConfig:
    batch_idx: tl.tensor
    head_idx: tl.tensor
    head_kv_idx: tl.tensor
    split_idx: tl.tensor
    m_block: tl.tensor
    softmax_scale: tl.tensor
    softmax_scale_log2: tl.tensor
    softmax_threshold: tl.tensor
    value_scale: tl.tensor
    actual_seqlen_q: tl.tensor
    actual_seqlen_k: tl.tensor
    offset_q: tl.tensor
    offset_k: tl.tensor
    padded_offset_q: tl.tensor
    padded_offset_k: tl.tensor
    window_size_sink: tl.tensor
    window_size_left: tl.tensor
    window_size_right: tl.tensor
    window_size_near: tl.tensor
    head_dim: tl.tensor
    PACK_GQA: tl.constexpr
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr
    NUM_SPLITS: tl.constexpr
    TILE_M: tl.constexpr
    TILE_N: tl.constexpr
    TILE_K: tl.constexpr
    IS_CAUSAL: tl.constexpr
    IS_LOCAL: tl.constexpr
    IS_SPARSE: tl.constexpr
    IS_SPLIT_KV: tl.constexpr
    IS_QUANT: tl.constexpr
    HAS_CU_SEQLENS_Q: tl.constexpr
    HAS_CU_SEQLENS_K: tl.constexpr
    HAS_SEQUSED_Q: tl.constexpr
    HAS_SEQUSED_K: tl.constexpr

    @triton.constexpr_function
    def __init__(
        self,
        batch_idx,
        head_idx,
        head_kv_idx,
        split_idx,
        m_block,
        softmax_scale,
        softmax_scale_log2,
        softmax_threshold,
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
        IS_SPARSE,
        IS_SPLIT_KV,
        IS_QUANT,
        HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K,
        HAS_SEQUSED_Q,
        HAS_SEQUSED_K,
    ):
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx
        self.m_block = m_block
        self.softmax_scale = softmax_scale
        self.softmax_scale_log2 = softmax_scale_log2
        self.softmax_threshold = softmax_threshold
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
        self.PACK_GQA = tl.constexpr(PACK_GQA)
        self.QHEAD_PER_KVHEAD_PACKGQA = tl.constexpr(QHEAD_PER_KVHEAD_PACKGQA)
        self.NUM_SPLITS = tl.constexpr(NUM_SPLITS)
        self.TILE_M = tl.constexpr(TILE_M)
        self.TILE_N = tl.constexpr(TILE_N)
        self.TILE_K = tl.constexpr(TILE_K)
        self.IS_CAUSAL = tl.constexpr(IS_CAUSAL)
        self.IS_LOCAL = tl.constexpr(IS_LOCAL)
        self.IS_SPARSE = tl.constexpr(IS_SPARSE)
        self.IS_SPLIT_KV = tl.constexpr(IS_SPLIT_KV)
        self.IS_QUANT = tl.constexpr(IS_QUANT)
        self.HAS_CU_SEQLENS_Q = tl.constexpr(HAS_CU_SEQLENS_Q)
        self.HAS_CU_SEQLENS_K = tl.constexpr(HAS_CU_SEQLENS_K)
        self.HAS_SEQUSED_Q = tl.constexpr(HAS_SEQUSED_Q)
        self.HAS_SEQUSED_K = tl.constexpr(HAS_SEQUSED_K)

    @staticmethod
    @triton.jit
    def create(
        batch_idx: tl.tensor,
        head_idx: tl.tensor,
        head_kv_idx: tl.tensor,
        split_idx: tl.tensor,
        m_block: tl.tensor,
        softmax_scale: tl.tensor = 0.0,
        softmax_threshold: tl.tensor = 0.0,
        query_scale: tl.tensor = None,
        key_scale: tl.tensor = None,
        value_scale: tl.tensor = None,
        window_size_sink: tl.tensor = 0,
        window_size_left: tl.tensor = 0,
        window_size_right: tl.tensor = 0,
        window_size_near: tl.tensor = 0,
        head_dim: tl.tensor = 0,
        cu_seqlens_q: tl.tensor = None,
        cu_seqlens_k: tl.tensor = None,
        seqused_q: tl.tensor = None,
        seqused_k: tl.tensor = None,
        seqlen_q: tl.tensor = 0,
        seqlen_k: tl.tensor = 0,
        PACK_GQA: tl.constexpr = False,
        QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr = 1,
        NUM_SPLITS: tl.constexpr = 1,
        TILE_M: tl.constexpr = 64,
        TILE_N: tl.constexpr = 64,
        TILE_K: tl.constexpr = 64,
        IS_CAUSAL: tl.constexpr = False,
        IS_LOCAL: tl.constexpr = False,
        IS_SPARSE: tl.constexpr = False,
        IS_SPLIT_KV: tl.constexpr = False,
        IS_QUANT: tl.constexpr = False,
        HAS_CU_SEQLENS_Q: tl.constexpr = False,
        HAS_CU_SEQLENS_K: tl.constexpr = False,
        HAS_SEQUSED_Q: tl.constexpr = False,
        HAS_SEQUSED_K: tl.constexpr = False,
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
        :param IS_SPARSE: boolean flag indicating if sparse attention is enabled
        :type IS_SPARSE: bool
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
        if IS_QUANT:
            q_scale = tl.load(query_scale)
            k_scale = tl.load(key_scale)
            v_scale = tl.load(value_scale)
        else:
            q_scale = tl.core.to_tensor(1.0)
            k_scale = tl.core.to_tensor(1.0)
            v_scale = tl.core.to_tensor(1.0)
        LOG2E: tl.constexpr = 1.44269504089
        softmax_scale_log2 = softmax_scale * LOG2E * q_scale * k_scale
        return AttnFwdConfig(
            tl.core.to_tensor(batch_idx),
            tl.core.to_tensor(head_idx),
            tl.core.to_tensor(head_kv_idx),
            tl.core.to_tensor(split_idx),
            tl.core.to_tensor(m_block),
            tl.core.to_tensor(softmax_scale),
            tl.core.to_tensor(softmax_scale_log2),
            tl.core.to_tensor(softmax_threshold),
            tl.core.to_tensor(v_scale),
            tl.core.to_tensor(actual_seqlen_q),
            tl.core.to_tensor(actual_seqlen_k),
            tl.core.to_tensor(offset_q),
            tl.core.to_tensor(offset_k),
            tl.core.to_tensor(padded_offset_q),
            tl.core.to_tensor(padded_offset_k),
            tl.core.to_tensor(window_size_sink),
            tl.core.to_tensor(window_size_left),
            tl.core.to_tensor(window_size_right),
            tl.core.to_tensor(window_size_near),
            tl.core.to_tensor(head_dim),
            PACK_GQA,
            QHEAD_PER_KVHEAD_PACKGQA,
            NUM_SPLITS,
            TILE_M,
            TILE_N,
            TILE_K,
            IS_CAUSAL,
            IS_LOCAL,
            IS_SPARSE,
            IS_SPLIT_KV,
            IS_QUANT,
            HAS_CU_SEQLENS_Q,
            HAS_CU_SEQLENS_K,
            HAS_SEQUSED_Q,
            HAS_SEQUSED_K,
        )

    @triton.jit
    def get_softmax_threshold_log2(self) -> tl.tensor:
        """
        Compute the log2-domain softmax threshold for the current M block.

        :return: softmax threshold in log2-domain for the current M block
        """
        return seqlen_info.get_softmax_threshold(
            softmax_threshold=self.softmax_threshold,
            m_block=self.m_block,
            seqlen_q=self.actual_seqlen_q,
            seqlen_k=self.actual_seqlen_k,
            row_offsets=self.offs_m(),
            IS_CAUSAL=self.IS_CAUSAL,
            QHEAD_PER_KVHEAD_PACKGQA=self.QHEAD_PER_KVHEAD_PACKGQA,
            IS_LOG2=True,
        )

    @triton.jit
    def offs_m(self):
        return tl.arange(0, self.TILE_M)

    @triton.jit
    def offs_n(self):
        return tl.arange(0, self.TILE_N)

    @triton.jit
    def offs_k(self):
        return tl.arange(0, self.TILE_K)


@aggregate
class AttnBwdConfig:
    batch_idx: tl.tensor
    head_idx: tl.tensor
    head_kv_idx: tl.tensor
    split_idx: tl.tensor
    n_block: tl.tensor
    softmax_scale: tl.tensor
    softmax_scale_log2: tl.tensor
    softmax_threshold: tl.tensor
    query_scale: tl.tensor
    key_scale: tl.tensor
    value_scale: tl.tensor
    actual_seqlen_q: tl.tensor
    actual_seqlen_k: tl.tensor
    offset_q: tl.tensor
    offset_k: tl.tensor
    padded_offset_q: tl.tensor
    padded_offset_k: tl.tensor
    window_size_sink: tl.tensor
    window_size_left: tl.tensor
    window_size_right: tl.tensor
    window_size_near: tl.tensor
    head_dim: tl.tensor
    QHEAD_PER_KVHEAD: tl.constexpr
    NUM_SPLITS: tl.constexpr
    TILE_M: tl.constexpr
    TILE_N: tl.constexpr
    TILE_K: tl.constexpr
    IS_CAUSAL: tl.constexpr
    IS_LOCAL: tl.constexpr
    IS_SPARSE: tl.constexpr
    IS_SPLIT_QO: tl.constexpr
    IS_QUANT: tl.constexpr
    HAS_CU_SEQLENS_Q: tl.constexpr
    HAS_CU_SEQLENS_K: tl.constexpr
    HAS_SEQUSED_Q: tl.constexpr
    HAS_SEQUSED_K: tl.constexpr

    @triton.constexpr_function
    def __init__(
        self,
        batch_idx,
        head_idx,
        head_kv_idx,
        split_idx,
        n_block,
        softmax_scale,
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
        IS_SPARSE,
        IS_SPLIT_QO,
        IS_QUANT,
        HAS_CU_SEQLENS_Q,
        HAS_CU_SEQLENS_K,
        HAS_SEQUSED_Q,
        HAS_SEQUSED_K,
    ):
        self.batch_idx = batch_idx
        self.head_idx = head_idx
        self.head_kv_idx = head_kv_idx
        self.split_idx = split_idx
        self.n_block = n_block
        self.softmax_scale = softmax_scale
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
        self.QHEAD_PER_KVHEAD = tl.constexpr(QHEAD_PER_KVHEAD)
        self.NUM_SPLITS = tl.constexpr(NUM_SPLITS)
        self.TILE_M = tl.constexpr(TILE_M)
        self.TILE_N = tl.constexpr(TILE_N)
        self.TILE_K = tl.constexpr(TILE_K)
        self.IS_CAUSAL = tl.constexpr(IS_CAUSAL)
        self.IS_LOCAL = tl.constexpr(IS_LOCAL)
        self.IS_SPARSE = tl.constexpr(IS_SPARSE)
        self.IS_SPLIT_QO = tl.constexpr(IS_SPLIT_QO)
        self.IS_QUANT = tl.constexpr(IS_QUANT)
        self.HAS_CU_SEQLENS_Q = tl.constexpr(HAS_CU_SEQLENS_Q)
        self.HAS_CU_SEQLENS_K = tl.constexpr(HAS_CU_SEQLENS_K)
        self.HAS_SEQUSED_Q = tl.constexpr(HAS_SEQUSED_Q)
        self.HAS_SEQUSED_K = tl.constexpr(HAS_SEQUSED_K)

    @staticmethod
    @triton.jit
    def create(
        batch_idx: tl.tensor,
        head_idx: tl.tensor,
        head_kv_idx: tl.tensor,
        split_idx: tl.tensor,
        n_block: tl.tensor,
        softmax_scale: tl.tensor = 0.0,
        softmax_threshold: tl.tensor = 0.0,
        query_scale: tl.tensor = None,
        key_scale: tl.tensor = None,
        value_scale: tl.tensor = None,
        window_size_sink: tl.tensor = 0,
        window_size_left: tl.tensor = 0,
        window_size_right: tl.tensor = 0,
        window_size_near: tl.tensor = 0,
        head_dim: tl.tensor = 0,
        cu_seqlens_q: tl.tensor = None,
        cu_seqlens_k: tl.tensor = None,
        seqused_q: tl.tensor = None,
        seqused_k: tl.tensor = None,
        seqlen_q: tl.tensor = 0,
        seqlen_k: tl.tensor = 0,
        QHEAD_PER_KVHEAD: tl.constexpr = 1,
        NUM_SPLITS: tl.constexpr = 1,
        TILE_M: tl.constexpr = 64,
        TILE_N: tl.constexpr = 64,
        TILE_K: tl.constexpr = 64,
        IS_CAUSAL: tl.constexpr = False,
        IS_LOCAL: tl.constexpr = False,
        IS_SPARSE: tl.constexpr = False,
        IS_SPLIT_QO: tl.constexpr = False,
        IS_QUANT: tl.constexpr = False,
        HAS_CU_SEQLENS_Q: tl.constexpr = False,
        HAS_CU_SEQLENS_K: tl.constexpr = False,
        HAS_SEQUSED_Q: tl.constexpr = False,
        HAS_SEQUSED_K: tl.constexpr = False,
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
        :param IS_SPARSE: boolean flag indicating if sparse attention is enabled
        :type IS_SPARSE: bool
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
        if IS_QUANT:
            q_scale = tl.load(query_scale)
            k_scale = tl.load(key_scale)
            v_scale = tl.load(value_scale)
        else:
            q_scale = tl.core.to_tensor(1.0)
            k_scale = tl.core.to_tensor(1.0)
            v_scale = tl.core.to_tensor(1.0)
        LOG2E: tl.constexpr = 1.44269504089
        softmax_scale_log2 = softmax_scale * LOG2E
        return AttnBwdConfig(
            tl.core.to_tensor(batch_idx),
            tl.core.to_tensor(head_idx),
            tl.core.to_tensor(head_kv_idx),
            tl.core.to_tensor(split_idx),
            tl.core.to_tensor(n_block),
            tl.core.to_tensor(softmax_scale),
            tl.core.to_tensor(softmax_scale_log2),
            tl.core.to_tensor(softmax_threshold),
            tl.core.to_tensor(q_scale),
            tl.core.to_tensor(k_scale),
            tl.core.to_tensor(v_scale),
            tl.core.to_tensor(actual_seqlen_q),
            tl.core.to_tensor(actual_seqlen_k),
            tl.core.to_tensor(offset_q),
            tl.core.to_tensor(offset_k),
            tl.core.to_tensor(padded_offset_q),
            tl.core.to_tensor(padded_offset_k),
            tl.core.to_tensor(window_size_sink),
            tl.core.to_tensor(window_size_left),
            tl.core.to_tensor(window_size_right),
            tl.core.to_tensor(window_size_near),
            tl.core.to_tensor(head_dim),
            QHEAD_PER_KVHEAD,
            NUM_SPLITS,
            TILE_M,
            TILE_N,
            TILE_K,
            IS_CAUSAL,
            IS_LOCAL,
            IS_SPARSE,
            IS_SPLIT_QO,
            IS_QUANT,
            HAS_CU_SEQLENS_Q,
            HAS_CU_SEQLENS_K,
            HAS_SEQUSED_Q,
            HAS_SEQUSED_K,
        )

    @triton.jit
    def get_softmax_threshold_log2(self, m_block: tl.tensor) -> tl.tensor:
        """
        Compute the log2-domain softmax threshold for an given block.

        :param m_block: current block index along the M dimension
        :type m_block: tensor

        :return: softmax threshold in log2-domain for the given block
        """
        return seqlen_info.get_softmax_threshold(
            softmax_threshold=self.softmax_threshold,
            m_block=m_block,
            seqlen_q=self.actual_seqlen_q,
            seqlen_k=self.actual_seqlen_k,
            row_offsets=self.offs_m(),
            IS_CAUSAL=self.IS_CAUSAL,
            QHEAD_PER_KVHEAD_PACKGQA=1,
            IS_LOG2=True,
        )

    @triton.jit
    def offs_m(self):
        return tl.arange(0, self.TILE_M)

    @triton.jit
    def offs_n(self):
        return tl.arange(0, self.TILE_N)

    @triton.jit
    def offs_k(self):
        return tl.arange(0, self.TILE_K)


@aggregate
class AttnFwdBlockScheduler:
    n_block_min: tl.tensor
    n_block_max: tl.tensor
    n_block_max_no_mask: tl.tensor
    n_block_window_min: tl.tensor
    n_block_window_max: tl.tensor
    n_block_window_min_no_mask: tl.tensor
    n_block_window_max_no_mask: tl.tensor
    n_block_sink_min: tl.tensor
    n_block_sink_max: tl.tensor

    @triton.constexpr_function
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

    @staticmethod
    @triton.jit
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
        ) = block_info.get_n_block_min_max(
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
        n_block_max_no_mask = block_info.get_n_block_max_no_causal_local_mask(
            seqlen_q=config.actual_seqlen_q,
            seqlen_k=config.actual_seqlen_k,
            m_block=config.m_block,
            n_block_min=n_block_min,
            n_block_max=n_block_max - 1,
            window_size_right=0,
            window_size_near=config.window_size_near,
            TILE_N=config.TILE_N,
            TILE_M=config.TILE_M,
            IS_CAUSAL=config.IS_CAUSAL,
            IS_LOCAL=config.IS_LOCAL,
            QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
        )

        if config.IS_LOCAL:
            # Compute local n_block range for this m_block
            n_block_window_max_no_mask = block_info.get_n_block_max_no_causal_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                m_block=config.m_block,
                n_block_min=n_block_window_min,
                n_block_max=n_block_window_max,
                window_size_right=config.window_size_right,
                window_size_near=config.window_size_near,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_CAUSAL=False,
                IS_LOCAL=True,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
            n_block_window_min_no_mask = block_info.get_n_block_min_before_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                m_block=config.m_block,
                n_block_min=n_block_window_min,
                n_block_max=n_block_window_max_no_mask,
                window_size_left=config.window_size_left,
                window_size_right=config.window_size_right,
                window_size_near=config.window_size_near,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_LOCAL=True,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        else:
            n_block_window_min = tl.core.to_tensor(0)
            n_block_window_max = tl.core.to_tensor(0)
            n_block_window_min_no_mask = tl.core.to_tensor(0)
            n_block_window_max_no_mask = tl.core.to_tensor(0)

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

    @triton.jit
    def is_empty(self) -> tl.tensor:
        """
        Check whether all forward key-block ranges are empty.

        :return: boolean tensor indicating whether there are no N blocks to process
        """
        return (
            (self.n_block_max <= self.n_block_min)
            & (self.n_block_window_max <= self.n_block_window_min)
            & (self.n_block_sink_max <= self.n_block_sink_min)
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
    m_block_sink_min: tl.tensor
    m_block_sink_max: tl.tensor

    @triton.constexpr_function
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

    @staticmethod
    @triton.jit
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
        ) = block_info.get_m_block_min_max(
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
        m_block_min_no_mask = block_info.get_m_block_min_no_causal_local_mask(
            seqlen_q=config.actual_seqlen_q,
            seqlen_k=config.actual_seqlen_k,
            n_block=config.n_block,
            m_block_min=m_block_min,
            m_block_max=m_block_max - 1,
            window_size_right=0,
            window_size_near=config.window_size_near,
            TILE_N=config.TILE_N,
            TILE_M=config.TILE_M,
            IS_CAUSAL=config.IS_CAUSAL or config.IS_LOCAL,
            IS_LOCAL=config.IS_LOCAL,
        )

        if config.IS_LOCAL:
            # Compute local m_block range for this n_block
            m_block_window_min_no_mask = block_info.get_m_block_min_no_causal_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                n_block=config.n_block,
                m_block_min=m_block_window_min,
                m_block_max=m_block_window_max,
                window_size_right=config.window_size_right,
                window_size_near=config.window_size_near,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_CAUSAL=False,
                IS_LOCAL=True,
            )
            m_block_window_max_no_mask = block_info.get_m_block_max_before_local_mask(
                seqlen_q=config.actual_seqlen_q,
                seqlen_k=config.actual_seqlen_k,
                n_block=config.n_block,
                m_block_min=m_block_window_min_no_mask,
                m_block_max=m_block_window_max,
                window_size_left=config.window_size_left,
                window_size_right=config.window_size_right,
                window_size_near=config.window_size_near,
                TILE_N=config.TILE_N,
                TILE_M=config.TILE_M,
                IS_LOCAL=True,
            )
        else:
            m_block_window_min_no_mask = tl.core.to_tensor(0)
            m_block_window_max_no_mask = tl.core.to_tensor(0)
            m_block_sink_min = tl.core.to_tensor(0)
            m_block_sink_max = tl.core.to_tensor(0)

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

    @triton.jit
    def is_empty(self) -> tl.tensor:
        """
        Check whether all backward query-block ranges are empty.

        :return: boolean tensor indicating whether there are no M blocks to process
        """
        return (
            (self.m_block_max <= self.m_block_min)
            & (self.m_block_window_max <= self.m_block_window_min)
            & (self.m_block_sink_max <= self.m_block_sink_min)
        )


@aggregate
class AttnFwdPointerScheduler:
    q_base_ptr: tl.tensor | tl.constexpr
    k_base_ptr: tl.tensor | tl.constexpr
    v_base_ptr: tl.tensor | tl.constexpr
    out_base_ptr: tl.tensor | tl.constexpr
    lse_base_ptr: tl.tensor | tl.constexpr
    stride_qh: tl.tensor
    stride_qm: tl.tensor
    stride_kb: tl.tensor
    stride_kn: tl.tensor
    stride_vb: tl.tensor
    stride_vn: tl.tensor
    stride_oh: tl.tensor
    stride_om: tl.tensor
    stride_lh: tl.tensor

    @triton.constexpr_function
    def __init__(
        self,
        q_base_ptr,
        k_base_ptr,
        v_base_ptr,
        out_base_ptr,
        lse_base_ptr,
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
        self.q_base_ptr = q_base_ptr if q_base_ptr is not None else tl.constexpr(None)
        self.k_base_ptr = k_base_ptr if k_base_ptr is not None else tl.constexpr(None)
        self.v_base_ptr = v_base_ptr if v_base_ptr is not None else tl.constexpr(None)
        self.out_base_ptr = out_base_ptr if out_base_ptr is not None else tl.constexpr(None)
        self.lse_base_ptr = lse_base_ptr if lse_base_ptr is not None else tl.constexpr(None)
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
    @triton.jit
    def create(
        config: AttnFwdConfig,
        Q: tl.tensor = None,
        K: tl.tensor = None,
        V: tl.tensor = None,
        Out: tl.tensor = None,
        Lse: tl.tensor = None,
        stride_qb: tl.tensor = 0,
        stride_qh: tl.tensor = 0,
        stride_qm: tl.tensor = 0,
        stride_kb: tl.tensor = 0,
        stride_kh: tl.tensor = 0,
        stride_kn: tl.tensor = 0,
        stride_vb: tl.tensor = 0,
        stride_vh: tl.tensor = 0,
        stride_vn: tl.tensor = 0,
        stride_ob: tl.tensor = 0,
        stride_oh: tl.tensor = 0,
        stride_om: tl.tensor = 0,
        stride_os: tl.tensor = 0,
        stride_lb: tl.tensor = 0,
        stride_lh: tl.tensor = 0,
        stride_ls: tl.tensor = 0,
    ) -> "AttnFwdPointerScheduler":
        """
        Build forward base pointers.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param Q: Q tensor base pointer
        :type Q: tensor
        :param K: K tensor base pointer
        :type K: tensor
        :param V: V tensor base pointer
        :type V: tensor
        :param Out: O tensor base pointer
        :type Out: tensor
        :param Lse: LSE tensor base pointer
        :type Lse: tensor
        :param stride_qb: Q batch stride
        :type stride_qb: tensor
        :param stride_qh: Q head stride
        :type stride_qh: tensor
        :param stride_qm: Q sequence stride
        :type stride_qm: tensor
        :param stride_kb: K batch stride
        :type stride_kb: tensor
        :param stride_kh: K head stride
        :type stride_kh: tensor
        :param stride_kn: K sequence stride
        :type stride_kn: tensor
        :param stride_vb: V batch stride
        :type stride_vb: tensor
        :param stride_vh: V head stride
        :type stride_vh: tensor
        :param stride_vn: V sequence stride
        :type stride_vn: tensor
        :param stride_ob: O batch stride
        :type stride_ob: tensor
        :param stride_oh: O head stride
        :type stride_oh: tensor
        :param stride_om: O sequence stride
        :type stride_om: tensor
        :param stride_os: O split stride
        :type stride_os: tensor
        :param stride_lb: LSE batch stride
        :type stride_lb: tensor
        :param stride_lh: LSE head stride
        :type stride_lh: tensor
        :param stride_ls: LSE split stride
        :type stride_ls: tensor

        :return: forward pointer scheduler for the selected batch, head, and split
        """
        # Initialize base pointers
        q_base_ptr = (
            seqlen_info.offset_batch_Q(
                base_ptr=Q + config.head_idx * stride_qh if not config.PACK_GQA else Q,
                batch_idx=config.batch_idx,
                offset=config.offset_q,
                padded_offset=config.padded_offset_q,
                stride_batch=stride_qb,
                stride_seq=stride_qm,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_Q,
                USE_PADDED=False,
            )
            if Q is not None
            else None
        )
        k_base_ptr = (
            seqlen_info.offset_batch_K(
                base_ptr=K + config.head_kv_idx * stride_kh,
                batch_idx=config.batch_idx,
                offset=config.offset_k,
                padded_offset=config.padded_offset_k,
                stride_batch=stride_kb,
                stride_seq=stride_kn,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_K,
                USE_PADDED=False,
            )
            if K is not None
            else None
        )
        v_base_ptr = (
            seqlen_info.offset_batch_K(
                base_ptr=V + config.head_kv_idx * stride_vh,
                batch_idx=config.batch_idx,
                offset=config.offset_k,
                padded_offset=config.padded_offset_k,
                stride_batch=stride_vb,
                stride_seq=stride_vn,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_K,
                USE_PADDED=False,
            )
            if V is not None
            else None
        )
        out_base_ptr = (
            seqlen_info.offset_batch_Q(
                base_ptr=(Out + config.head_idx * stride_oh if not config.PACK_GQA else Out),
                batch_idx=config.batch_idx,
                offset=config.offset_q,
                padded_offset=config.padded_offset_q,
                stride_batch=stride_ob,
                stride_seq=stride_om,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_Q,
                USE_PADDED=False,
            )
            if Out is not None
            else None
        )
        lse_base_ptr = (
            seqlen_info.offset_batch_Q(
                base_ptr=(Lse + config.head_idx * stride_lh if not config.PACK_GQA else Lse),
                batch_idx=config.batch_idx,
                offset=config.offset_q,
                padded_offset=config.padded_offset_q,
                stride_batch=stride_lb,
                stride_seq=1,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_Q,
                USE_PADDED=False,
            )
            if Lse is not None
            else None
        )

        # For split KV, offset output and LSE base pointers by split_idx
        if config.IS_SPLIT_KV:
            if Out is not None:
                out_base_ptr += config.split_idx * stride_os
            if Lse is not None:
                lse_base_ptr += config.split_idx * stride_ls

        return AttnFwdPointerScheduler(
            q_base_ptr,
            k_base_ptr,
            v_base_ptr,
            out_base_ptr,
            lse_base_ptr,
            tl.core.to_tensor(stride_qh),
            tl.core.to_tensor(stride_qm),
            tl.core.to_tensor(stride_kb),
            tl.core.to_tensor(stride_kn),
            tl.core.to_tensor(stride_vb),
            tl.core.to_tensor(stride_vn),
            tl.core.to_tensor(stride_oh),
            tl.core.to_tensor(stride_om),
            tl.core.to_tensor(stride_lh),
        )

    @triton.jit
    def make_q_ptrs(
        self,
        config: AttnFwdConfig,
    ) -> tl.tensor:
        """
        Construct query pointers for the current M block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig

        :return: Q pointers of shape [TILE_M, TILE_K]
        """
        if config.PACK_GQA:
            return seqlen_info.make_pack_gqa_ptrs(
                base_ptr=self.q_base_ptr,
                m_block=config.m_block,
                head_idx=config.head_idx,
                stride_head=self.stride_qh,
                stride_seq=self.stride_qm,
                offs_m=config.offs_m(),
                offs_k=config.offs_k(),
                TILE_K=config.TILE_K,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        else:
            return tl.make_tensor_descriptor(
                self.q_base_ptr,
                shape=[config.actual_seqlen_q, config.head_dim],
                strides=[self.stride_qm, 1],
                block_shape=[config.TILE_M, config.TILE_K],
            )

    @triton.jit
    def make_k_ptrs(
        self,
        config: AttnFwdConfig,
    ):
        """
        Construct K pointers for the current N block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig

        :return: K tensor descriptor with block shape [TILE_N, TILE_K]
        """
        return tl.make_tensor_descriptor(
            self.k_base_ptr,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_kn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_v_ptrs(
        self,
        config: AttnFwdConfig,
    ):
        """
        Construct V pointers for the current N block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig

        :return: V tensor descriptor with block shape [TILE_N, TILE_K]
        """
        return tl.make_tensor_descriptor(
            self.v_base_ptr,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_vn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_out_ptrs(
        self,
        config: AttnFwdConfig,
    ):
        """
        Construct O pointers for the current M block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig

        :return: O pointers of shape [TILE_M, TILE_K]
        """
        if config.PACK_GQA:
            return seqlen_info.make_pack_gqa_ptrs(
                base_ptr=self.out_base_ptr,
                m_block=config.m_block,
                head_idx=config.head_idx,
                stride_head=self.stride_oh,
                stride_seq=self.stride_om,
                offs_m=config.offs_m(),
                offs_k=config.offs_k(),
                TILE_K=config.TILE_K,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        else:
            return tl.make_tensor_descriptor(
                self.out_base_ptr,
                shape=[config.actual_seqlen_q, config.head_dim],
                strides=[self.stride_om, 1],
                block_shape=[config.TILE_M, config.TILE_K],
            )

    @triton.jit
    def make_lse_ptrs(self, config: AttnFwdConfig):
        """
        Construct LSE pointers for the current M block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig

        :return: LSE pointers of shape [TILE_M]
        """
        if config.PACK_GQA:
            return seqlen_info.make_pack_gqa_ptrs(
                base_ptr=self.lse_base_ptr,
                m_block=config.m_block,
                head_idx=config.head_idx,
                stride_head=self.stride_lh,
                stride_seq=1,
                offs_m=config.offs_m(),
                offs_k=config.offs_k(),
                TILE_K=1,
                QHEAD_PER_KVHEAD_PACKGQA=config.QHEAD_PER_KVHEAD_PACKGQA,
            )
        else:
            return seqlen_info.make_ptrs(
                base_ptr=self.lse_base_ptr,
                mn_block=config.m_block,
                stride_seq=1,
                offs_mn=config.offs_m(),
                offs_k=config.offs_k(),
                TILE_K=1,
            )

    @triton.jit
    def load_q(self, config: AttnFwdConfig, q_ptrs):
        """
        Load the Q tile for the current M block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param q_ptrs: Q pointer
        :type q_ptrs: tensor

        :return: Q tile of shape [TILE_M, TILE_K]
        """
        if config.PACK_GQA:
            offs_m = config.m_block * config.TILE_M + config.offs_m()
            return tl.load(
                q_ptrs,
                mask=((offs_m // config.QHEAD_PER_KVHEAD_PACKGQA) < config.actual_seqlen_q)[:, None]
                & (config.offs_k() < config.head_dim)[None, :],
                other=0.0,
                cache_modifier=".ca",
            )
        else:
            return q_ptrs.load([config.m_block * config.TILE_M, 0])

    @triton.jit
    def load_k(self, config: AttnFwdConfig, k_ptrs, n_block):
        """
        Load the K tile for the selected N block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param k_ptrs: K pointer
        :type k_ptrs: tensor
        :param n_block: selected N block index
        :type n_block: tensor

        :return: K tile of shape [TILE_N, TILE_K]
        """
        return k_ptrs.load([n_block * config.TILE_N, 0])

    @triton.jit
    def load_v(self, config: AttnFwdConfig, v_ptrs, n_block):
        """
        Load the V tile for the selected N block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param v_ptrs: V pointer
        :type v_ptrs: tensor
        :param n_block: selected N block index
        :type n_block: tensor

        :return: V tile of shape [TILE_N, TILE_K]
        """
        return v_ptrs.load([n_block * config.TILE_N, 0])

    @triton.jit
    def store_out(
        self,
        config: AttnFwdConfig,
        out_ptrs,
        o_tile,
    ):
        """
        Store the O tile for the current M block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param out_ptrs: O pointer
        :type out_ptrs: tensor
        :param o_tile: O tile of shape [TILE_M, TILE_K]
        :type o_tile: tensor
        """
        if not config.IS_SPLIT_KV:
            if config.PACK_GQA:
                o_tile = o_tile.to(out_ptrs.dtype.element_ty)
            else:
                o_tile = o_tile.to(out_ptrs.dtype)
        if config.PACK_GQA:
            offs_m = config.m_block * config.TILE_M + config.offs_m()
            tl.store(
                out_ptrs,
                o_tile,
                mask=((offs_m // config.QHEAD_PER_KVHEAD_PACKGQA) < config.actual_seqlen_q)[:, None]
                & (config.offs_k() < config.head_dim)[None, :],
                cache_modifier=".wb",
            )
        else:
            out_ptrs.store([config.m_block * config.TILE_M, 0], o_tile)

    @triton.jit
    def store_lse(self, config: AttnFwdConfig, lse_ptrs, lse_tile):
        """
        Store the LSE tile for the current M block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param lse_ptrs: LSE pointer
        :type lse_ptrs: tensor
        :param lse_tile: LSE tile of shape [TILE_M]
        :type lse_tile: tensor
        """
        if config.PACK_GQA:
            offs_m = config.m_block * config.TILE_M + config.offs_m()
            tl.store(
                lse_ptrs,
                lse_tile,
                mask=((offs_m // config.QHEAD_PER_KVHEAD_PACKGQA) < config.actual_seqlen_q),
                cache_modifier=".wb",
            )
        else:
            offs_m = config.m_block * config.TILE_M + config.offs_m()
            tl.store(
                lse_ptrs,
                lse_tile,
                mask=offs_m < config.actual_seqlen_q,
                cache_modifier=".wb",
            )

    @triton.jit
    def store_empty(
        self,
        config: AttnFwdConfig,
        out_ptrs,
        lse_ptrs,
    ):
        """
        Store zero O and -inf LSE tiles for an empty M block.

        :param config: attention forward configuration
        :type config: AttnFwdConfig
        :param out_ptrs: O pointer
        :type out_ptrs: tensor
        :param lse_ptrs: LSE pointer
        :type lse_ptrs: tensor
        """
        lse_tile = tl.full((config.TILE_M,), float("-inf"), dtype=tl.float32)
        self.store_lse(config, lse_ptrs, lse_tile)
        o_tile = tl.zeros((config.TILE_M, config.TILE_K), dtype=tl.float32)
        self.store_out(config, out_ptrs, o_tile)


@aggregate
class AttnBwdPointerScheduler:
    q_base_ptr: tl.tensor | tl.constexpr
    k_base_ptr: tl.tensor | tl.constexpr
    v_base_ptr: tl.tensor | tl.constexpr
    lse_base_ptr: tl.tensor | tl.constexpr
    do_base_ptr: tl.tensor | tl.constexpr
    dpsum_base_ptr: tl.tensor | tl.constexpr
    dq_accum_base_ptr: tl.tensor | tl.constexpr
    dk_base_ptr: tl.tensor | tl.constexpr
    dv_base_ptr: tl.tensor | tl.constexpr
    stride_qm: tl.tensor
    stride_kn: tl.tensor
    stride_vn: tl.tensor
    stride_dom: tl.tensor
    stride_dqam: tl.tensor
    stride_dkn: tl.tensor
    stride_dvn: tl.tensor

    @triton.constexpr_function
    def __init__(
        self,
        q_base_ptr,
        k_base_ptr,
        v_base_ptr,
        lse_base_ptr,
        do_base_ptr,
        dpsum_base_ptr,
        dq_accum_base_ptr,
        dk_base_ptr,
        dv_base_ptr,
        stride_qm,
        stride_kn,
        stride_vn,
        stride_dom,
        stride_dqam,
        stride_dkn,
        stride_dvn,
    ):
        self.q_base_ptr = q_base_ptr if q_base_ptr is not None else tl.constexpr(None)
        self.k_base_ptr = k_base_ptr if k_base_ptr is not None else tl.constexpr(None)
        self.v_base_ptr = v_base_ptr if v_base_ptr is not None else tl.constexpr(None)
        self.lse_base_ptr = lse_base_ptr if lse_base_ptr is not None else tl.constexpr(None)
        self.do_base_ptr = do_base_ptr if do_base_ptr is not None else tl.constexpr(None)
        self.dpsum_base_ptr = dpsum_base_ptr if dpsum_base_ptr is not None else tl.constexpr(None)
        self.dq_accum_base_ptr = (
            dq_accum_base_ptr if dq_accum_base_ptr is not None else tl.constexpr(None)
        )
        self.dk_base_ptr = dk_base_ptr if dk_base_ptr is not None else tl.constexpr(None)
        self.dv_base_ptr = dv_base_ptr if dv_base_ptr is not None else tl.constexpr(None)
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
        Q: tl.tensor = None,
        K: tl.tensor = None,
        V: tl.tensor = None,
        LSELog2: tl.tensor = None,
        dO: tl.tensor = None,
        dPsum: tl.tensor = None,
        dQaccum: tl.tensor = None,
        dK: tl.tensor = None,
        dV: tl.tensor = None,
        stride_qb: tl.tensor = 0,
        stride_qh: tl.tensor = 0,
        stride_qm: tl.tensor = 0,
        stride_kb: tl.tensor = 0,
        stride_kh: tl.tensor = 0,
        stride_kn: tl.tensor = 0,
        stride_vb: tl.tensor = 0,
        stride_vh: tl.tensor = 0,
        stride_vn: tl.tensor = 0,
        stride_dob: tl.tensor = 0,
        stride_doh: tl.tensor = 0,
        stride_dom: tl.tensor = 0,
        stride_lb: tl.tensor = 0,
        stride_lh: tl.tensor = 0,
        stride_pb: tl.tensor = 0,
        stride_ph: tl.tensor = 0,
        stride_dqab: tl.tensor = 0,
        stride_dqah: tl.tensor = 0,
        stride_dqam: tl.tensor = 0,
        stride_dkb: tl.tensor = 0,
        stride_dkh: tl.tensor = 0,
        stride_dkn: tl.tensor = 0,
        stride_dks: tl.tensor = 0,
        stride_dvb: tl.tensor = 0,
        stride_dvh: tl.tensor = 0,
        stride_dvn: tl.tensor = 0,
        stride_dvs: tl.tensor = 0,
    ) -> "AttnBwdPointerScheduler":
        """
        Build backward base pointers.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param Q: Q tensor base pointer
        :type Q: tensor
        :param K: K tensor base pointer
        :type K: tensor
        :param V: V tensor base pointer
        :type V: tensor
        :param LSELog2: LSELog2 tensor base pointer
        :type LSELog2: tensor
        :param dO: dO tensor base pointer
        :type dO: tensor
        :param dPsum: dPsum tensor base pointer
        :type dPsum: tensor
        :param dQaccum: dQaccum tensor base pointer
        :type dQaccum: tensor
        :param dK: dK tensor base pointer
        :type dK: tensor
        :param dV: dV tensor base pointer
        :type dV: tensor
        :param stride_qb: Q batch stride
        :type stride_qb: tensor
        :param stride_qh: Q head stride
        :type stride_qh: tensor
        :param stride_qm: Q sequence stride
        :type stride_qm: tensor
        :param stride_kb: K batch stride
        :type stride_kb: tensor
        :param stride_kh: K head stride
        :type stride_kh: tensor
        :param stride_kn: K sequence stride
        :type stride_kn: tensor
        :param stride_vb: V batch stride
        :type stride_vb: tensor
        :param stride_vh: V head stride
        :type stride_vh: tensor
        :param stride_vn: V sequence stride
        :type stride_vn: tensor
        :param stride_dob: dO batch stride
        :type stride_dob: tensor
        :param stride_doh: dO head stride
        :type stride_doh: tensor
        :param stride_dom: dO sequence stride
        :type stride_dom: tensor
        :param stride_lb: LSELog2 batch stride
        :type stride_lb: tensor
        :param stride_lh: LSELog2 head stride
        :type stride_lh: tensor
        :param stride_pb: dPsum batch stride
        :type stride_pb: tensor
        :param stride_ph: dPsum head stride
        :type stride_ph: tensor
        :param stride_dqab: dQaccum batch stride
        :type stride_dqab: tensor
        :param stride_dqah: dQaccum head stride
        :type stride_dqah: tensor
        :param stride_dqam: dQaccum sequence stride
        :type stride_dqam: tensor
        :param stride_dkb: dK batch stride
        :type stride_dkb: tensor
        :param stride_dkh: dK head stride
        :type stride_dkh: tensor
        :param stride_dkn: dK sequence stride
        :type stride_dkn: tensor
        :param stride_dks: dK split stride
        :type stride_dks: tensor
        :param stride_dvb: dV batch stride
        :type stride_dvb: tensor
        :param stride_dvh: dV head stride
        :type stride_dvh: tensor
        :param stride_dvn: dV sequence stride
        :type stride_dvn: tensor
        :param stride_dvs: dV split stride
        :type stride_dvs: tensor

        :return: backward pointer scheduler for the selected batch, head, and split
        """
        # Initialize base pointers
        q_base_ptr = (
            seqlen_info.offset_batch_Q(
                base_ptr=Q + config.head_idx * stride_qh,
                batch_idx=config.batch_idx,
                offset=config.offset_q,
                padded_offset=config.padded_offset_q,
                stride_batch=stride_qb,
                stride_seq=stride_qm,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_Q,
                USE_PADDED=False,
            )
            if Q is not None
            else None
        )
        k_base_ptr = (
            seqlen_info.offset_batch_K(
                base_ptr=K + config.head_kv_idx * stride_kh,
                batch_idx=config.batch_idx,
                offset=config.offset_k,
                padded_offset=config.padded_offset_k,
                stride_batch=stride_kb,
                stride_seq=stride_kn,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_K,
                USE_PADDED=False,
            )
            if K is not None
            else None
        )
        v_base_ptr = (
            seqlen_info.offset_batch_K(
                base_ptr=V + config.head_kv_idx * stride_vh,
                batch_idx=config.batch_idx,
                offset=config.offset_k,
                padded_offset=config.padded_offset_k,
                stride_batch=stride_vb,
                stride_seq=stride_vn,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_K,
                USE_PADDED=False,
            )
            if V is not None
            else None
        )
        lse_base_ptr = (
            seqlen_info.offset_batch_Q(
                base_ptr=LSELog2 + config.head_idx * stride_lh,
                batch_idx=config.batch_idx,
                offset=config.offset_q,
                padded_offset=config.padded_offset_q,
                stride_batch=stride_lb,
                stride_seq=1,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_Q,
                USE_PADDED=True,
            )
            if LSELog2 is not None
            else None
        )
        do_base_ptr = (
            seqlen_info.offset_batch_Q(
                base_ptr=dO + config.head_idx * stride_doh,
                batch_idx=config.batch_idx,
                offset=config.offset_q,
                padded_offset=config.padded_offset_q,
                stride_batch=stride_dob,
                stride_seq=stride_dom,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_Q,
                USE_PADDED=False,
            )
            if dO is not None
            else None
        )
        dpsum_base_ptr = (
            seqlen_info.offset_batch_Q(
                base_ptr=dPsum + config.head_idx * stride_ph,
                batch_idx=config.batch_idx,
                offset=config.offset_q,
                padded_offset=config.padded_offset_q,
                stride_batch=stride_pb,
                stride_seq=1,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_Q,
                USE_PADDED=True,
            )
            if dPsum is not None
            else None
        )
        dq_accum_base_ptr = (
            seqlen_info.offset_batch_Q(
                base_ptr=dQaccum + config.head_idx * stride_dqah,
                batch_idx=config.batch_idx,
                offset=config.offset_q,
                padded_offset=config.padded_offset_q,
                stride_batch=stride_dqab,
                stride_seq=stride_dqam,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_Q,
                USE_PADDED=True,
            )
            if dQaccum is not None
            else None
        )
        dk_base_ptr = (
            seqlen_info.offset_batch_K(
                base_ptr=dK + config.head_kv_idx * stride_dkh,
                batch_idx=config.batch_idx,
                offset=config.offset_k,
                padded_offset=config.padded_offset_k,
                stride_batch=stride_dkb,
                stride_seq=stride_dkn,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_K,
                USE_PADDED=True,
            )
            if dK is not None
            else None
        )
        dv_base_ptr = (
            seqlen_info.offset_batch_K(
                base_ptr=dV + config.head_kv_idx * stride_dvh,
                batch_idx=config.batch_idx,
                offset=config.offset_k,
                padded_offset=config.padded_offset_k,
                stride_batch=stride_dvb,
                stride_seq=stride_dvn,
                HAS_CU_SEQLENS=config.HAS_CU_SEQLENS_K,
                USE_PADDED=True,
            )
            if dV is not None
            else None
        )

        # For split QO, offset key and value gradients base pointers by split_idx
        if config.IS_SPLIT_QO:
            if dK is not None:
                dk_base_ptr += config.split_idx * stride_dks
            if dV is not None:
                dv_base_ptr += config.split_idx * stride_dvs

        return AttnBwdPointerScheduler(
            q_base_ptr,
            k_base_ptr,
            v_base_ptr,
            lse_base_ptr,
            do_base_ptr,
            dpsum_base_ptr,
            dq_accum_base_ptr,
            dk_base_ptr,
            dv_base_ptr,
            tl.core.to_tensor(stride_qm),
            tl.core.to_tensor(stride_kn),
            tl.core.to_tensor(stride_vn),
            tl.core.to_tensor(stride_dom),
            tl.core.to_tensor(stride_dqam),
            tl.core.to_tensor(stride_dkn),
            tl.core.to_tensor(stride_dvn),
        )

    @triton.jit
    def make_q_ptrs(self, config: AttnBwdConfig):
        """
        Construct Q pointers for the current M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: Q pointers of shape [TILE_M, TILE_K]
        """
        return tl.make_tensor_descriptor(
            self.q_base_ptr,
            shape=[config.actual_seqlen_q, config.head_dim],
            strides=[self.stride_qm, 1],
            block_shape=[config.TILE_M, config.TILE_K],
        )

    @triton.jit
    def make_k_ptrs(self, config: AttnBwdConfig):
        """
        Construct K pointers for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: K pointers of shape [TILE_N, TILE_K]
        """
        return tl.make_tensor_descriptor(
            self.k_base_ptr,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_kn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_v_ptrs(self, config: AttnBwdConfig):
        """
        Construct V pointers for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: V pointers of shape [TILE_N, TILE_K]
        """
        return tl.make_tensor_descriptor(
            self.v_base_ptr,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_vn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_lse_ptrs(self, config: AttnBwdConfig):
        """
        Construct LSE pointers for the current M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: LSE pointers of shape [TILE_M]
        """
        return self.lse_base_ptr

    @triton.jit
    def make_do_ptrs(self, config: AttnBwdConfig):
        """
        Construct dO pointers for the current M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: dO pointers of shape [TILE_M, TILE_K]
        """
        return tl.make_tensor_descriptor(
            self.do_base_ptr,
            shape=[config.actual_seqlen_q, config.head_dim],
            strides=[self.stride_dom, 1],
            block_shape=[config.TILE_M, config.TILE_K],
        )

    @triton.jit
    def make_dpsum_ptrs(self, config: AttnBwdConfig):
        """
        Construct dPsum pointers for the current M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: dPsum pointers of shape [TILE_M]
        """
        return self.dpsum_base_ptr

    @triton.jit
    def make_dq_accum_ptrs(self, config: AttnBwdConfig):
        """
        Construct dQaccum pointers for the current M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: dQaccum tensor descriptor with block shape [TILE_M, TILE_K]
        """
        return tl.make_tensor_descriptor(
            self.dq_accum_base_ptr,
            shape=[config.actual_seqlen_q, config.head_dim],
            strides=[self.stride_dqam, 1],
            block_shape=[config.TILE_M, config.TILE_K],
        )

    @triton.jit
    def make_dk_ptrs(self, config: AttnBwdConfig):
        """
        Construct dK pointers for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: dK pointers of shape [TILE_N, TILE_K]
        """
        return tl.make_tensor_descriptor(
            self.dk_base_ptr,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_dkn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def make_dv_ptrs(self, config: AttnBwdConfig):
        """
        Construct dV pointers for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig

        :return: dV pointers of shape [TILE_N, TILE_K]
        """
        return tl.make_tensor_descriptor(
            self.dv_base_ptr,
            shape=[config.actual_seqlen_k, config.head_dim],
            strides=[self.stride_dvn, 1],
            block_shape=[config.TILE_N, config.TILE_K],
        )

    @triton.jit
    def load_q(self, config: AttnBwdConfig, q_ptrs, m_block):
        """
        Load the Q tile for the selected M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param q_ptrs: Q pointer
        :type q_ptrs: tensor
        :param m_block: selected M block index
        :type m_block: tensor

        :return: Q tile of shape [TILE_M, TILE_K]
        """
        q = q_ptrs.load([m_block * config.TILE_M, 0])
        if config.IS_QUANT:
            q = (q * config.query_scale).to(self.do_base_ptr.dtype.element_ty)
        return q

    @triton.jit
    def load_k(self, config: AttnBwdConfig, k_ptrs):
        """
        Load the K tile for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param k_ptrs: K pointer
        :type k_ptrs: tensor

        :return: K tile of shape [TILE_N, TILE_K]
        """
        k = k_ptrs.load([config.n_block * config.TILE_N, 0])
        if config.IS_QUANT:
            k = (k * config.key_scale).to(self.do_base_ptr.dtype.element_ty)
        return k

    @triton.jit
    def load_v(self, config: AttnBwdConfig, v_ptrs):
        """
        Load the V tile for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param v_ptrs: V pointer
        :type v_ptrs: tensor

        :return: V tile of shape [TILE_N, TILE_K]
        """
        v = v_ptrs.load([config.n_block * config.TILE_N, 0])
        if config.IS_QUANT:
            v = (v * config.value_scale).to(self.do_base_ptr.dtype.element_ty)
        return v

    @triton.jit
    def load_do(self, config: AttnBwdConfig, do_ptrs, m_block):
        """
        Load the dO tile for the selected M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param do_ptrs: dO pointer
        :type do_ptrs: tensor
        :param m_block: selected M block index
        :type m_block: tensor

        :return: dO tile of shape [TILE_M, TILE_K]
        """
        return do_ptrs.load([m_block * config.TILE_M, 0])

    @triton.jit
    def load_lse(self, config: AttnBwdConfig, lse_ptrs, m_block):
        """
        Load the LSE tile for the selected M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param lse_ptrs: LSE pointer
        :type lse_ptrs: tensor
        :param m_block: selected M block index
        :type m_block: tensor

        :return: LSE tile of shape [TILE_M]
        """
        offs_m = m_block * config.TILE_M + config.offs_m()
        return tl.load(
            lse_ptrs + offs_m,
            mask=offs_m < config.actual_seqlen_q,
            other=0.0,
        )

    @triton.jit
    def load_dpsum(self, config: AttnBwdConfig, dpsum_ptrs, m_block):
        """
        Load the dPsum tile for the selected M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param dpsum_ptrs: dPsum pointer
        :type dpsum_ptrs: tensor
        :param m_block: selected M block index
        :type m_block: tensor

        :return: dPsum tile of shape [TILE_M]
        """
        offs_m = m_block * config.TILE_M + config.offs_m()
        return tl.load(
            dpsum_ptrs + offs_m,
            mask=offs_m < config.actual_seqlen_q,
            other=0.0,
        )

    @triton.jit
    def store_dq(self, config: AttnBwdConfig, dq_ptrs, m_block, dq_tile):
        """
        Store the dQ tile for the selected M block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param dq_ptrs: dQaccum pointer
        :type dq_ptrs: tensor
        :param m_block: selected M block index
        :type m_block: tensor
        :param dq_tile: dQ tile of shape [TILE_M, TILE_K]
        :type dq_tile: tensor
        """
        dq_ptrs.atomic_add([m_block * config.TILE_M, 0], dq_tile)

    @triton.jit
    def store_dk(self, config: AttnBwdConfig, dk_ptrs, dk_tile):
        """
        Store the dK tile for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param dk_ptrs: dK pointer
        :type dk_ptrs: tensor
        :param dk_tile: dK tile of shape [TILE_N, TILE_K]
        :type dk_tile: tensor
        """
        if config.QHEAD_PER_KVHEAD > 1:
            dk_ptrs.atomic_add([config.n_block * config.TILE_N, 0], dk_tile)
        else:
            dk_ptrs.store([config.n_block * config.TILE_N, 0], dk_tile)

    @triton.jit
    def store_dv(self, config: AttnBwdConfig, dv_ptrs, dv_tile):
        """
        Store the dV tile for the current N block.

        :param config: attention backward configuration
        :type config: AttnBwdConfig
        :param dv_ptrs: dV pointer
        :type dv_ptrs: tensor
        :param dv_tile: dV tile of shape [TILE_N, TILE_K]
        :type dv_tile: tensor
        """
        if config.QHEAD_PER_KVHEAD > 1:
            dv_ptrs.atomic_add([config.n_block * config.TILE_N, 0], dv_tile)
        else:
            dv_ptrs.store([config.n_block * config.TILE_N, 0], dv_tile)


@aggregate
class AttnMaskScheduler:
    fixed_block: tl.tensor
    actual_seqlen_q: tl.tensor
    actual_seqlen_k: tl.tensor
    window_size_sink: tl.tensor
    window_size_left: tl.tensor
    window_size_right: tl.tensor
    window_size_near: tl.tensor
    TILE_M: tl.constexpr
    TILE_N: tl.constexpr
    QHEAD_PER_KVHEAD_PACKGQA: tl.constexpr
    SWAP_AB: tl.constexpr

    @triton.constexpr_function
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
        self.TILE_M = tl.constexpr(TILE_M)
        self.TILE_N = tl.constexpr(TILE_N)
        self.QHEAD_PER_KVHEAD_PACKGQA = tl.constexpr(QHEAD_PER_KVHEAD_PACKGQA)
        self.SWAP_AB = tl.constexpr(SWAP_AB)

    @staticmethod
    @triton.jit
    def create(
        config: "AttnFwdConfig | AttnBwdConfig",
        SWAP_AB: tl.constexpr = False,
    ) -> "AttnMaskScheduler":
        """
        Build a mask scheduler from an attention configuration.

        :param config: forward or backward configuration
        :type config: AttnFwdConfig or AttnBwdConfig
        :param SWAP_AB: boolean flag indicating if M and N are swapped for backward
        :type SWAP_AB: bool

        :return: mask scheduler bound to the fixed block and sequence metadata
        """
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

    @triton.jit
    def apply_mask(
        self,
        acc_s: tl.tensor,
        iter_block: tl.tensor,
        offs_m: tl.tensor,
        offs_n: tl.tensor,
        MASK_SEQLEN: tl.constexpr = True,
        MASK_CAUSAL: tl.constexpr = False,
        MASK_LOCAL: tl.constexpr = False,
        MASK_SINK: tl.constexpr = False,
    ) -> tl.tensor:
        """
        Apply the selected seqlen, causal, local, and sink masks to the attention scores.

        :param acc_s: attention scores
        :type acc_s: tensor
        :param iter_block: block index of the iterated dimension
        :type iter_block: tensor
        :param offs_m: offsets within the M dimension
        :type offs_m: tensor
        :param offs_n: offsets within the N dimension
        :type offs_n: tensor
        :param MASK_SEQLEN: boolean flag indicating if seqlen masking is required
        :type MASK_SEQLEN: bool
        :param MASK_CAUSAL: boolean flag indicating if causal masking is required
        :type MASK_CAUSAL: bool
        :param MASK_LOCAL: boolean flag indicating if local masking is required
        :type MASK_LOCAL: bool
        :param MASK_SINK: boolean flag indicating if sink masking is required
        :type MASK_SINK: bool

        :return: masked attention scores
        """
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
    softmax_scale_log2: tl.tensor
    value_scale: tl.tensor

    @triton.constexpr_function
    def __init__(
        self,
        softmax_scale_log2,
        value_scale,
    ):
        self.softmax_scale_log2 = softmax_scale_log2
        self.value_scale = value_scale

    @staticmethod
    @triton.jit
    def create(
        config: "AttnFwdConfig | AttnBwdConfig",
    ) -> "SoftmaxScheduler":
        """
        Build a softmax scheduler from an attention configuration.

        :param config: forward or backward configuration
        :type config: AttnFwdConfig or AttnBwdConfig

        :return: softmax scheduler carrying score and value scales
        """
        return SoftmaxScheduler(
            config.softmax_scale_log2,
            config.value_scale,
        )

    @triton.jit
    def online_softmax(
        self,
        acc_s: tl.tensor,
        row_max: tl.tensor,
        row_sum: tl.tensor,
        row_max_curr: tl.tensor = None,
        CHECK_INF: tl.constexpr = False,
    ) -> tuple[tl.tensor, tl.tensor, tl.tensor, tl.tensor]:
        """
        Apply online softmax and update its running row statistics.

        :param acc_s: attention scores
        :type acc_s: tensor
        :param row_max: running maximum values per row
        :type row_max: tensor
        :param row_sum: running sum values per row
        :type row_sum: tensor
        :param row_max_curr: precomputed current row maximum values
        :type row_max_curr: tensor
        :param CHECK_INF: boolean flag indicating if negative infinity is clamped
        :type CHECK_INF: bool

        :return p: online softmax probabilities for the current score
        :return row_max_new: updated running maximum values per row
        :return row_sum_new: updated running sum values per row
        :return row_scale: scaling factors for the previous output accumulator
        """
        return softmax.online_softmax(
            acc_s=acc_s,
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=self.softmax_scale_log2,
            row_max_curr=row_max_curr,
            CHECK_INF=CHECK_INF,
        )

    @triton.jit
    def score_skip(
        self,
        acc_s: tl.tensor,
        row_max: tl.tensor,
        row_sum: tl.tensor,
        softmax_threshold_log2: tl.tensor,
    ) -> tuple[tl.tensor, tl.tensor]:
        """
        Determine whether the current score can be skipped.

        :param acc_s: attention scores tensor
        :type acc_s: tensor
        :param row_max: running maximum values per row
        :type row_max: tensor
        :param row_sum: running sum values per row
        :type row_sum: tensor
        :param softmax_threshold_log2: softmax threshold in log2-domain
        :type softmax_threshold_log2: tensor

        :return is_skip: boolean indicating whether the block can be skipped
        :return row_max_curr: current row maximum values
        """
        return softmax.score_skip(
            acc_s=acc_s,
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=self.softmax_scale_log2,
            softmax_threshold_log2=softmax_threshold_log2,
        )

    @triton.jit
    def prob_skip(
        self,
        p_log2: tl.tensor,
        softmax_threshold_log2: tl.tensor,
    ) -> tl.tensor:
        """
        Determine whether the current block can be skipped based on the softmax threshold.

        :param p_log2: attention probabilities tensor in log2-domain
        :type p_log2: tensor
        :param softmax_threshold_log2: softmax threshold in log2-domain
        :type softmax_threshold_log2: tensor

        :return is_skip: boolean indicating whether the block can be skipped
        """
        return softmax.prob_skip(
            p_log2=p_log2,
            softmax_threshold_log2=softmax_threshold_log2,
        )

    @triton.jit
    def rescale_o(
        self,
        acc_o: tl.tensor,
        row_scale: tl.tensor,
    ) -> tl.tensor:
        """
        Rescale an output accumulator after an online softmax update.

        :param acc_o: output accumulator tensor
        :type acc_o: tensor
        :param row_scale: scaling factors for each query row
        :type row_scale: tensor

        :return: rescaled output accumulator
        """
        return softmax.rescale_o(
            acc_o=acc_o,
            row_scale=row_scale,
        )

    @triton.jit
    def finalize(
        self,
        row_max: tl.tensor,
        row_sum: tl.tensor,
        IS_LOG2: tl.constexpr = False,
        CHECK_NAN: tl.constexpr = True,
    ) -> tuple[tl.tensor, tl.tensor]:
        """
        Finalize the online softmax output scale and logsumexp.

        :param row_max: final maximum values per row
        :type row_max: tensor
        :param row_sum: final sum values per row
        :type row_sum: tensor
        :param IS_LOG2: boolean flag indicating if logsumexp remains in log2-space
        :type IS_LOG2: bool
        :param CHECK_NAN: boolean flag indicating if invalid row sums are sanitized
        :type CHECK_NAN: bool

        :return row_scale: final scaling factors for the output accumulator
        :return lse: final logsumexp values per row
        """
        return softmax.finalize(
            row_max=row_max,
            row_sum=row_sum,
            scale_log2=self.softmax_scale_log2,
            final_scale=self.value_scale,
            IS_LOG2=IS_LOG2,
            CHECK_NAN=CHECK_NAN,
        )
