# Copyright (c) 2026, Jingze Shi.
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from typing import Tuple, Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from flash_sparse_attn.ops.cute.seqlen_info import SeqlenInfoQK


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    is_local: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    window_size_sink: Optional[Int32] = None
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    window_size_dist: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    num_splits: Int32 = 1
    # If True, the scheduler packs num_splits into the top 16 bits of split_idx
    pack_split_idx: cutlass.Constexpr[bool] = False
    num_n_blocks_per_split: Optional[cutlass.Constexpr[Int32]] = None

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32, Int32, Int32, Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        n_block_min = Int32(0)
        n_block_window_max = n_block_max
        n_block_window_min = n_block_min
        n_block_sink_min = Int32(0)
        n_block_sink_max = Int32(0)
        if const_expr(self.is_causal or self.is_local):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            m_idx_max = cutlass.min(m_idx_max, seqlen_info.seqlen_q)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_block_max = cutlass.min(n_block_max, cute.ceil_div(n_idx, self.tile_n))
            if const_expr(self.is_local):
                n_idx_right = n_idx - self.window_size_dist - self.window_size_right
                n_block_window_max = cutlass.min(
                    n_block_window_max,
                    cutlass.max(cute.ceil_div(n_idx_right, self.tile_n), 0),
                )
        if const_expr(self.is_local):
            n_block_sink_max = cutlass.min(
                cute.ceil_div(self.window_size_sink, self.tile_n),
                cutlass.max(cute.ceil_div(n_idx, self.tile_n), 0),
            )
            n_block_sink_exclude_max = cute.ceil_div(self.window_size_sink, self.tile_n)
            m_idx_min = m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_dist = n_idx - self.window_size_dist
            n_block_min = cutlass.max(n_idx_dist // self.tile_n, 0)
            n_block_min = cutlass.max(n_block_min, n_block_sink_exclude_max)
            n_idx_left = (
                n_idx - self.window_size_dist - self.window_size_right - self.window_size_left
            )
            n_block_window_min = cutlass.max(n_idx_left // self.tile_n, 0)
            n_block_window_min = cutlass.max(n_block_window_min, n_block_sink_exclude_max)
        if const_expr(self.is_split_kv):
            if const_expr(self.pack_split_idx):
                # Unpack num_splits from top 16 bits of split_idx (packed by scheduler)
                num_splits = split_idx >> 16
                split_idx = split_idx & 0xFFFF
            else:
                num_splits = self.num_splits
            if const_expr(self.num_n_blocks_per_split is not None):
                num_n_blocks_per_split = self.num_n_blocks_per_split
                n_block_min = n_block_min + split_idx * num_n_blocks_per_split
                n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
                n_block_sink_max = n_block_sink_max if split_idx == 0 else Int32(0)
                if const_expr(self.is_local):
                    n_block_window_min = n_block_min
                    n_block_window_max = n_block_min
            elif const_expr(self.is_local):
                n_block_diag_min = (
                    cutlass.max(
                        cute.ceil_div(cutlass.max(n_idx_dist + 1, 0), self.tile_n),
                        0,
                    )
                    if seqlen_info.seqlen_q == 1
                    else n_block_min
                )
                n_block_diag_min = cutlass.max(n_block_diag_min, n_block_sink_exclude_max)
                n_block_diag_max = n_block_max
                n_block_window_max = cutlass.max(n_block_window_max, n_block_window_min)
                total_n_blocks = cutlass.max(n_block_window_max - n_block_window_min, 0)
                base = total_n_blocks // num_splits
                extra = total_n_blocks % num_splits
                n_block_window_min_new = n_block_window_min + (
                    split_idx * (base + 1)
                    if split_idx < extra
                    else extra * (base + 1) + (split_idx - extra) * base
                )
                n_block_count = base + 1 if split_idx < extra else base
                n_block_window_max = cutlass.min(
                    n_block_window_min_new + n_block_count,
                    n_block_window_max,
                )
                n_block_window_min = n_block_window_min_new
                n_block_sink_max = n_block_sink_max if split_idx == 0 else Int32(0)
                n_block_non_diag_max = cutlass.max(n_block_window_max, n_block_sink_max)
                n_block_max = (
                    n_block_diag_max if split_idx >= num_splits - 1 else n_block_non_diag_max
                )
                n_block_min = (
                    n_block_diag_min if split_idx >= num_splits - 1 else n_block_non_diag_max
                )
            else:
                total_n_blocks = cutlass.max(n_block_max - n_block_min, 0)
                base = total_n_blocks // num_splits
                extra = total_n_blocks % num_splits
                n_block_min_new = n_block_min + (
                    split_idx * (base + 1)
                    if split_idx < extra
                    else extra * (base + 1) + (split_idx - extra) * base
                )
                n_block_count = base + 1 if split_idx < extra else base
                n_block_max = cutlass.min(n_block_min_new + n_block_count, n_block_max)
                n_block_min = n_block_min_new
                n_block_sink_max = n_block_sink_max if split_idx == 0 else Int32(0)
        n_block_min = cutlass.min(n_block_min, n_block_max)
        if const_expr(self.is_local):
            n_block_window_max = cutlass.max(
                cutlass.min(n_block_window_max, n_block_min),
                n_block_window_min,
            )
            n_block_sink_max = cutlass.max(n_block_sink_max, n_block_sink_min)
        else:
            n_block_window_min = Int32(0)
            n_block_window_max = Int32(0)
        return (
            n_block_min,
            n_block_max,
            n_block_window_min,
            n_block_window_max,
            n_block_sink_min,
            n_block_sink_max,
        )

    @cute.jit
    def get_m_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        n_block: Int32,
    ) -> Tuple[Int32, Int32, Int32, Int32, Int32, Int32]:
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
        m_block_min = Int32(0)
        m_block_window_max = m_block_max
        m_block_window_min = m_block_min
        m_block_sink_min = Int32(0)
        m_block_sink_max = Int32(0)
        if const_expr(self.is_causal or self.is_local):
            n_idx_min = n_block * self.tile_n
            m_idx = n_idx_min + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_block_min = cutlass.max(m_block_min, m_idx // self.tile_m)
            if const_expr(self.is_local):
                m_idx_right = m_idx + self.window_size_dist + self.window_size_right
                m_block_window_min = cutlass.max(m_block_window_min, m_idx_right // self.tile_m)
        if const_expr(self.is_local):
            n_block_sink_exclude_max = cute.ceil_div(self.window_size_sink, self.tile_n)
            is_sink_block = n_block < n_block_sink_exclude_max
            n_idx_min = n_block * self.tile_n
            m_idx_sink = n_idx_min + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_block_sink_min = cutlass.max(m_idx_sink // self.tile_m, 0)
            m_block_sink_max = (
                cute.ceil_div(seqlen_info.seqlen_q, self.tile_m) if is_sink_block else Int32(0)
            )
            m_block_sink_min = m_block_sink_min if is_sink_block else Int32(0)
            n_idx_max = (n_block + 1) * self.tile_n
            m_idx = n_idx_max + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            m_idx_dist = m_idx + self.window_size_dist
            m_block_max = cutlass.min(m_block_max, cute.ceil_div(m_idx_dist, self.tile_m))
            m_idx_left = (
                m_idx + self.window_size_dist + self.window_size_right + self.window_size_left
            )
            m_block_window_max = cutlass.min(
                m_block_window_max,
                cute.ceil_div(m_idx_left, self.tile_m),
            )
            m_block_min = Int32(0) if is_sink_block else m_block_min
            m_block_max = Int32(0) if is_sink_block else m_block_max
            m_block_window_min = Int32(0) if is_sink_block else m_block_window_min
            m_block_window_max = Int32(0) if is_sink_block else m_block_window_max
        m_block_min = cutlass.min(m_block_min, m_block_max)
        if const_expr(self.is_local):
            m_block_window_min = cutlass.min(
                cutlass.max(m_block_window_min, m_block_max),
                m_block_window_max,
            )
            m_block_sink_min = cutlass.min(m_block_sink_min, m_block_sink_max)
        else:
            m_block_window_min = Int32(0)
            m_block_window_max = Int32(0)
        return (
            m_block_min,
            m_block_max,
            m_block_window_min,
            m_block_window_max,
            m_block_sink_min,
            m_block_sink_max,
        )

    @cute.jit
    def get_n_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
        is_local: cutlass.Constexpr[Optional[bool]] = None,
    ) -> Int32:
        """If we have separate iterations with causal or local masking at the start, where do we stop"""
        is_local = self.is_local if const_expr(is_local is None) else is_local
        m_idx_min = m_block * self.tile_m
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_right = (
            n_idx
            if const_expr(not is_local)
            else n_idx - self.window_size_dist - self.window_size_right
        )
        return cutlass.max(n_block_min, n_idx_right // self.tile_n)

    @cute.jit
    def get_n_block_min_before_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        n_block_min: Int32,
    ) -> Int32:
        """If we have separate iterations with local masking at the end, where do we stop the non-masked iterations"""
        if const_expr(not self.is_local):
            return n_block_min
        else:
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = (
                n_idx - self.window_size_dist - self.window_size_right - self.window_size_left
            )
            return cutlass.max(n_block_min, cute.ceil_div(n_idx_left, self.tile_n))

    @cute.jit
    def get_m_block_min_causal_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        n_block: Int32,
        m_block_min: Int32,
    ) -> Int32:
        if const_expr(not self.is_causal and not self.is_local):
            return m_block_min
        n_idx_max = (n_block + 1) * self.tile_n
        m_idx = n_idx_max + seqlen_info.seqlen_q - seqlen_info.seqlen_k
        m_idx_right = (
            m_idx
            if const_expr(not self.is_local)
            else m_idx + self.window_size_dist + self.window_size_right
        )
        return cutlass.max(m_block_min, cute.ceil_div(m_idx_right, self.tile_m))

    @cute.jit
    def get_m_block_max_before_local_mask(
        self,
        seqlen_info: SeqlenInfoQK,
        n_block: Int32,
        m_block_max: Int32,
    ) -> Int32:
        if const_expr(not self.is_local):
            return m_block_max
        n_idx_min = n_block * self.tile_n
        m_idx = n_idx_min + seqlen_info.seqlen_q - seqlen_info.seqlen_k
        m_idx_left = m_idx + self.window_size_dist + self.window_size_right + self.window_size_left
        return cutlass.min(m_block_max, m_idx_left // self.tile_m)

    @cute.jit
    def get_n_block_max_for_m_block(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
    ) -> Int32:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        if const_expr(self.is_causal or self.is_local):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx_right = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_block_max = cutlass.min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        return n_block_max
