from typing import Optional
import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=['IS_CAUSAL', 'BLOCK_HEADDIM']
)
@triton.heuristics(
    {
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_preprocess(
    K,
    V,
    B,
    I,
    CuK,
    CuV,
    CuB,
    CuM,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bn,
    stride_ib,
    stride_ih,
    stride_ik,
    stride_ckb,
    stride_ckh,
    stride_ckk,
    stride_cvb,
    stride_cvh,
    stride_cvk,
    stride_cbb,
    stride_cbh,
    stride_cbk,
    stride_cmb,
    stride_cmh,
    stride_cmm,
    stride_cmk,
    nheads_k,
    seqlen_q,
    seqlen_k,
    window_size,
    headdim,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(0)
    off_b = off_hb // nheads_k
    off_hk = off_hb % nheads_k

    # Initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize base pointers to K, V, B, I, CuK, CuV, CuB
    k_base_ptrs = (
        K + off_b * stride_kb + off_hk * stride_kh
    )
    v_base_ptrs = (
        V + off_b * stride_vb + off_hk * stride_vh
    )
    b_base_ptrs = (
        B + off_b * stride_bb + off_hk * stride_bh
    )
    i_base_ptrs = (
        I + off_b * stride_ib + off_hk * stride_ih
    )
    cuk_base_ptrs = (
        CuK + off_b * stride_ckb + off_hk * stride_ckh
    )
    cuv_base_ptrs = (
        CuV + off_b * stride_cvb + off_hk * stride_cvh
    )
    cub_base_ptrs = (
        CuB + off_b * stride_cbb + off_hk * stride_cbh
    )
    cum_base_ptrs = (
        CuM + off_b * stride_cmb + off_hk * stride_cmh
    )

    # Loop over blocks of window_size
    for start_k in range(0, window_size, BLOCK_N):
        start_k = tl.multiple_of(start_k, BLOCK_N)
        offs_k = start_k + offs_n

        # Load I
        i_ptrs = (
            i_base_ptrs + offs_k * stride_ik
        )
        gather_idx = tl.load(i_ptrs, mask=offs_k < window_size, other=0).to(tl.int64)
        valid_idx = (offs_k < window_size) & (gather_idx >= 0) & (gather_idx < seqlen_k)
        gather_idx = tl.where(valid_idx, gather_idx, 0)

        # Load K, V, B
        k_ptrs = (
            k_base_ptrs + gather_idx[:, None] * stride_kn + offs_d[None, :]
        )
        v_ptrs = (
            v_base_ptrs + gather_idx[:, None] * stride_vn + offs_d[None, :]
        )
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=valid_idx[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=valid_idx[:, None], other=0.0)
        else:
            k = tl.load(
                k_ptrs,
                mask=valid_idx[:, None] & (offs_d[None, :] < headdim),
                other=0.0
            )
            v = tl.load(
                v_ptrs,
                mask=valid_idx[:, None] & (offs_d[None, :] < headdim),
                other=0.0
            )
        b_ptrs = (
            b_base_ptrs + gather_idx * stride_bn
        )
        b = tl.load(b_ptrs, mask=valid_idx, other=0.0)
        
        # Store to CuK, CuV, CuB
        cuk_ptrs = (
            cuk_base_ptrs + offs_k[:, None] * stride_ckk + offs_d[None, :]
        )
        cuv_ptrs = (
            cuv_base_ptrs + offs_k[:, None] * stride_cvk + offs_d[None, :]
        )
        if EVEN_HEADDIM:
            tl.store(cuk_ptrs, k, mask=valid_idx[:, None])
            tl.store(cuv_ptrs, v, mask=valid_idx[:, None])
        else:
            tl.store(
                cuk_ptrs, k,
                mask=valid_idx[:, None] & (offs_d[None, :] < headdim),
            )
            tl.store(
                cuv_ptrs, v,
                mask=valid_idx[:, None] & (offs_d[None, :] < headdim),
            )
        cub_ptrs = (
            cub_base_ptrs + offs_k * stride_cbk
        )
        tl.store(cub_ptrs, b, mask=valid_idx)

        # Store mask to CuM
        for start_m in range(0, seqlen_q, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + tl.arange(0, BLOCK_M)

            cum_ptrs = (
                cum_base_ptrs + offs_m[:, None] * stride_cmm + offs_k[None, :] * stride_cmk
            )

            col_mask = offs_k < window_size
            row_mask = offs_m[:, None] < seqlen_q

            if IS_CAUSAL:
                mask = (offs_m[:, None] >= gather_idx[None, :]) & valid_idx[None, :]
            else:
                mask = valid_idx[None, :]

            cum = tl.where(row_mask & col_mask[None, :], mask, False)

            tl.store(cum_ptrs, cum, mask=row_mask & col_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BLOCK_HEADDIM']
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["window_size"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    CuK,
    CuV,
    CuB,
    CuM,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_ckb,
    stride_ckh,
    stride_ckk,
    stride_cvb,
    stride_cvh,
    stride_cvk,
    stride_cbb,
    stride_cbh,
    stride_cbk,
    stride_cmb,
    stride_cmh,
    stride_cmm,
    stride_cmk,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    h_h_k_ratio,
    seqlen_q,
    window_size,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q: tl.constexpr,
    CACHE_KEY_SEQLEN_K: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // nheads
    off_hq = off_bh % nheads
    off_hk = off_hq // h_h_k_ratio

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, CuK, CuV, CuM, CuB
    q_ptrs = (
        Q + off_b * stride_qb + off_hq * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    cuk_base_ptrs = (
        CuK + off_b * stride_ckb + off_hk * stride_ckh
    )
    cv_base_ptrs = (
        CuV + off_b * stride_cvb + off_hk * stride_cvh
    )
    cub_base_ptrs = (
        CuB + off_b * stride_cbb + off_hk * stride_cbh
    )
    cum_base_ptrs = (
        CuM + off_b * stride_cmb + off_hk * stride_cmh
    )

    # Initialize pointer to m and l
    lse_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Load q: it will stay in SRAM throughout
    if EVEN_M:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0
            )

    # Scale q
    q = (q * softmax_scale).to(q.dtype)

    # Loop over k, v and update accumulator
    for start_n in range(0, window_size, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        cum_ptrs = (
            cum_base_ptrs + offs_m[:, None] * stride_cmm + (start_n + offs_n)[None, :] * stride_cmk
        )
        # Load mask
        if EVEN_M & EVEN_N:
            m = tl.load(cum_ptrs)
        else:
            m = tl.load(
                cum_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < window_size),
                other=False,
            )

        # Check if any element in mask is non-zero
        any_active = tl.reduce_or(m, axis=None)

        # Skip this iteration if no active elements
        if any_active:

            # Load k
            cuk_ptrs = (
                cuk_base_ptrs + (start_n + offs_n)[:, None] * stride_ckk + offs_d[None, :]
            )
            if EVEN_N:
                if EVEN_HEADDIM:
                    k = tl.load(cuk_ptrs)
                else:
                    k = tl.load(cuk_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            else:
                if EVEN_HEADDIM:
                    k = tl.load(
                        cuk_ptrs,
                        mask=(start_n + offs_n)[:, None] < window_size,
                        other=0.0,
                    )
                else:
                    k = tl.load(
                        cuk_ptrs,
                        mask=((start_n + offs_n)[:, None] < window_size) & (offs_d[None, :] < headdim),
                        other=0.0,
                    )

            # Load bias
            cub_ptrs = (
                cub_base_ptrs + (start_n + offs_n) * stride_cbk
            )
            if EVEN_M & EVEN_N:
                b = tl.load(cub_ptrs)
            else:
                b = tl.load(
                    cub_ptrs,
                    mask=(start_n + offs_n) < window_size,
                    other=0.0,
                )
            
            # Initialize acc_s
            acc_s = b[None, :].to(tl.float32)

            # Compute acc_s
            acc_s += tl.dot(q, tl.trans(k))

            # Apply masks
            # Trying to combine the two masks seem to make the result wrong
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                acc_s += tl.where((start_n + offs_n)[None, :] < window_size, 0, float("-inf"))
            acc_s += tl.where(m, 0, float("-inf"))

            # Compute p
            m_ij = tl.maximum(tl.max(acc_s, 1), lse_i)
            p = tl.exp(acc_s - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            # Scale acc_o
            acc_o_scale = tl.exp(m_i - m_ij)

            # Update output accumulator
            acc_o = acc_o * acc_o_scale[:, None]

            # Load v
            cuv_ptrs = (
                cv_base_ptrs + (start_n + offs_n)[:, None] * stride_cvk + offs_d[None, :]
            )
            if EVEN_N:
                if EVEN_HEADDIM:
                    v = tl.load(cuv_ptrs)
                else:
                    v = tl.load(cuv_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            else:
                if EVEN_HEADDIM:
                    v = tl.load(
                        cuv_ptrs,
                        mask=(start_n + offs_n)[:, None] < window_size,
                        other=0.0,
                    )
                else:
                    v = tl.load(
                        cuv_ptrs,
                        mask=((start_n + offs_n)[:, None] < window_size) & (offs_d[None, :] < headdim),
                        other=0.0,
                    )

            # Compute acc_o
            acc_o += tl.dot(p.to(v.dtype), v)

            # Update statistics
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    o_scale = tl.where(lse_i > float("-inf"), o_scale, 0.0)
    acc_o = acc_o * o_scale[:, None]
    # Rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Write back l and m
    lse_ptrs = Lse + off_bh * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # Initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_hq * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Load o
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # Write back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    CuK,
    CuV,
    CuB,
    CuM,
    DO,
    DQ,
    DCuK,
    DCuV,
    DCuB,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_ckk,
    stride_cvk,
    stride_cbk,
    stride_cmm,
    stride_cmk,
    stride_dom,
    stride_dqm,
    stride_dckk,
    stride_dcvk,
    stride_dcbk,
    seqlen_q,
    window_size,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Initialize row/col offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :])
    cuk_ptrs = CuK + (offs_n[:, None] * stride_ckk + offs_d[None, :])
    cuv_ptrs = CuV + (offs_n[:, None] * stride_cvk + offs_d[None, :])
    cub_ptrs = CuB + (offs_n * stride_cbk)
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_d[None, :])
    dcuk_ptrs = DCuK + (offs_n[:, None] * stride_dckk + offs_d[None, :])
    dcuv_ptrs = DCuV + (offs_n[:, None] * stride_dcvk + offs_d[None, :])
    dcub_ptrs = DCuB + (offs_n * stride_dcbk)

    # Initialize dv, dk, db accumulators
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    db = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Load k and v, them will stay in SRAM throughout
    if EVEN_N:
        if EVEN_HEADDIM:
            k = tl.load(cuk_ptrs)
            v = tl.load(cuv_ptrs)
        else:
            k = tl.load(cuk_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(cuv_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(cuk_ptrs, mask=offs_n[:, None] < window_size, other=0.0)
            v = tl.load(cuv_ptrs, mask=offs_n[:, None] < window_size, other=0.0)
        else:
            k = tl.load(
                cuk_ptrs,
                mask=(offs_n[:, None] < window_size) & (offs_d[None, :] < headdim),
                other=0.0
            )
            v = tl.load(
                cuv_ptrs,
                mask=(offs_n[:, None] < window_size) & (offs_d[None, :] < headdim),
                other=0.0
            )
    if EVEN_N:
        b = tl.load(cub_ptrs)
    else:
        b = tl.load(cub_ptrs, mask=offs_n < window_size, other=0.0)

    # Scale k
    k = (k * softmax_scale).to(k.dtype)

    # Loop over q and update accumulators
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(0, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        # Load mask
        cum_ptrs = (
            CuM + offs_m_curr[:, None] * stride_cmm + offs_n[None, :] * stride_cmk
        )
        if EVEN_M & EVEN_N:
            m = tl.load(cum_ptrs)
        else:
            m = tl.load(
                cum_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < window_size),
                other=False,
            )

        # Check if any element in mask is non-zero
        any_active = tl.reduce_or(m, axis=None)

        # Skip this iteration if no active elements
        if any_active:
            # Load q
            if EVEN_M & EVEN_HEADDIM:
                q = tl.load(q_ptrs)
            else:
                if EVEN_HEADDIM:
                    q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
                else:
                    q = tl.load(
                        q_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                    )

            # Initialize acc_s
            acc_s = b[None, :].to(tl.float32)

            # Compute acc_s
            acc_s += tl.dot(q, tl.trans(k))

            # Apply masks
            # Trying to combine the two masks seem to make the result wrong
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                acc_s += tl.where(offs_n[None, :] < window_size, 0, float("-inf"))
            acc_s += tl.where(m, 0, float("-inf"))

            lse_i = tl.load(LSE + offs_m_curr)
            # p = tl.exp(acc_s - lse_i[:, None])
            p = tl.exp(acc_s - tl.where(lse_i > float("-inf"), lse_i, 0.0)[:, None])

            # Load do
            if EVEN_M & EVEN_HEADDIM:
                do = tl.load(do_ptrs)
            else:
                # There's a race condition if we just use m_mask and not d_mask.
                do = tl.load(
                    do_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

            # Compute dv
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)

            # Compute dp
            dp = tl.dot(do, tl.trans(v))
            
            # Putting the subtraction after the dp matmul (instead of before) is slightly faster
            Di = tl.load(D + offs_m_curr)

            # Compute ds
            # Converting ds to q.dtype here reduces register pressure and makes it much faster
            # for BLOCK_HEADDIM=128
            ds = (p * (dp - Di[:, None])).to(q.dtype)

            # Compute db
            db += tl.sum(ds, axis=0)

            # Compute dk
            dk += tl.dot(tl.trans(ds), q)

            # Compute dq
            if not ATOMIC_ADD:
                if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                    dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                    dq += tl.dot(ds, k).to(ds.dtype)
                    tl.store(dq_ptrs, dq, eviction_policy="evict_last")
                else:
                    if EVEN_HEADDIM:
                        dq = tl.load(
                            dq_ptrs,
                            mask=offs_m_curr[:, None] < seqlen_q,
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                        dq += tl.dot(ds, k).to(ds.dtype)
                        tl.store(
                            dq_ptrs,
                            dq,
                            mask=offs_m_curr[:, None] < seqlen_q,
                            eviction_policy="evict_last",
                        )
                    else:
                        dq = tl.load(
                            dq_ptrs,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                        dq += tl.dot(ds, k).to(ds.dtype)
                        tl.store(
                            dq_ptrs,
                            dq,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                            eviction_policy="evict_last",
                        )
            else:  # If we're parallelizing across the seqlen_k dimension
                dq = tl.dot(ds, k).to(ds.dtype)
                if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                    tl.atomic_add(dq_ptrs, dq)
                else:
                    if EVEN_HEADDIM:
                        tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                    else:
                        tl.atomic_add(
                            dq_ptrs,
                            dq,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        )

            # Increment pointers
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_dom
            dq_ptrs += BLOCK_M * stride_dqm
        else:
            # Increment pointers
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_dom
            dq_ptrs += BLOCK_M * stride_dqm

    # Scale dk
    dk = (dk * softmax_scale).to(dk.dtype)

    # Write back
    if EVEN_N:
        if EVEN_HEADDIM:
            tl.store(dcuk_ptrs, dk)
            tl.store(dcuv_ptrs, dv)
        else:
            tl.store(dcuk_ptrs, dk, mask=offs_d[None, :] < headdim)
            tl.store(dcuv_ptrs, dv, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dcuk_ptrs, dk, mask=offs_n[:, None] < window_size)
            tl.store(dcuv_ptrs, dv, mask=offs_n[:, None] < window_size)
        else:
            tl.store(dcuk_ptrs, dk, mask=(offs_n[:, None] < window_size) & (offs_d[None, :] < headdim))
            tl.store(dcuv_ptrs, dv, mask=(offs_n[:, None] < window_size) & (offs_d[None, :] < headdim))

    if EVEN_N:
        tl.store(dcub_ptrs, db)
    else:
        tl.store(dcub_ptrs, db, mask=(offs_n < window_size))


def init_to_zero(names):
    if isinstance(names, str):
        names = [names]
    def init_func(nargs):
        for name in names:
            nargs[name].zero_()
    return init_func


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero(["DQ", "DCuB"]),
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero(["DQ", "DCuB"]),
        ),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["window_size"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    CuK,
    CuV,
    CuB,
    CuM,
    DO,
    DQ,
    DCuK,
    DCuV,
    DCuB,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_ckb,
    stride_ckh,
    stride_ckk,
    stride_cvb,
    stride_cvh,
    stride_cvk,
    stride_cbb,
    stride_cbh,
    stride_cbk,
    stride_cmb,
    stride_cmh,
    stride_cmm,
    stride_cmk,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dckb,
    stride_dckh,
    stride_dckk,
    stride_dcvb,
    stride_dcvh,
    stride_dcvk,
    stride_dcbb,
    stride_dcbh,
    stride_dcbk,
    nheads,
    h_h_k_ratio,
    seqlen_q,
    window_size,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_hq = off_hb % nheads
    off_hk = off_hq // h_h_k_ratio

    # Advance offset pointers for batch and head
    Q += off_b * stride_qb + off_hq * stride_qh
    CuK += off_b * stride_ckb + off_hk * stride_ckh
    CuV += off_b * stride_cvb + off_hk * stride_cvh
    CuB += off_b * stride_cbb + off_hk * stride_cbh
    CuM += off_b * stride_cmb + off_hk * stride_cmh
    DO += off_b * stride_dob + off_hq * stride_doh
    DQ += off_b * stride_dqb + off_hq * stride_dqh
    DCuK += off_b * stride_dckb + off_hq * stride_dckh
    DCuV += off_b * stride_dcvb + off_hq * stride_dcvh
    DCuB += off_b * stride_dcbb + off_hq * stride_dcbh
    # Advance pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded

    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(window_size, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                CuK,
                CuV,
                CuB,
                CuM,
                DO,
                DQ,
                DCuK,
                DCuV,
                DCuB,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_ckk,
                stride_cvk,
                stride_cbk,
                stride_cmm,
                stride_cmk,
                stride_dom,
                stride_dqm,
                stride_dckk,
                stride_dcvk,
                stride_dcbk,
                seqlen_q,
                window_size,
                headdim,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                ATOMIC_ADD=False,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            CuK,
            CuV,
            CuB,
            CuM,
            DO,
            DQ,
            DCuK,
            DCuV,
            DCuB,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_ckk,
            stride_cvk,
            stride_cbk,
            stride_cmm,
            stride_cmk,
            stride_dom,
            stride_dqm,
            stride_dckk,
            stride_dcvk,
            stride_dcbk,
            seqlen_q,
            window_size,
            headdim,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            ATOMIC_ADD=True,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def _flash_dmattn_forward(q, k, v, b, i, softmax_scale=None, is_causal=False, window_size=None):
    # shape constraints
    batch, nheads, seqlen_q, d = q.shape
    _, nheads_k, seqlen_k, _ = k.shape

    assert nheads % nheads_k == 0, "Number of Q heads must be divisible by KV heads for GQA/MQA"
    assert d <= 128, "FlashDynamicMaskAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype == b.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert i.dtype == torch.int64, "Indices must be int64"
    assert q.is_cuda and k.is_cuda and v.is_cuda and b.is_cuda, "All tensors must be on GPU"

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    cu_k = torch.empty((batch, nheads_k, window_size, d), device=q.device, dtype=k.dtype)
    cu_v = torch.empty((batch, nheads_k, window_size, d), device=q.device, dtype=v.dtype)
    cu_b = torch.empty((batch, nheads_k, window_size), device=q.device, dtype=b.dtype)
    cu_m = torch.zeros((batch, nheads_k, seqlen_q, window_size), device=q.device, dtype=torch.bool)

    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = (batch * nheads_k,)
    _fwd_preprocess[grid](
        k, v, b, i,
        cu_k, cu_v, cu_b, cu_m,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        i.stride(0), i.stride(1), i.stride(2),
        cu_k.stride(0), cu_k.stride(1), cu_k.stride(2),
        cu_v.stride(0), cu_v.stride(1), cu_v.stride(2),
        cu_b.stride(0), cu_b.stride(1), cu_b.stride(2),
        cu_m.stride(0), cu_m.stride(1), cu_m.stride(2), cu_m.stride(3),
        nheads_k, seqlen_q, seqlen_k, window_size, d, is_causal, BLOCK_HEADDIM
    )

    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        cu_k, cu_v, cu_b, cu_m,
        o, lse, softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        cu_k.stride(0), cu_k.stride(1), cu_k.stride(2),
        cu_v.stride(0), cu_v.stride(1), cu_v.stride(2),
        cu_b.stride(0), cu_b.stride(1), cu_b.stride(2),
        cu_m.stride(0), cu_m.stride(1), cu_m.stride(2), cu_m.stride(3),
        o.stride(0), o.stride(1), o.stride(2),
        nheads, nheads // nheads_k, seqlen_q, window_size, seqlen_q_rounded, d,
        seqlen_q // 32,
        window_size // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # BLOCK_HEADDIM=d,
        BLOCK_HEADDIM,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    return o, lse, softmax_scale, cu_k, cu_v, cu_b, cu_m


def _flash_dmattn_backward(
    do, q, cuk, cuv, cub, cum, i, o, lse, softmax_scale, seqlen_q, seqlen_k, window_size
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, nheads, _, d = q.shape
    _, nheads_k, _, _ = cuk.shape

    assert nheads % nheads_k == 0, "Number of Q heads must be divisible by KV heads for GQA/MQA"
    assert d <= 128, "FlashDynamicMaskAttention only support head dimensions up to 128"
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    seqlen_k_rounded = math.ceil(seqlen_k / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    # delta = torch.zeros_like(lse)
    dk = torch.zeros(batch, nheads_k, seqlen_k, d, device=q.device, dtype=q.dtype)
    dv = torch.zeros(batch, nheads_k, seqlen_k, d, device=q.device, dtype=q.dtype)
    db = torch.zeros(batch, nheads_k, seqlen_k, device=q.device, dtype=q.dtype)

    dk_expanded = torch.empty(batch, nheads, window_size, d, device=q.device, dtype=q.dtype)
    dv_expanded = torch.empty(batch, nheads, window_size, d, device=q.device, dtype=q.dtype)
    db_expanded = torch.empty(batch, nheads, window_size, device=q.device, dtype=q.dtype)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        nheads, seqlen_q, seqlen_q_rounded, d,
        BLOCK_M=64,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4
    grid = lambda META: (
        triton.cdiv(window_size, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q, cuk, cuv, cub, cum, do,
        dq_accum, dk_expanded, dv_expanded, db_expanded,
        lse, delta, softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        cuk.stride(0), cuk.stride(1), cuk.stride(2),
        cuv.stride(0), cuv.stride(1), cuv.stride(2),
        cub.stride(0), cub.stride(1), cub.stride(2),
        cum.stride(0), cum.stride(1), cum.stride(2), cum.stride(3),
        do.stride(0), do.stride(1), do.stride(2),
        dq_accum.stride(0), dq_accum.stride(1), dq_accum.stride(2),
        dk_expanded.stride(0), dk_expanded.stride(1), dk_expanded.stride(2),
        dv_expanded.stride(0), dv_expanded.stride(1), dv_expanded.stride(2),
        db_expanded.stride(0), db_expanded.stride(1), db_expanded.stride(2),
        nheads, nheads // nheads_k, seqlen_q, window_size, seqlen_q_rounded, d,
        seqlen_q // 32,
        window_size // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq = dq_accum.to(q.dtype)

    if nheads != nheads_k:
        dk_expanded = dk_expanded.view(batch, nheads_k, nheads // nheads_k, window_size, d).sum(dim=2)
        dv_expanded = dv_expanded.view(batch, nheads_k, nheads // nheads_k, window_size, d).sum(dim=2)
        db_expanded = db_expanded.view(batch, nheads_k, nheads // nheads_k, window_size).sum(dim=2)

    dk.scatter_add_(
        dim=2,
        index=i.unsqueeze(-1).expand(-1, -1, -1, d),
        src=dk_expanded,
    )
    dv.scatter_add_(
        dim=2,
        index=i.unsqueeze(-1).expand(-1, -1, -1, d),
        src=dv_expanded,
    )
    db.scatter_add_(
        dim=2,
        index=i,
        src=db_expanded,
    )

    return dq, dk, dv, db


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


class FlashDMAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, attn_bias, attn_indices, is_causal=False, softmax_scale=None):
        """
        query: (batch_size, nheads, seqlen_q, headdim)
        key: (batch_size, nheads_k, seqlen_k, headdim)
        value: (batch_size, nheads_k, seqlen_k, headdim)
        attn_bias: (batch_size, nheads_k, seqlen_k)
        attn_indices: (batch_size, nheads_k, window_size)
        is_causal: bool, whether to apply causal masking
        softmax_scale: float, scaling factor for attention scores
        """

        # Make sure that the last dimension is contiguous
        query, key, value, attn_bias, attn_indices = [maybe_contiguous(x) for x in [query, key, value, attn_bias, attn_indices]]

        # Padding to multiple of 8 for 16-bit memory allocations
        head_size_og = query.size(3)
        if head_size_og % 8 != 0:
            query = torch.nn.functional.pad(query, [0, 8 - head_size_og % 8])
            key = torch.nn.functional.pad(key, [0, 8 - head_size_og % 8])
            value = torch.nn.functional.pad(value, [0, 8 - head_size_og % 8])
        seqlen_k_rounded = round_multiple(key.shape[2], 128)
        if attn_bias.shape[-1] != seqlen_k_rounded:
                attn_bias = torch.nn.functional.pad(attn_bias, [0, seqlen_k_rounded - attn_bias.shape[-1]])
        window_size = attn_indices.shape[-1]

        o, lse, ctx.softmax_scale, cu_key, cu_value, cu_attn_bias, cu_attn_mask = _flash_dmattn_forward(
            query,
            key,
            value,
            attn_bias,
            attn_indices,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_size=window_size,
        )
        ctx.save_for_backward(query, cu_key, cu_value, cu_attn_bias, cu_attn_mask, attn_indices, o, lse)
        ctx.seqlen_q = query.size(2)
        ctx.seqlen_k = key.size(2)
        ctx.window_size = window_size

        o = o[..., : head_size_og]
        return o

    @staticmethod
    def backward(ctx, do):
        query, cu_key, cu_value, cu_attn_bias, cu_attn_mask, attn_indices, o, lse = ctx.saved_tensors

        head_size_og = do.size(3)
        do_padded = do
        if head_size_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_og % 8])

        dq, dk, dv, db = _flash_dmattn_backward(
            do_padded,
            query,
            cu_key,
            cu_value,
            cu_attn_bias,
            cu_attn_mask,
            attn_indices,
            o,
            lse,
            softmax_scale=ctx.softmax_scale,
            seqlen_q=ctx.seqlen_q,
            seqlen_k=ctx.seqlen_k,
            window_size=ctx.window_size,
        )

        # We could have padded the head dimension
        dq = dq[..., : do.shape[-1]]  
        dk = dk[..., : do.shape[-1]]
        dv = dv[..., : do.shape[-1]]

        return dq, dk, dv, db, None, None, None


def triton_dmattn_func(query, key, value, attn_bias, attn_indices, is_causal=False, softmax_scale=None):
    return FlashDMAttnFunc.apply(query, key, value, attn_bias, attn_indices, is_causal, softmax_scale)