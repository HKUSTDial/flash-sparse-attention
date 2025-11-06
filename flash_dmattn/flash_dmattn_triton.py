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
    key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'IS_CAUSAL', 'HAS_MASK', 'HAS_BIAS', 'BLOCK_HEADDIM']
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Mask,
    Bias,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    nheads_k,
    nheads_mask,
    nheads_bias,
    h_h_k_ratio,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q: tl.constexpr,
    CACHE_KEY_SEQLEN_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_hq = off_hb % nheads
    off_hk = off_hq // h_h_k_ratio
    if HAS_MASK:
        if nheads_mask == 1:
            off_hmask = 0
        elif nheads_mask == nheads_k:
            off_hmask = off_hk
        else:
            off_hmask = off_hq
    if HAS_BIAS:
        if nheads_bias == 1:
            off_hbbias = 0
        elif nheads_bias == nheads_k:
            off_hbbias = off_hk
        else:
            off_hbbias = off_hq
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V, Mask, Bias
    q_ptrs = (
        Q + off_b * stride_qb + off_hq * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_hk * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_hk * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    m_ptrs = (
        Mask + off_b * stride_mb + off_hmask * stride_mh + (offs_m[:, None] * stride_mm + offs_n[None, :])
    ) if HAS_MASK else None
    b_ptrs = (
        Bias + off_b * stride_bb + off_hbbias * stride_bh + (offs_m[:, None] * stride_bm + offs_n[None, :])
    ) if HAS_BIAS else None

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
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )

    # Scale q
    q = (q * softmax_scale).to(q.dtype)

    # Loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        if HAS_MASK:
            # Load mask
            if EVEN_M & EVEN_N:
                mask = tl.load(m_ptrs + start_n)
            else:
                mask = tl.load(
                    m_ptrs + start_n,
                    mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                    other=False
                )

            # Check if any element in mask is non-zero
            any_active = tl.reduce_or(mask, axis=None)
        else:
            any_active = True

        # Skip this iteration if no active elements
        if any_active:

            # Load k
            if EVEN_N:
                if EVEN_HEADDIM:
                    k = tl.load(k_ptrs + start_n * stride_kn)
                else:
                    k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
            else:
                if EVEN_HEADDIM:
                    k = tl.load(
                        k_ptrs + start_n * stride_kn,
                        mask=(start_n + offs_n)[:, None] < seqlen_k,
                        other=0.0,
                    )
                else:
                    k = tl.load(
                        k_ptrs + start_n * stride_kn,
                        mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                        other=0.0,
                    )

            if HAS_BIAS:
                # Load bias
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
                acc_s = bias
            else:
                acc_s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            # Compute acc_s
            acc_s += tl.dot(q, tl.trans(k))

            # Apply masks
            # Trying to combine the three masks seem to make the result wrong
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                acc_s += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
            if IS_CAUSAL:
                acc_s += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
            if HAS_MASK:
                acc_s += tl.where(mask, 0, float("-inf"))

            # Compute p
            m_ij = tl.maximum(tl.max(acc_s, 1), lse_i)
            p = tl.exp(acc_s - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            # Scale acc_o
            acc_o_scale = tl.exp(m_i - m_ij)

            # Update output accumulator
            acc_o = acc_o * acc_o_scale[:, None]

            # Load v
            if EVEN_N:
                if EVEN_HEADDIM:
                    v = tl.load(v_ptrs + start_n * stride_vn)
                else:
                    v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
            else:
                if EVEN_HEADDIM:
                    v = tl.load(
                        v_ptrs + start_n * stride_vn,
                        mask=(start_n + offs_n)[:, None] < seqlen_k,
                        other=0.0,
                    )
                else:
                    v = tl.load(
                        v_ptrs + start_n * stride_vn,
                        mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                        other=0.0,
                    )

            # Compute acc_o
            acc_o += tl.dot(p.to(v.dtype), v)

            # Update statistics
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    # Rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
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
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
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
    K,
    V,
    Mask,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    DBias,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_mm,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    stride_dbm,
    seqlen_q,
    seqlen_k,
    headdim,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ACCUM_DBIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # Initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    if HAS_MASK:
        m_ptrs = Mask + (offs_qm[:, None] * stride_mm + offs_n[None, :])
    if HAS_BIAS:
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    db_ptrs = DBias + (offs_qm[:, None] * stride_dbm + offs_n[None, :])
    # Initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255). In this case the for loop may have zero step,
    # and pipelining with the bias matrix could screw it up. So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])

        if EVEN_N:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv)
                tl.store(dk_ptrs, dk)
            else:
                tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
                tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
        else:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
                tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
            else:
                tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
                tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
        return

    # Load k and v, them will stay in SRAM throughout
    if EVEN_N:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )

    # Scale k
    k = (k * softmax_scale).to(k.dtype)

    # Initialize accumulator for dbias if needed
    acc_dbias = tl.zeros([BLOCK_N], dtype=tl.float32) if (HAS_BIAS and ACCUM_DBIAS) else None

    # Loop over q and update accumulators
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        if HAS_MASK:
            # Load mask
            if EVEN_M & EVEN_N:
                mask = tl.load(m_ptrs)
            else:
                mask = tl.load(
                    m_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                    other=False,
                )

            # Check if any element in mask is non-zero
            any_active = tl.reduce_or(mask, axis=None)
        else:
            any_active = True
        
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

            if HAS_BIAS:
                # Load bias
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
                acc_s = bias
            else:
                acc_s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            # Compute acc_s
            acc_s += tl.dot(q, tl.trans(k))

            # Apply masks
            # Trying to combine the three masks seem to make the result wrong
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                acc_s += tl.where(offs_n[None, :] < seqlen_k, 0, float("-inf"))
            if IS_CAUSAL:
                acc_s += tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), 0, float("-inf"))
            if HAS_MASK:
                acc_s += tl.where(mask, 0, float("-inf"))

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

            # Write back
            if not (EVEN_M & EVEN_N):
                tl.debug_barrier()
            if HAS_BIAS:
                if ACCUM_DBIAS:
                    acc_dbias += tl.sum(ds, axis=0)
                else:
                    if EVEN_M & EVEN_N:
                        tl.store(
                            db_ptrs,
                            ds,
                        )
                    else:
                        tl.store(
                            db_ptrs,
                            ds,
                            mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        )

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
            do_ptrs += BLOCK_M * stride_dom
            dq_ptrs += BLOCK_M * stride_dqm
            if HAS_BIAS:
                db_ptrs += BLOCK_M * stride_dbm
            q_ptrs += BLOCK_M * stride_qm
            if HAS_MASK:
                m_ptrs += BLOCK_M * stride_mm
            if HAS_BIAS:
                b_ptrs += BLOCK_M * stride_bm

    # Scale dk
    dk = (dk * softmax_scale).to(dk.dtype)

    # Write back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    if HAS_BIAS and ACCUM_DBIAS:
        if EVEN_N:
            tl.store(DBias + offs_n, acc_dbias)
        else:
            tl.store(DBias + offs_n, acc_dbias, mask=(offs_n < seqlen_k))

    if EVEN_N:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


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
            pre_hook=init_to_zero(["DQ", "DBias"]),
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero(["DQ", "DBias"]),
        ),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "IS_CAUSAL", "HAS_MASK", "HAS_BIAS", "HAS_INDICE", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        "ACCUM_DBIAS": lambda args: args["HAS_BIAS"] and (args["stride_dbm"] == 0) and (args["seqlen_q"] > 1),
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Mask,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    DBias,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dbb,
    stride_dbh,
    stride_dbm,
    nheads,
    nheads_k,
    nheads_mask,
    nheads_bias,
    h_h_k_ratio,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    ACCUM_DBIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_hq = off_hb % nheads
    off_hk = off_hq // h_h_k_ratio
    if HAS_MASK:
        if nheads_mask == 1:
            off_hmask = 0
        elif nheads_mask == nheads_k:
            off_hmask = off_hk
        else:
            off_hmask = off_hq
    if HAS_BIAS:
        if nheads_bias == 1:
            off_hbbias = 0
        elif nheads_bias == nheads_k:
            off_hbbias = off_hk
        else:
            off_hbbias = off_hq

    # Advance offset pointers for batch and head
    Q += off_b * stride_qb + off_hq * stride_qh
    K += off_b * stride_kb + off_hk * stride_kh
    V += off_b * stride_vb + off_hk * stride_vh
    if HAS_MASK:
        Mask += off_b * stride_mb + off_hmask * stride_mh
    if HAS_BIAS:
        Bias += off_b * stride_bb + off_hbbias * stride_bh
    DO += off_b * stride_dob + off_hq * stride_doh
    DQ += off_b * stride_dqb + off_hq * stride_dqh
    DK += off_b * stride_dkb + off_hq * stride_dkh
    DV += off_b * stride_dvb + off_hq * stride_dvh
    if HAS_BIAS:
        DBias += off_b * stride_dbb + off_hq * stride_dbh
    # Advance pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded

    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Mask,
                Bias,
                DO,
                DQ,
                DK,
                DV,
                DBias,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_mm,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                stride_dbm,
                seqlen_q,
                seqlen_k,
                headdim,
                IS_CAUSAL=IS_CAUSAL,
                HAS_MASK=HAS_MASK,
                HAS_BIAS=HAS_BIAS,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                ATOMIC_ADD=False,
                ACCUM_DBIAS=ACCUM_DBIAS,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Mask,
            Bias,
            DO,
            DQ,
            DK,
            DV,
            DBias,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_mm,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            stride_dbm,
            seqlen_q,
            seqlen_k,
            headdim,
            IS_CAUSAL=IS_CAUSAL,
            HAS_MASK=HAS_MASK,
            HAS_BIAS=HAS_BIAS,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            ATOMIC_ADD=True,
            ACCUM_DBIAS=ACCUM_DBIAS,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def _flash_dmattn_forward(q, k, v, mask, bias, softmax_scale=None, is_causal=False):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    assert nheads % nheads_k == 0, "Number of Q heads must be divisible by KV heads for GQA/MQA"
    assert d <= 128, "FlashDynamicMaskAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    has_mask = mask is not None
    if has_mask:
        assert mask.dtype == torch.bool, "Only support bool"
        assert mask.is_cuda
        nheads_mask = mask.shape[1]
    else:
        nheads_mask = 1
        mask = torch.empty(0, device=q.device, dtype=torch.bool)

    has_bias = bias is not None
    if has_bias:
        assert bias.dtype == q.dtype, "Only support fp16 and bf16"
        assert bias.is_cuda
        nheads_bias = bias.shape[1]
    else:
        nheads_bias = 1
        bias = torch.empty(0, device=q.device, dtype=q.dtype)

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        mask,
        bias,
        o,
        lse,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        ((0 if (has_mask and mask.shape[0] == 1) else (mask.stride(0) if has_mask else 0))),
        ((0 if (has_mask and mask.shape[1] == 1) else (mask.stride(1) if has_mask else 0))),
        ((0 if (has_mask and mask.shape[2] == 1) else (mask.stride(2) if has_mask else 0))),
        ((0 if (has_bias and bias.shape[0] == 1) else (bias.stride(0) if has_bias else 0))),
        ((0 if (has_bias and bias.shape[1] == 1) else (bias.stride(1) if has_bias else 0))),
        ((0 if (has_bias and bias.shape[2] == 1) else (bias.stride(2) if has_bias else 0))),
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        nheads_k,
        nheads_mask,
        nheads_bias,
        nheads // nheads_k,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=is_causal, HAS_MASK=has_mask, HAS_BIAS=has_bias, BLOCK_HEADDIM=d,
        is_causal,
        has_mask,
        has_bias,
        BLOCK_HEADDIM,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


def _flash_dmattn_backward(
    do, q, k, v, mask, bias, o, lse, softmax_scale=None, is_causal=False
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, dk = k.shape

    assert nheads % nheads_k == 0, "Number of Q heads must be divisible by KV heads for GQA/MQA"
    assert d <= 128, "FlashDynamicMaskAttention only support head dimensions up to 128"
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    seqlen_k_rounded = math.ceil(seqlen_k / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)

    has_mask = mask is not None
    if has_mask:
        assert mask.dtype == torch.bool, "Only support bool"
        nheads_mask = mask.shape[1]
    else:
        nheads_mask = 1
        mask = torch.empty(0, device=q.device, dtype=torch.bool)

    has_bias = bias is not None
    if has_bias:
        assert bias.dtype == q.dtype, "Only support fp16 and bf16"
        nheads_bias = bias.shape[1]
    else:
        nheads_bias = 1
        bias = torch.empty(0, device=q.device, dtype=q.dtype)

    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    # delta = torch.zeros_like(lse)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbias = torch.empty_like(bias) if has_bias else torch.empty(0, device=q.device, dtype=q.dtype)

    dk_expanded = torch.empty(batch, seqlen_k, nheads, d, device=q.device, dtype=q.dtype) if nheads != nheads_k else dk
    dv_expanded = torch.empty(batch, seqlen_k, nheads, d, device=q.device, dtype=q.dtype) if nheads != nheads_k else dv
    if has_bias:
        if (
            nheads_bias != nheads
            or ((bias.shape[0] == 1) and (batch > 1))
            or ((bias.shape[-2] == 1) and (seqlen_q > 1))
        ):
            if bias.shape[-2] == 1:
                dbias_expanded = torch.zeros(batch, nheads, 1, seqlen_k_rounded, device=q.device, dtype=dbias.dtype)
            else:
                dbias_expanded = torch.zeros(batch, nheads, seqlen_q, seqlen_k_rounded, device=q.device, dtype=dbias.dtype)
        else:
            dbias_expanded = dbias
    else:
        dbias_expanded = dbias

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=64,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        mask,
        bias,
        do,
        dq_accum,
        dk_expanded,
        dv_expanded,
        dbias_expanded,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        ((0 if (has_mask and mask.shape[0] == 1) else (mask.stride(0) if has_mask else 0))),
        ((0 if (has_mask and mask.shape[1] == 1) else (mask.stride(1) if has_mask else 0))),
        ((0 if (has_mask and mask.shape[2] == 1) else (mask.stride(2) if has_mask else 0))),
        ((0 if (has_bias and bias.shape[0] == 1) else (bias.stride(0) if has_bias else 0))),
        ((0 if (has_bias and bias.shape[1] == 1) else (bias.stride(1) if has_bias else 0))),
        ((0 if (has_bias and bias.shape[2] == 1) else (bias.stride(2) if has_bias else 0))),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk_expanded.stride(0),
        dk_expanded.stride(2),
        dk_expanded.stride(1),
        dv_expanded.stride(0),
        dv_expanded.stride(2),
        dv_expanded.stride(1),
        (dbias_expanded.stride(0) if has_bias else 0),
        (dbias_expanded.stride(1) if has_bias else 0),
        ((0 if (has_bias and bias.shape[-2] == 1) else (dbias_expanded.stride(2) if has_bias else 0))),
        nheads,
        nheads_k,
        nheads_mask,
        nheads_bias,
        nheads // nheads_k,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=is_causal, HAS_MASK=has_mask, HAS_BIAS=has_bias, BLOCK_HEADDIM=BLOCK_HEADDIM,
        is_causal,
        has_mask,
        has_bias,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq = dq_accum.to(q.dtype)
    if nheads != nheads_k:
        dk = dk_expanded.view(batch, seqlen_k, nheads_k, nheads // nheads_k, d).sum(dim=3)
        dv = dv_expanded.view(batch, seqlen_k, nheads_k, nheads // nheads_k, d).sum(dim=3)
    if has_bias:
        if (
            nheads_bias != nheads
            and bias.shape[0] == batch
            and bias.shape[-2] == seqlen_q
        ):
            dbias = dbias_expanded.view(batch, nheads_bias, nheads // nheads_bias, seqlen_q, seqlen_k_rounded).sum(dim=2)
        else:
            if bias.shape[-2] == 1:
                dbias_expanded = dbias_expanded.view(batch, nheads_bias, nheads // nheads_bias, 1, seqlen_k_rounded).sum(dim=2)
            else:
                dbias_expanded = dbias_expanded.view(batch, nheads_bias, nheads // nheads_bias, seqlen_q, seqlen_k_rounded).sum(dim=2)
            if bias.shape[0] == 1:
                dbias_expanded = dbias_expanded.sum(dim=0, keepdim=True)
            dbias.copy_(dbias_expanded)
    return dq, dk, dv, dbias if has_bias else None


def maybe_contiguous(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


class FlashDMAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, attn_mask=None, attn_bias=None, is_causal=False, softmax_scale=None):
        """
        query: (batch_size, seqlen_q, nheads, headdim)
        key: (batch_size, seqlen_k, nheads, headdim)
        value: (batch_size, seqlen_k, nheads, headdim)
        attn_mask: optional, (batch, nheads, seqlen_q, seqlen_k)
        attn_bias: optional, (batch, nheads, seqlen_q, seqlen_k)
        is_causal: bool, whether to apply causal masking
        softmax_scale: float, scaling factor for attention scores
        """

        # Make sure that the last dimension is contiguous
        query, key, value, attn_mask, attn_bias = [maybe_contiguous(x) for x in [query, key, value, attn_mask, attn_bias]]

        # Padding to multiple of 8 for 16-bit memory allocations
        head_size_og = query.size(3)
        if head_size_og % 8 != 0:
            query = torch.nn.functional.pad(query, [0, 8 - head_size_og % 8])
            key = torch.nn.functional.pad(key, [0, 8 - head_size_og % 8])
            value = torch.nn.functional.pad(value, [0, 8 - head_size_og % 8])
        seqlen_k_rounded = round_multiple(key.shape[1], 128)
        if attn_mask is not None and attn_mask.shape[-1] != seqlen_k_rounded:
            if attn_mask.shape[-1] == 1:
                attn_mask = attn_mask.expand(*attn_mask.shape[:-1], seqlen_k_rounded)
            else:
                attn_mask = torch.nn.functional.pad(attn_mask, [0, seqlen_k_rounded - attn_mask.shape[-1]])
        if attn_bias is not None and attn_bias.shape[-1] != seqlen_k_rounded:
            if attn_bias.shape[-1] == 1:
                attn_bias = attn_bias.expand(*attn_bias.shape[:-1], seqlen_k_rounded)
            else:
                attn_bias = torch.nn.functional.pad(attn_bias, [0, seqlen_k_rounded - attn_bias.shape[-1]])

        o, lse, ctx.softmax_scale = _flash_dmattn_forward(
            query,
            key,
            value,
            attn_mask,
            attn_bias,
            softmax_scale=softmax_scale,
            is_causal=is_causal
        )
        ctx.save_for_backward(query, key, value, o, lse, attn_mask, attn_bias)
        ctx.is_causal = is_causal
        ctx.seqlen_k_bias_og = attn_bias.shape[-1] if attn_bias is not None else 0
        return o

    @staticmethod
    def backward(ctx, do):
        query, key, value, o, lse, attn_mask, attn_bias = ctx.saved_tensors

        head_size_og = do.size(3)
        do_padded = do
        if head_size_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_og % 8])

        dq, dk, dv, dbias = _flash_dmattn_backward(
            do_padded,
            query,
            key,
            value,
            attn_mask,
            attn_bias,
            o,
            lse,
            softmax_scale=ctx.softmax_scale,
            is_causal=ctx.is_causal,
        )

        # We could have padded the head dimension
        dq = dq[..., : do.shape[-1]]  
        dk = dk[..., : do.shape[-1]]
        dv = dv[..., : do.shape[-1]]

        if dbias is not None:
            dbias = dbias[..., :key.shape[1]].sum(dim=-1, keepdim=True) if ctx.seqlen_k_bias_og == 1 else dbias[..., : key.shape[1]]

        return dq, dk, dv, None, dbias, None, None


def triton_dmattn_func(query, key, value, attn_mask=None, attn_bias=None, is_causal=False, softmax_scale=None):
    return FlashDMAttnFunc.apply(query, key, value, attn_mask, attn_bias, is_causal, softmax_scale)
