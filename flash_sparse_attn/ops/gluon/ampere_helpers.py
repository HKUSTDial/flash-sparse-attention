# Copyright (c) 2026, Jingze Shi.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2


@gluon.jit
def reshape_acc_to_frgA(acc):
    MMA_K: gl.constexpr = 16
    TILE_M: gl.constexpr = acc.shape[0]
    TILE_N: gl.constexpr = acc.shape[1]

    if TILE_N == MMA_K:
        return (acc,)
    elif TILE_N == 2 * MMA_K:
        a0, a1 = acc.reshape((TILE_M, 2, MMA_K)).permute(0, 2, 1).split()
        return (a0, a1)
    elif TILE_N == 4 * MMA_K:
        a02, a13 = (
            acc.reshape((TILE_M, 4, MMA_K))
            .permute(0, 2, 1)
            .reshape((TILE_M, MMA_K, 2, 2))
            .split()
        )
        a0, a2 = a02.split()
        a1, a3 = a13.split()
        return (a0, a1, a2, a3)
    elif TILE_N == 8 * MMA_K:
        a0246, a1357 = (
            acc.reshape((TILE_M, 8, MMA_K))
            .permute(0, 2, 1)
            .reshape((TILE_M, MMA_K, 2, 2, 2))
            .split()
        )
        a04, a26 = a0246.split()
        a15, a37 = a1357.split()
        a0, a4 = a04.split()
        a2, a6 = a26.split()
        a1, a5 = a15.split()
        a3, a7 = a37.split()
        return (a0, a1, a2, a3, a4, a5, a6, a7)
    else:
        gl.static_assert(
            False, "reshape_acc_to_frgA requires TILE_N in {16, 32, 64, 128}"
        )


@gluon.jit
def gemm(
    acc,
    sA,
    sB,
    lhs_layout: gl.constexpr,
    rhs_layout: gl.constexpr,
    hook_fn=None,
):
    MMA_K: gl.constexpr = 16
    TILE_K: gl.constexpr = sA.shape[1]
    rA = sA.slice(0, MMA_K, dim=1).load(lhs_layout)
    rB = sB.slice(0, MMA_K, dim=1).permute((1, 0)).load(rhs_layout)
    for k in gl.static_range(0, TILE_K, MMA_K):
        if k + MMA_K < TILE_K:
            rA_next = sA.slice(k + MMA_K, MMA_K, dim=1).load(lhs_layout)
            rB_next = sB.slice(k + MMA_K, MMA_K, dim=1).permute((1, 0)).load(rhs_layout)
        acc = mma_v2(rA, rB, acc)
        if k == 0 and hook_fn is not None:
            hook_fn()
        if k + MMA_K < TILE_K:
            rA = rA_next
            rB = rB_next
    return acc


@gluon.jit
def gemm_rs(
    acc,
    rA,
    sB,
    lhs_layout: gl.constexpr,
    rhs_layout: gl.constexpr,
    hook_fn=None,
):
    MMA_K: gl.constexpr = 16
    TILE_K: gl.constexpr = rA.shape[1]
    frgA = reshape_acc_to_frgA(rA)
    rA = gl.convert_layout(frgA[0].to(sB.dtype), lhs_layout)
    rB = sB.slice(0, MMA_K, dim=0).load(rhs_layout)
    for k in gl.static_range(0, TILE_K, MMA_K):
        if k + MMA_K < TILE_K:
            rA_next = gl.convert_layout(frgA[k // MMA_K + 1].to(sB.dtype), lhs_layout)
            rB_next = sB.slice(k + MMA_K, MMA_K, dim=0).load(rhs_layout)
        acc = mma_v2(rA, rB, acc)
        if k == 0 and hook_fn is not None:
            hook_fn()
        if k + MMA_K < TILE_K:
            rA = rA_next
            rB = rB_next
    return acc
