# Copyright (c) 2026, Jingze Shi.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import _core
from triton.experimental.gluon.language.nvidia.hopper import (
    warpgroup_mma_wait as _warpgroup_mma_wait,
)


@_core.builtin
def warpgroup_mma_wait(num_outstanding=0, deps=None, _semantic=None):
    # Wait for each warpgroup's own MMA work
    result = _warpgroup_mma_wait(num_outstanding=num_outstanding, deps=deps, _semantic=_semantic)
    results = result if isinstance(result, tuple) else (result,)
    for value in results:
        value.handle.set_attr("warpGroupLocal", _semantic.builder.get_unit_attr())
    return result


@gluon.jit
def pack_mma_operand(rA, dtype: gl.constexpr, layout: gl.constexpr):
    gl.static_assert(dtype == gl.bfloat16 or dtype == gl.float16 or dtype.is_fp8())
    rA = gl.convert_layout(rA, layout)
    TILE_M: gl.constexpr = rA.shape[0]
    TILE_N: gl.constexpr = rA.shape[1]
    if gl.constexpr(dtype.is_fp8()):
        rA = rA.to(dtype).to(gl.uint8, bitcast=True)
        a, b = rA.reshape([TILE_M, TILE_N // 4, 2, 2]).split()
        a0, a1 = a.split()
        b0, b1 = b.split()
        packed = (
            a0.to(gl.uint32)
            | (b0.to(gl.uint32) << 8)
            | (a1.to(gl.uint32) << 16)
            | (b1.to(gl.uint32) << 24)
        )
    else:
        a, b = rA.reshape([TILE_M, TILE_N // 2, 2]).split()
        asm: gl.constexpr = (
            "cvt.rn.bf16x2.f32 $0, $2, $1;"
            if dtype == gl.bfloat16
            else "cvt.rn.f16x2.f32 $0, $2, $1;"
        )
        packed = gl.inline_asm_elementwise(
            asm,
            constraints="=r,f,f",
            args=[a, b],
            dtype=gl.uint32,
            is_pure=True,
            pack=1,
        )
    return packed


@gluon.jit
def unpack_mma_operand(rA, dtype: gl.constexpr, layout: gl.constexpr):
    TILE_M: gl.constexpr = rA.shape[0]
    TILE_N: gl.constexpr = rA.shape[1] * 2
    a, b = gl.inline_asm_elementwise(
        "mov.b32 {$0, $1}, $2;",
        constraints="=h,=h,r",
        args=[rA],
        dtype=(gl.uint16, gl.uint16),
        is_pure=True,
        pack=1,
    )
    a = a.to(dtype, bitcast=True)
    b = b.to(dtype, bitcast=True)
    rA = gl.join(a, b)
    return gl.convert_layout(rA.reshape([TILE_M, TILE_N]), layout)
