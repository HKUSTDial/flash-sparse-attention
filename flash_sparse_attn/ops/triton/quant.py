import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_fp8_kernel(
    X,
    Out,
    Scale,
    n_elements,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)

    local_amax = tl.max(tl.abs(x))
    tl.atomic_max(Scale, local_amax, sem="relaxed")

    scale = tl.load(Scale)
    invscale = FP8_MAX / tl.maximum(scale, 1e-12)

    x_fp8 = (x * invscale).to(FP8_DTYPE)
    tl.store(Out + offsets, x_fp8, mask=mask)


def quantize_fp8(x: torch.Tensor, dtype: torch.dtype = torch.float8_e5m2):
    assert x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert dtype in (torch.float8_e5m2, torch.float8_e4m3fn)
    tl_dtype = tl.float8e5 if dtype == torch.float8_e5m2 else tl.float8e4nv
    FP8_MAX = 57344.0 if dtype == torch.float8_e5m2 else 448.0

    n_elements = x.numel()
    scale = torch.zeros(1, device=x.device, dtype=torch.float32)
    x_fp8 = torch.empty_like(x, dtype=dtype)

    BLOCK = 512

    def grid(META):
        return (triton.cdiv(n_elements, META["BLOCK"]),)

    _quantize_fp8_kernel[grid](
        x,
        x_fp8,
        scale,
        n_elements,
        BLOCK=BLOCK,
        FP8_MAX=FP8_MAX,
        FP8_DTYPE=tl_dtype,
    )

    scale = scale / FP8_MAX

    return x_fp8, scale
