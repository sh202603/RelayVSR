"""FP8 (E4M3) linears for the Wan transformer blocks.

Weights are quantized once with a per-tensor scale; activations get a dynamic
per-tensor scale on every call. The product runs in cuBLASLt through
``torch._scaled_mm`` and returns BF16. This halves the block weights on the GPU
and changes the output slightly, so it is opt-in.

The quantization kernels and the linear are adapted from flashvsr-sm89-ops
(https://github.com/aireet/flashvsr-sm89-ops, Apache-2.0) as vendored in
FlashVSR_plus; the module selection is rewritten for this model.
"""

import warnings

import torch
import torch.nn as nn
import triton
import triton.language as tl

FP8_MAX = 448.0  # largest finite float8_e4m3fn value


@triton.jit
def _absmax_kernel(x_ptr, amax_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(x), axis=0))


@triton.jit
def _quant_kernel(x_ptr, amax_ptr, out_ptr, n, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    s = tl.maximum(tl.load(amax_ptr), 1e-12) / FP8_MAX
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Saturate: an out-of-range value would become NaN in e4m3fn.
    v = tl.minimum(tl.maximum(x / s, -FP8_MAX), FP8_MAX)
    tl.store(out_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def quantize_fp8(x2: torch.Tensor):
    """Contiguous [M, K] -> (FP8 codes, FP32 scale), without a host sync."""
    n = x2.numel()
    amax = torch.zeros(1, dtype=torch.float32, device=x2.device)
    block = 8192 if n >= 8192 else triton.next_power_of_2(max(n, 1))
    grid = (triton.cdiv(n, block),)
    warps = 4 if block <= 2048 else 8
    _absmax_kernel[grid](x2, amax, n, BLOCK=block, num_warps=warps)
    x8 = torch.empty(x2.shape, dtype=torch.float8_e4m3fn, device=x2.device)
    _quant_kernel[grid](x2, amax, x8, n, FP8_MAX=FP8_MAX, BLOCK=block, num_warps=warps)
    return x8, amax.clamp_min(1e-12) / FP8_MAX


class FP8Linear(nn.Module):
    """Drop-in for an ``nn.Linear`` on a CUDA BF16 path."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.detach()
        scale = (weight.abs().amax().float() / FP8_MAX).clamp(min=1e-12)
        self.register_buffer("w_scale", scale, persistent=False)
        self.register_buffer(
            "weight", (weight.float() / scale).to(torch.float8_e4m3fn), persistent=False
        )
        bias = None if linear.bias is None else linear.bias.detach().to(torch.bfloat16)
        self.register_buffer("bias", bias, persistent=False)
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def _apply(self, fn, recurse=True):
        # ``module.to(dtype=...)`` would silently cast the FP8 weight back.
        dtypes = (self.weight.dtype, self.w_scale.dtype)
        super()._apply(fn, recurse)
        if (self.weight.dtype, self.w_scale.dtype) != dtypes:
            raise TypeError("FP8Linear cannot change dtype; convert after the final cast.")
        return self

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        x8, scale = quantize_fp8(x2.contiguous())
        y = torch._scaled_mm(
            x8,
            self.weight.t(),
            scale_a=scale,
            scale_b=self.w_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
        )
        return y.reshape(*shape[:-1], self.out_features)


def _probe(device):
    """Raise unless this GPU and cuBLASLt build provide the FP8 product."""
    if torch.cuda.get_device_capability(device) < (8, 9):
        raise RuntimeError("FP8 needs an sm89+ GPU (RTX 40 series or newer)")
    linear = nn.Linear(64, 48, device=device, dtype=torch.bfloat16)
    out = FP8Linear(linear)(torch.randn(40, 64, device=device, dtype=torch.bfloat16))
    if not torch.isfinite(out.float()).all():
        raise RuntimeError("FP8 product returned non-finite values")


def _targets(block):
    """(parent, attribute) of every linear evaluated per keyframe in one block.

    The cross-attention K/V projections are skipped: they only build the static
    text cache, which is computed before conversion.
    """
    yield from ((block.attn1, name) for name in ("to_q", "to_k", "to_v"))
    yield block.attn1.to_out, "0"
    yield block.attn2, "to_q"
    yield block.attn2.to_out, "0"
    yield block.ffn.net[0], "proj"
    yield block.ffn.net, "2"


@torch.no_grad()
def convert_wan_fp8(wan) -> int:
    """Replace the block linears of a loaded BF16 CUDA model in place.

    Returns the number of converted layers, or 0 after a warning when FP8 is
    unavailable; the model is then left untouched.
    """
    parameter = next(wan.parameters())
    try:
        if parameter.device.type != "cuda" or parameter.dtype != torch.bfloat16:
            raise RuntimeError("FP8 needs a CUDA model in bf16")
        _probe(parameter.device)
        pending = [target for block in wan.blocks for target in _targets(block)]
        if not all(isinstance(getattr(p, n), nn.Linear) for p, n in pending):
            raise RuntimeError("unexpected Wan block layout")
    except Exception as error:  # noqa: BLE001 - any failure means "keep bf16"
        warnings.warn(
            f"FP8 Wan weights disabled ({type(error).__name__}: {error}); using bf16.",
            stacklevel=2,
        )
        return 0
    for parent, name in pending:
        # One layer at a time, so bf16 and FP8 copies of the model never coexist.
        setattr(parent, name, FP8Linear(getattr(parent, name)))
    torch.cuda.empty_cache()
    return len(pending)


def release_cross_attention_kv(wan) -> None:
    """Move the text K/V projections to host memory once their cache exists.

    Streaming reads the projected text K/V from the conditioning cache, so
    these weights are idle on the GPU. The output is unaffected.
    """
    for block in wan.blocks:
        for name in ("to_k", "to_v"):
            getattr(block.attn2, name).to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
