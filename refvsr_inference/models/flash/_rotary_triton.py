"""Fused 3D-RoPE for inference (forward only).

The eager form in ``_flashvsr_ops`` is the reference: it makes several strided passes
over the tensor. This kernel computes the same values in one pass.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(
    X,
    COS,
    SIN,
    OUT,
    ROWS,
    TOKENS,
    HEADS,
    xs0,
    xs1,
    xs2,
    xs3,
    D: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    valid = row < ROWS
    token = row % TOKENS
    head = (row // TOKENS) % HEADS
    batch = row // (TOKENS * HEADS)
    d = tl.arange(0, D)
    # The input keeps its own strides: a projected query is a transposed view.
    base = batch * xs0 + head * xs1 + token * xs2
    x = tl.load(X + base[:, None] + d[None, :] * xs3, valid[:, None], 0)
    pair = tl.load(X + base[:, None] + (d ^ 1)[None, :] * xs3, valid[:, None], 0)
    cos = tl.load(COS + token[:, None] * D + d[None, :], valid[:, None], 0)
    sin = tl.load(SIN + token[:, None] * D + d[None, :], valid[:, None], 0)
    # The eager form works in the tensor dtype, so each product is rounded before the sum.
    direct = (x.to(tl.float32) * cos.to(tl.float32)).to(x.dtype).to(tl.float32)
    cross = (pair.to(tl.float32) * sin.to(tl.float32)).to(x.dtype).to(tl.float32)
    even = (d % 2 == 0)[None, :]
    result = tl.where(even, direct - cross, cross + direct)
    tl.store(OUT + row[:, None] * D + d[None, :], result.to(x.dtype), valid[:, None])


@torch.no_grad()
def triton_rotary(tensor, cosines, sines):
    """Rotate [B,heads,tokens,D] by per-token tables [tokens,D]; the result is contiguous."""
    batch, heads, tokens, dim = tensor.shape
    output = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    rows = batch * heads * tokens
    block_rows = 32
    with torch.cuda.device(tensor.device):
        _rotate[(triton.cdiv(rows, block_rows),)](
            tensor,
            cosines.contiguous(),
            sines.contiguous(),
            output,
            rows,
            tokens,
            heads,
            *tensor.stride(),
            dim,
            block_rows,
            # A contracted multiply-add would skip the rounding of the products.
            enable_fp_fusion=False,
        )
    return output
