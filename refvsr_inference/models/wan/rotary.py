"""Rotary position embedding of Wan queries and keys.

The eager form is the reference. On CUDA a fused kernel computes the same values
in one pass over the tensor instead of several strided ones.
"""

import torch
import triton
import triton.language as tl


def apply_wan_rotary_eager(hidden_states, rotary_emb):
    freqs_cos, freqs_sin = rotary_emb
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cos[..., 0::2], freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


@triton.jit
def _rotate(
    X,
    COS,
    SIN,
    OUT,
    ROWS,
    ZERO,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    valid = row < ROWS
    token = (row // HEADS) % TOKENS
    d = tl.arange(0, D)
    x = tl.load(X + row[:, None] * D + d[None, :], valid[:, None], 0)
    pair = tl.load(X + row[:, None] * D + (d ^ 1)[None, :], valid[:, None], 0)
    # Channel pair (2i, 2i+1) uses the cosine at 2i and the sine at 2i+1, as the eager form.
    cos = tl.load(COS + token[:, None] * D + (d & -2)[None, :], valid[:, None], 0)
    sin = tl.load(SIN + token[:, None] * D + (d | 1)[None, :], valid[:, None], 0)
    # The eager form rounds both fp32 products before adding them. The compiler contracts
    # the multiply and the add into an FMA even with enable_fp_fusion=False, which is off by
    # one ulp for about one element in 1e5. XOR with a zero it cannot see keeps them apart.
    direct = (x.to(tl.float32) * cos).to(tl.int32, bitcast=True) ^ ZERO
    cross = (pair.to(tl.float32) * sin).to(tl.int32, bitcast=True) ^ ZERO
    direct = direct.to(tl.float32, bitcast=True)
    cross = cross.to(tl.float32, bitcast=True)
    even = (d % 2 == 0)[None, :]
    result = tl.where(even, direct - cross, cross + direct)
    tl.store(OUT + row[:, None] * D + d[None, :], result.to(x.dtype), valid[:, None])


def _fused_supported(hidden_states, freqs_cos, freqs_sin):
    if hidden_states.ndim != 4:
        return False
    _, tokens, _, dim = hidden_states.shape
    return (
        hidden_states.is_cuda
        and hidden_states.dtype in (torch.bfloat16, torch.float16)
        and hidden_states.is_contiguous()
        and hidden_states.numel() > 0
        and dim % 2 == 0
        # The kernel reproduces the promotion of a half-precision tensor with fp32 tables.
        and freqs_cos.dtype == torch.float32
        and freqs_sin.dtype == torch.float32
        and tuple(freqs_cos.shape) == (1, tokens, 1, dim)
        and tuple(freqs_sin.shape) == (1, tokens, 1, dim)
        and freqs_cos.device == hidden_states.device
        and freqs_sin.device == hidden_states.device
        and not hidden_states.requires_grad
    )


def apply_wan_rotary(hidden_states, rotary_emb):
    """Rotate [B,tokens,heads,dim] by per-token tables [1,tokens,1,dim]."""
    freqs_cos, freqs_sin = rotary_emb
    if not _fused_supported(hidden_states, freqs_cos, freqs_sin):
        return apply_wan_rotary_eager(hidden_states, rotary_emb)
    batch, tokens, heads, dim = hidden_states.shape
    out = torch.empty_like(hidden_states)
    rows = batch * tokens * heads
    block_rows = 32
    with torch.cuda.device(hidden_states.device):
        _rotate[(triton.cdiv(rows, block_rows),)](
            hidden_states,
            freqs_cos.contiguous(),
            freqs_sin.contiguous(),
            out,
            rows,
            0,
            tokens,
            heads,
            dim,
            block_rows,
            enable_fp_fusion=False,
        )
    return out
