"""Private Transformer operations used by conditional FlashVSR."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class _KVCache:
    """Unrotated, normalized keys and values for one VSR layer."""

    key: torch.Tensor
    value: torch.Tensor


class _RotaryEmbedding3D(nn.Module):
    """Relative 3D rotary embeddings for temporal and spatial coordinates."""

    def __init__(
        self,
        *,
        temporal_dim: int = 16,
        height_dim: int = 24,
        width_dim: int = 24,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        dimensions = (temporal_dim, height_dim, width_dim)
        if any(dimension <= 0 or dimension % 2 != 0 for dimension in dimensions):
            raise ValueError(f"3D-RoPE dimensions must be positive and even: {dimensions}.")
        if theta <= 0:
            raise ValueError(f"RoPE theta must be positive, got {theta}.")

        self.head_dim = sum(dimensions)
        self._frequency_dimensions = dimensions
        self._frequency_theta = float(theta)
        self.register_buffer(
            "temporal_inv_freq",
            self._build_inv_freq(temporal_dim, theta),
            persistent=False,
        )
        self.register_buffer(
            "height_inv_freq",
            self._build_inv_freq(height_dim, theta),
            persistent=False,
        )
        self.register_buffer(
            "width_inv_freq",
            self._build_inv_freq(width_dim, theta),
            persistent=False,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Module.to()/half()/bfloat16() also cast nonpersistent buffers. Rebuild
        # from the frequency definition, never by upcasting rounded frequencies.
        # This also materializes valid constants after meta -> to_empty().
        for name, dimension in zip(
            ("temporal_inv_freq", "height_inv_freq", "width_inv_freq"),
            self._frequency_dimensions,
            strict=True,
        ):
            self._buffers[name] = self._build_inv_freq(
                dimension, self._frequency_theta, device=self._buffers[name].device
            )
        return self

    @staticmethod
    def _build_inv_freq(dimension: int, theta: float, *, device=None) -> torch.Tensor:
        indices = torch.arange(0, dimension, 2, dtype=torch.float32, device=device)
        return 1.0 / (theta ** (indices / dimension))

    def frequencies(
        self,
        temporal_positions: torch.Tensor,
        height_positions: torch.Tensor,
        width_positions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cosines = []
        sines = []
        for positions, inv_freq in (
            (temporal_positions, self.temporal_inv_freq),
            (height_positions, self.height_inv_freq),
            (width_positions, self.width_inv_freq),
        ):
            angles = positions.to(dtype=torch.float32).unsqueeze(-1) * inv_freq
            cosines.append(torch.cos(angles).repeat_interleave(2, dim=-1))
            sines.append(torch.sin(angles).repeat_interleave(2, dim=-1))
        return (
            torch.cat(cosines, dim=-1).to(dtype=dtype),
            torch.cat(sines, dim=-1).to(dtype=dtype),
        )

    @staticmethod
    def apply(
        tensor: torch.Tensor,
        cosines: torch.Tensor,
        sines: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate [..., tokens, D] by per-token tables [tokens, D].

        On CUDA a fused kernel computes the values of :meth:`apply_eager` in one pass.
        """
        if _fused_rotary_usable(tensor, cosines, sines):
            from ._rotary_triton import triton_rotary

            return triton_rotary(tensor, cosines, sines)
        return _RotaryEmbedding3D.apply_eager(tensor, cosines, sines)

    @staticmethod
    def apply_eager(
        tensor: torch.Tensor,
        cosines: torch.Tensor,
        sines: torch.Tensor,
    ) -> torch.Tensor:
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.empty_like(tensor)
        cosines = cosines.unsqueeze(0).unsqueeze(0)
        sines = sines.unsqueeze(0).unsqueeze(0)
        rotated[..., 0::2] = even * cosines[..., 0::2] - odd * sines[..., 0::2]
        rotated[..., 1::2] = even * sines[..., 1::2] + odd * cosines[..., 1::2]
        return rotated


_fused_rotary_checked: dict[tuple, bool] = {}


def _fused_rotary_usable(tensor: torch.Tensor, cosines: torch.Tensor, sines: torch.Tensor) -> bool:
    """Whether the fused kernel may replace the eager form for this call."""
    dim = tensor.shape[-1] if tensor.ndim == 4 else 0
    if not (
        tensor.is_cuda
        and tensor.dtype in (torch.bfloat16, torch.float16)
        and not tensor.requires_grad
        and 0 < tensor.numel() < 2**31
        # The kernel indexes the channels with a power-of-two range.
        and dim >= 2
        and dim & (dim - 1) == 0
        and all(
            table.dtype == tensor.dtype
            and table.device == tensor.device
            and tuple(table.shape) == tuple(tensor.shape[2:])
            and not table.requires_grad
            for table in (cosines, sines)
        )
    ):
        return False
    key = (tensor.device, tensor.dtype, dim)
    usable = _fused_rotary_checked.get(key)
    if usable is None:
        usable = _fused_rotary_checked[key] = _fused_rotary_matches_eager(*key)
    return usable


def _fused_rotary_matches_eager(device: torch.device, dtype: torch.dtype, dim: int) -> bool:
    """Compare the fused kernel with the eager form once per device, dtype and width.

    The rounding of the kernel depends on how it is compiled, and the fused pass is only
    an optimization: where it is not bit-identical, the eager form stays in use.
    """
    from ._rotary_triton import triton_rotary

    generator = torch.Generator().manual_seed(0)
    tokens, heads = 4096, 4
    # Strided like a projected query: [B,tokens,heads,D] viewed as [B,heads,tokens,D].
    sample = torch.randn(1, tokens, heads, dim, generator=generator).mul_(4)
    sample = sample.to(device=device, dtype=dtype).transpose(1, 2)
    angles = torch.rand(tokens, dim, generator=generator).mul_(2 * math.pi).to(device)
    cosines, sines = angles.cos().to(dtype), angles.sin().to(dtype)
    matches = torch.equal(
        triton_rotary(sample, cosines, sines),
        _RotaryEmbedding3D.apply_eager(sample, cosines, sines),
    )
    if not matches:
        warnings.warn(
            "The fused Flash RoPE kernel does not reproduce the eager form on this "
            "platform; using the eager form.",
            stacklevel=2,
        )
    return matches


class _SwiGLU(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.gate_up_proj = nn.Linear(
            dim,
            2 * hidden_dim,
            bias=False,
        )

        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


__all__ = ["_KVCache", "_RotaryEmbedding3D", "_SwiGLU"]
