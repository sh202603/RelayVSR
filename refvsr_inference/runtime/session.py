"""Build a ready stream from two checkpoints; shared by the CLI and embedding callers."""

from __future__ import annotations

import torch

from ..checkpoints import load_checkpoints
from ..models.wan.fp8 import convert_wan_fp8, release_cross_attention_kv
from ..models.wan.sampling import WanVSRPipeline
from .streaming import CollaborativeFrameStream

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@torch.no_grad()
def build_stream(
    wan_checkpoint,
    flash_checkpoint,
    *,
    device="cuda",
    dtype="bf16",
    window_size=16,
    seed=42,
    kv_cache="auto",
    kv_cache_dtype="native",
    wan_weights="bf16",
) -> CollaborativeFrameStream:
    """Load both models and return a stream whose ``frames()`` can be called per video.

    The options mean what the CLI options of the same names mean. ``stream.scale``
    is the factor of the checkpoints (4 or 2) and ``stream.wan_weights`` the
    weights in use: ``bf16`` when FP8 was requested but is unavailable.
    """
    if wan_weights not in ("bf16", "fp8"):
        raise ValueError("wan_weights must be bf16 or fp8.")
    if wan_weights == "fp8" and dtype != "bf16":
        raise ValueError("--wan-weights fp8 requires --dtype bf16.")
    device = torch.device(device)
    compute_dtype = DTYPES[dtype]
    wan, flash = load_checkpoints(
        wan_checkpoint, flash_checkpoint, device=device, dtype=compute_dtype
    )
    sampler = WanVSRPipeline(
        scheduler_config={"num_train_timesteps": 1000, "shift": 5.0},
        sampling={
            "execution_mode": "streaming",
            "num_inference_steps": 1,
            "timesteps": [1000],
            "guidance_scale": 1.0,
            "seed": seed,
            "kv_cache_window_size": wan.temporal_context_frames,
        },
    )
    sampler.prepare_streaming(
        model=wan, prompt_embeds=wan.prompt_embeds, device=device, compute_dtype=compute_dtype
    )
    # The text K/V are cached now, so their projections can leave the GPU.
    release_cross_attention_kv(wan)
    # FP8 comes last: a later dtype cast would turn the weights back.
    resolved_weights = "fp8" if wan_weights == "fp8" and convert_wan_fp8(wan) else "bf16"
    stream = CollaborativeFrameStream(
        wan_model=wan,
        flash_model=flash,
        wan_pipeline=sampler,
        prompt_embeds=wan.prompt_embeds,
        device=device,
        compute_dtype=compute_dtype,
        window_size=window_size,
        seed=seed,
        wan_kv_cache=kv_cache,
        wan_kv_cache_dtype=kv_cache_dtype,
    )
    stream.wan_weights = resolved_weights
    return stream


def to_uint8(rgb: torch.Tensor) -> torch.Tensor:
    """An ``OutputFrame.rgb`` ([1,3,H,W] in [-1,1]) as uint8 [H,W,3] on its device."""
    return rgb[0].add(1).mul_(127.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
