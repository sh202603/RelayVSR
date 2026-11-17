"""Load separate Wan and Flash safetensors checkpoints."""

import json

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file

from .models.flash.model import FlashVSRConditionNet
from .models.wan.transformer import WanTransformer3DModel

# Metadata ``format`` of each checkpoint to its upscaling factor. The 2x pair is
# lightx2v/RelayVSR_X2; both files of a pair must have the same factor.
WAN_FORMATS = {"refvsr-wan-v1": 4, "relayvsr-wan-x2-v1": 2}
FLASH_FORMATS = {"refvsr-flash-v1": 4, "relayvsr-flash-x2-v1": 2}


@torch.no_grad()
def load_checkpoints(wan_path, flash_path, *, device="cuda", dtype=torch.bfloat16):
    """Wan includes its prompt; both files include their architecture metadata.

    The factor (4x or 2x) follows from the files; ``flash.upscale_factor`` holds it.
    """
    with safe_open(str(wan_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        wan_scale = WAN_FORMATS.get(metadata.get("format"))
        if wan_scale is None:
            raise ValueError("Expected a RelayVSR Wan safetensors checkpoint.")
        wan_config = json.loads(metadata["config"])
        # The 2x checkpoint repeats its factor in the config; the model does not take it.
        if wan_config.pop("upscale_factor", wan_scale) != wan_scale:
            raise ValueError("Wan checkpoint format and upscale_factor disagree.")
        prompt_shape = handle.get_slice("prompt_embeds").get_shape()
        if (
            len(prompt_shape) != 3
            or prompt_shape[0] != 1
            or prompt_shape[1] < 1
            or prompt_shape[2] != wan_config["text_dim"]
        ):
            raise ValueError("Checkpoint prompt shape does not match the Wan model.")
    with safe_open(str(flash_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        flash_scale = FLASH_FORMATS.get(metadata.get("format"))
        if flash_scale is None:
            raise ValueError("Expected a RelayVSR Flash safetensors checkpoint.")
        flash_config = json.loads(metadata["config"])
    if (
        flash_config.get("condition_mode") != "endpoints"
        or flash_config.get("temporal_rope_policy") != "current_relative"
    ):
        raise ValueError("Flash checkpoint must use endpoints and current_relative RoPE.")
    if flash_config.setdefault("upscale_factor", flash_scale) != flash_scale:
        raise ValueError("Flash checkpoint format and upscale_factor disagree.")
    if wan_scale != flash_scale:
        raise ValueError(
            f"The checkpoints are for different factors: {wan_path} is {wan_scale}x, "
            f"{flash_path} is {flash_scale}x."
        )
    variant = flash_config.pop("variant", "S")
    with init_empty_weights():
        wan = WanTransformer3DModel.from_config(wan_config)
        wan.lq_proj = wan.build_lq_proj(wan_scale)
        wan.register_buffer("prompt_embeds", torch.empty(prompt_shape))
        flash = FlashVSRConditionNet.from_variant(variant, **flash_config)
    wan.load_state_dict(load_file(str(wan_path), device="cpu"), strict=True, assign=True)
    flash.load_state_dict(load_file(str(flash_path), device="cpu"), strict=True, assign=True)
    wan = wan.to(device=device, dtype=dtype).eval().requires_grad_(False)
    flash = flash.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return wan, flash
