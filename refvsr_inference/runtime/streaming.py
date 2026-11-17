"""Bounded-lookahead collaboration with immediate, ordered frame delivery."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

import torch
import torch.nn.functional as F

from .timing import DeviceSpan


@dataclass(frozen=True)
class InputFrame:
    index: int
    rgb: torch.Tensor  # uint8 CHW, native LR
    source_time: float = 0.0  # seconds relative to the session origin
    arrival_time: float = 0.0


@dataclass(frozen=True)
class OutputFrame:
    index: int
    rgb: torch.Tensor  # GPU/CPU float BCHW in [-1,1]
    source_time: float
    arrival_time: float
    window_start: int
    window_end: int
    timings: tuple[tuple[str, DeviceSpan], ...] = ()


def prepare_lr(frame: torch.Tensor, *, device, dtype, stride: int = 8):
    """Normalize native RGB and pad for the Wan latent/patch grid."""
    if frame.ndim != 3 or frame.shape[0] != 3 or frame.dtype != torch.uint8:
        raise ValueError("Streaming input must be uint8 RGB [3,H,W].")
    h, w = frame.shape[-2:]
    if min(h, w) <= 0:
        raise ValueError("Frame dimensions must be positive.")
    x = frame.unsqueeze(0).to(device=device, non_blocking=True).to(dtype)
    x = x.div(127.5).sub(1.0)
    return F.pad(x, (0, (-w) % stride, 0, (-h) % stride), mode="replicate")


class CollaborativeFrameStream:
    """Model-only generator shared by deployment and paced benchmarking.

    It buffers at most K native frames, retains four (configured) Wan keyframes,
    and resets Flash video history at each overlapping endpoint window. Yielding
    suspends before the next Flash decode, so downstream sees individual frames.
    """

    def __init__(
        self,
        *,
        wan_model,
        flash_model,
        wan_pipeline,
        prompt_embeds,
        device,
        compute_dtype,
        window_size=16,
        seed=42,
        reuse_endpoint_kv=True,
        precompute_rope=True,
        wan_kv_cache="gpu",
        wan_kv_cache_dtype="native",
    ):
        if wan_kv_cache not in ("gpu", "cpu", "auto"):
            raise ValueError("wan_kv_cache must be gpu, cpu or auto.")
        if wan_kv_cache_dtype not in ("native", "int8"):
            raise ValueError("wan_kv_cache_dtype must be native or int8.")
        if window_size < 2:
            raise ValueError("Collaborative window_size must be at least 2.")
        if flash_model.condition_mode != "endpoints":
            raise ValueError("Collaborative streaming requires endpoint conditions.")
        if flash_model.training or wan_model.training:
            raise ValueError("Streaming models must be in eval() mode.")
        self.wan = wan_model
        self.flash = flash_model
        # Output pixels per input pixel: 4, or 2 with the 2x checkpoints.
        self.scale = int(flash_model.upscale_factor)
        # Native pixels per Wan latent; the latent grid is 1/16 of the output.
        self.latent_stride = 16 // self.scale
        # The 4x Wan reads a 4x bilinear upsample of the frame, the 2x Wan the frame itself.
        self.wan_input_scale = 4 if self.scale == 4 else 1
        self.pipeline = wan_pipeline
        self.prompt = prompt_embeds
        self.device = torch.device(device)
        self.dtype = compute_dtype
        self.window_size = int(window_size)
        self.seed = seed
        self.reuse_endpoint_kv = reuse_endpoint_kv
        self.precompute_rope = precompute_rope
        self.wan_kv_cache = wan_kv_cache
        self.wan_kv_cache_dtype = wan_kv_cache_dtype
        self.resolved_wan_kv_cache = None
        self.last_keyframe_indices: deque[int] = deque(maxlen=10000)
        self.keyframe_count = 0
        self.capture_state = False

    @property
    def stride(self):
        """Native pixels per Wan token; frames are padded to a multiple of it."""
        patch = self.wan.config.patch_size
        return self.latent_stride * max(int(patch[1]), int(patch[2]))

    @torch.no_grad()
    def prepare(self, height, width):
        stride = self.stride
        geometry = (
            height + (-height) % stride,
            width + (-width) % stride,
            self.window_size,
            self.dtype,
            self.device,
        )
        if self.precompute_rope and getattr(self, "_prepared_geometry", None) != geometry:
            self.flash.prepare_streaming(geometry[0], geometry[1], self.window_size)
            self._prepared_geometry = geometry

    def wan_kv_cache_bytes(self, padded_height, padded_width):
        """Size of the Wan self-attention K/V rings for one native LR geometry."""
        config = self.wan.config
        _, patch_h, patch_w = (int(v) for v in config.patch_size)
        # Wan sees the output through a 16x latent grid: LR / 4 per latent at 4x, LR / 8 at 2x.
        latent = self.latent_stride
        tokens = (padded_height // (latent * patch_h)) * (padded_width // (latent * patch_w))
        window = self.wan.temporal_context_frames
        window = 4 if window == -1 else window
        dim = int(config.num_attention_heads) * int(config.attention_head_dim)
        item = torch.empty((), dtype=self.dtype).element_size()
        layer = 2 * window * tokens * dim * item
        if self.wan_kv_cache_dtype == "int8":
            # int8 codes for every layer, plus the one shared window in the compute dtype.
            return len(self.wan.blocks) * 2 * window * tokens * dim + layer
        return len(self.wan.blocks) * layer

    def resolve_wan_kv_cache(self, padded_height, padded_width):
        """Pick the K/V location; ``auto`` keeps it on the GPU only when it fits."""
        if self.wan_kv_cache != "auto":
            return self.wan_kv_cache
        if self.device.type != "cuda":
            return "gpu"
        free, _ = torch.cuda.mem_get_info(self.device)
        free += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        # Headroom for Wan activations and the Flash decoder, which grows with the output.
        headroom = 2**30 + 320 * (self.scale * padded_height) * (self.scale * padded_width)
        needed = self.wan_kv_cache_bytes(padded_height, padded_width)
        return "gpu" if needed + headroom <= free else "cpu"

    def reset(self):
        """Drop the state kept for :meth:`checkpoint`, which holds the K/V rings of the
        last stream. ``frames()`` calls this itself; call it after a finished stream to
        give the memory back before the next one starts."""
        self._checkpoint_source = None

    def checkpoint(self) -> dict:
        """State needed to continue after the frame at the current window end.

        Valid once the frame whose index equals its ``window_end`` has been
        yielded and until the generator is advanced again. Continuing needs
        only the Wan rings, the endpoint latent and the noise generator: Flash
        restarts its history at every window.
        """
        point = getattr(self, "_checkpoint_source", None)
        if point is None:
            raise RuntimeError("No window is in progress.")
        index, latent, wan_state, generator = point
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "format": "refvsr-stream-v1",
            "frame": index,
            "keyframe_count": self.keyframe_count,
            "latent": latent.cpu(),
            "generator": None if generator is None else generator[0].get_state().cpu(),
            "wan": wan_state.model_state.export(),
        }

    @torch.no_grad()
    def frames(
        self, source: Iterable[InputFrame], *, resume: dict | None = None
    ) -> Iterator[OutputFrame]:
        """Yield super-resolved frames in order.

        With ``resume`` (from :meth:`checkpoint`), ``source`` must start at the
        checkpointed frame; that frame is consumed again but not yielded again.
        """
        iterator = iter(source)
        first = next(iterator, None)
        if first is None:
            raise ValueError("Cannot infer an empty stream.")
        if resume is not None and resume.get("format") != "refvsr-stream-v1":
            raise ValueError("Unknown stream checkpoint.")
        if first.index != (0 if resume is None else resume["frame"]):
            raise ValueError("The stream must start at frame 0 or at the checkpointed frame.")
        shape = tuple(first.rgb.shape)
        h, w = shape[-2:]
        stride, scale = self.stride, self.scale
        ph, pw = h + (-h) % stride, w + (-w) % stride
        # Before this stream measures the free memory and allocates its own rings.
        self.reset()
        self.prepare(h, w)
        self.resolved_wan_kv_cache = self.resolve_wan_kv_cache(ph, pw)
        wan_state = self.pipeline.create_streaming_state(
            self.wan,
            kv_cache_device=self.resolved_wan_kv_cache,
            kv_cache_dtype=self.wan_kv_cache_dtype,
        )
        generator = (
            None
            if self.seed is None
            else [torch.Generator(device=self.device).manual_seed(self.seed)]
        )
        self.last_keyframe_indices.clear()
        self.keyframe_count = 0

        pending_timings = []

        def keyframe(packet):
            span = DeviceSpan(self.device)
            native = prepare_lr(packet.rgb, device=self.device, dtype=self.dtype, stride=stride)
            upsampled = native
            if self.wan_input_scale != 1:
                upsampled = F.interpolate(
                    native, scale_factor=self.wan_input_scale, mode="bilinear", align_corners=False
                )
            pending_timings.append(("wan_input", span.finish()))
            span = DeviceSpan(self.device)
            latent = self.pipeline.sample_streaming_latent_step(
                model=self.wan,
                lq_frame=upsampled.unsqueeze(2),
                spatial_scale_factor=self.latent_stride * self.wan_input_scale,
                prompt_embeds=self.prompt,
                streaming_state=wan_state,
                generator=generator,
                device=self.device,
                compute_dtype=self.dtype,
            )[:, :, 0].to(self.dtype)
            pending_timings.append(("wan", span.finish()))
            self.last_keyframe_indices.append(packet.index)
            self.keyframe_count += 1
            if self.capture_state:
                self.last_wan_latent = latent
            span = DeviceSpan(self.device)
            prepared = self.flash.prepare_condition(latent) if self.reuse_endpoint_kv else None
            pending_timings.append(("flash_condition", span.finish()))
            return latent, prepared

        if resume is None:
            # First condition can be computed while the producer collects lookahead.
            start_latent, start_prepared = keyframe(first)
        else:
            wan_state.model_state.restore(resume["wan"], device=self.device, dtype=self.dtype)
            if generator is not None:
                generator[0].set_state(resume["generator"])
            self.keyframe_count = int(resume["keyframe_count"])
            start_latent = resume["latent"].to(device=self.device, dtype=self.dtype)
            start_prepared = (
                self.flash.prepare_condition(start_latent) if self.reuse_endpoint_kv else None
            )
        start = first
        first_window = resume is None
        while True:
            window = [start, *islice(iterator, self.window_size - 1)]
            for offset, packet in enumerate(window):
                if packet.index != start.index + offset or tuple(packet.rgb.shape) != shape:
                    raise ValueError(
                        "Streaming frames must be consecutive with constant dimensions."
                    )
            if len(window) == 1 and not first_window:
                return
            end = window[-1]
            if end.index == start.index:
                end_latent, end_prepared = start_latent, start_prepared
            else:
                end_latent, end_prepared = keyframe(end)
            self._checkpoint_source = (end.index, end_latent, wan_state, generator)
            count = max(2, len(window))
            span = DeviceSpan(self.device)
            state = (
                self.flash.begin_stream(
                    start_prepared,
                    end_prepared,
                    total_frames=count,
                    input_height=ph,
                    input_width=pw,
                )
                if self.reuse_endpoint_kv
                else None
            )
            pending_timings.append(("flash_condition", span.finish()))
            for offset, packet in enumerate(window):
                span = DeviceSpan(self.device)
                native = prepare_lr(packet.rgb, device=self.device, dtype=self.dtype, stride=stride)
                kwargs = {}
                if offset == 0 and not self.reuse_endpoint_kv:
                    kwargs = {
                        "condition_latent": start_latent,
                        "end_condition_latent": end_latent,
                        "total_frames": count,
                    }
                pending_timings.append(("flash_input", span.finish()))
                span = DeviceSpan(self.device)
                prediction, state = self.flash.decode_step(native, state, **kwargs)
                pending_timings.append(("flash", span.finish()))
                if self.capture_state:
                    self.last_flash_state = state
                # The overlap must update history, but has already been emitted.
                if first_window or offset:
                    timings = tuple(pending_timings)
                    pending_timings.clear()
                    yield OutputFrame(
                        packet.index,
                        prediction[..., : h * scale, : w * scale],
                        packet.source_time,
                        packet.arrival_time,
                        start.index,
                        end.index,
                        timings,
                    )
            first_window = False
            start, start_latent, start_prepared = end, end_latent, end_prepared
            if len(window) < self.window_size:
                return
