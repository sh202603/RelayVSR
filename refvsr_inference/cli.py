"""Command line interface: `relayvsr`, or `python inference.py` from the repository."""

import argparse
import sys
import time
from pathlib import Path

import torch

from refvsr_inference.media import (
    VideoWriter,
    frame_count,
    has_audio,
    input_fps,
    iter_frames,
    publish_video,
)
from refvsr_inference.runtime.progress import Progress
from refvsr_inference.runtime.resume import SegmentStore, file_fingerprint
from refvsr_inference.runtime.session import build_stream, to_uint8
from refvsr_inference.runtime.streaming import InputFrame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="relayvsr",
        description="RelayVSR video super-resolution: 4x, or 2x with the 2x checkpoints",
    )
    # The repository root: default weights and the demo input live next to the package.
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input", type=Path, default=project_root / "assets" / "demo" / "input.mp4",
        help="Video or ordered frame directory (default: assets/demo/input.mp4)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output MP4")
    weights = project_root / "weights"
    parser.add_argument(
        "--wan-checkpoint", type=Path, default=weights / "wan.safetensors",
        help="Wan checkpoint (default: weights/wan.safetensors)",
    )
    parser.add_argument(
        "--flash-checkpoint", type=Path, default=weights / "flash.safetensors",
        help="Flash checkpoint (default: weights/flash.safetensors, S step-60000 EMA)",
    )
    parser.add_argument("--window-size", type=int, default=16, help="Reference interval + 1")
    parser.add_argument("--fps", help="Override FPS; required for frame directories")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--kv-cache", choices=["auto", "gpu", "cpu"], default="auto",
        help="Where the Wan K/V cache lives; cpu trades speed for VRAM (default: auto)",
    )
    parser.add_argument(
        "--kv-cache-dtype", choices=["native", "int8"], default="native",
        help="int8 halves the Wan K/V cache but changes the output slightly",
    )
    parser.add_argument(
        "--wan-weights", choices=["bf16", "fp8"], default="bf16",
        help="fp8 halves the Wan block weights on the GPU but changes the output slightly",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--encode-threads", type=int, default=1,
        help="x264 threads; 0 lets the encoder choose (default: 1)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-audio", action="store_true", help="Do not copy the input audio")
    parser.add_argument(
        "--resume", action="store_true",
        help="Write restartable parts to <output>.parts and continue an interrupted run",
    )
    parser.add_argument(
        "--resume-interval", type=int, default=600,
        help="Approximate frames per restartable part (default: 600)",
    )
    return parser.parse_args(argv)


@torch.no_grad()
def run(args):
    if args.window_size < 2 or args.cpu_threads < 1:
        raise ValueError("Require window-size >= 2 and cpu-threads >= 1.")
    if args.resume_interval < 1:
        raise ValueError("Require resume-interval >= 1.")
    if args.encode_threads < 0:
        raise ValueError("Require encode-threads >= 0.")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Output must end with .mp4.")
    if args.output.resolve() == args.input.resolve():
        raise ValueError("Input and output must differ.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite to replace it.")
    for checkpoint in (args.wan_checkpoint, args.flash_checkpoint):
        if checkpoint.suffix != ".safetensors" or not checkpoint.is_file():
            raise FileNotFoundError(f"RelayVSR checkpoint not found: {checkpoint}")
    torch.set_num_threads(args.cpu_threads)
    fps = input_fps(args.input, args.fps)
    device = torch.device(args.device)
    stream = build_stream(
        args.wan_checkpoint,
        args.flash_checkpoint,
        device=device,
        dtype=args.dtype,
        window_size=args.window_size,
        seed=args.seed,
        kv_cache=args.kv_cache,
        kv_cache_dtype=args.kv_cache_dtype,
        wan_weights=args.wan_weights,
    )
    wan, wan_weights = stream.wan, stream.wan_weights
    store = checkpoint = None
    if args.resume:
        # Parts must end on keyframes, which are window_size - 1 frames apart.
        step = args.window_size - 1
        segment_frames = max(1, round(args.resume_interval / step)) * step
        store = SegmentStore(
            args.output.with_name(args.output.name + ".parts"),
            {
                "input": file_fingerprint(args.input),
                "wan": file_fingerprint(args.wan_checkpoint),
                "flash": file_fingerprint(args.flash_checkpoint),
                "window_size": args.window_size,
                "seed": args.seed,
                "dtype": args.dtype,
                "wan_weights": wan_weights,
                "kv_cache_dtype": args.kv_cache_dtype,
                "start_frame": args.start_frame,
                "max_frames": args.max_frames,
                "fps": str(fps),
                "encode_threads": args.encode_threads,
            },
            segment_frames,
        )
        checkpoint = store.load_checkpoint()
    # A resumed stream consumes the checkpointed frame again without emitting it.
    first = store.boundary if checkpoint is not None else 0
    done = first + 1 if checkpoint is not None else 0
    if done:
        print(f"Resuming after frame {first} ({len(store.segments)} finished parts)")
    frames = iter_frames(
        args.input,
        max_frames=None if args.max_frames is None else args.max_frames - first,
        start_frame=args.start_frame + first,
    )
    packets = (
        InputFrame(i, rgb, source_time=i / float(fps)) for i, rgb in enumerate(frames, first)
    )
    config = wan.config
    print(
        f"Wan: {config.num_layers} layers, "
        f"dim {config.num_attention_heads * config.attention_head_dim}, "
        f"{wan.temporal_context_frames} cached keyframes; window size {args.window_size}; "
        f"{stream.scale}x upscaling"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stage_ms, stage_calls = {}, {}
    # Frames of the whole run, counting the ones an earlier run already finished.
    expected = frame_count(args.input)
    if expected is not None:
        expected = max(0, expected - args.start_frame)
        if args.max_frames is not None:
            expected = min(expected, args.max_frames)
    started = time.perf_counter()
    progress = Progress(expected or None, done)
    writer, new_frames = None, 0
    try:
        for result in stream.frames(packets, resume=checkpoint):
            if writer is None:
                writer = VideoWriter(
                    args.output if store is None else store.tail_path,
                    fps,
                    threads=args.encode_threads,
                )
                writer.__enter__()
            # Queued for the encoder thread; the stream does not wait for x264.
            writer.write(to_uint8(result.rgb).cpu())
            new_frames += 1
            progress.update()
            # The frame is already on the CPU, so resolving its spans cannot stall the stream.
            for name, span in result.timings:
                stage_ms[name] = stage_ms.get(name, 0.0) + span.milliseconds()
                stage_calls[name] = stage_calls.get(name, 0) + 1
            if (
                store is not None
                and result.index == result.window_end
                and result.index % store.segment_frames == 0
                and result.index > 0
            ):
                writer.__exit__(None, None, None)
                writer = None
                segment = store.next_segment_path()
                store.tail_path.replace(segment)
                store.commit(segment, stream.checkpoint())
        if writer is not None:
            finished, writer = writer, None
            finished.__exit__(None, None, None)
    except BaseException:
        if writer is not None:
            writer.__exit__(*sys.exc_info())
        raise
    finally:
        progress.close()
        packets.close()
        frames.close()
    audio_from = None if args.no_audio or not has_audio(args.input) else args.input
    audio_start = float(args.start_frame / fps)
    copied_audio = False
    if store is not None:
        parts = store.parts + ([store.tail_path] if store.tail_path.is_file() else [])
        copied_audio = publish_video(
            parts, args.output, fps, audio_from=audio_from, audio_start=audio_start
        )
        store.remove()
    elif audio_from is not None:
        copied_audio = publish_video(
            [args.output], args.output, fps, audio_from=audio_from, audio_start=audio_start
        )
    processing = progress.last_draw - progress.started
    elapsed = time.perf_counter() - started
    total = done + new_frames
    print(f"Saved {total} frames to {args.output} ({elapsed:.2f}s)")
    print(f"Audio: {'copied' if copied_audio else 'none'}")
    print(
        f"Wan weights: {wan_weights}; "
        f"K/V cache: {stream.resolved_wan_kv_cache}, {args.kv_cache_dtype}"
    )
    print(
        f"Throughput: {new_frames / processing:.2f} fps for {new_frames} frames in "
        f"{processing:.2f}s (decode, inference, encode and warm-up); "
        f"{new_frames / elapsed:.2f} fps including the final mux"
    )
    for name, total_ms in stage_ms.items():
        calls = stage_calls[name]
        print(
            f"  {name:<16}{total_ms / 1000:9.2f}s {calls:6d} calls {total_ms / calls:9.2f} ms/call"
        )
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 2**30
        reserved = torch.cuda.max_memory_reserved(device) / 2**30
        print(f"Peak VRAM: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")

def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
