"""Incremental RGB input and atomic MP4 output, with rational frame rates."""

from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _video_stream(container):
    if not container.streams.video:
        raise ValueError("Input has no video stream.")
    return container.streams.video[0]


def input_fps(path, override=None):
    if override is not None:
        fps = Fraction(str(override))
    else:
        path = Path(path)
        if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES:
            raise ValueError("Image inputs require --fps, for example --fps 24.")
        with av.open(str(path)) as container:
            fps = _video_stream(container).average_rate
            if fps is None:
                raise ValueError("Video frame rate is unavailable; specify --fps.")
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    return fps


def frame_count(path):
    """Return the number of input frames, or None when the container does not tell.

    Nothing is decoded: for a video this is the container's count, or else its
    duration times the frame rate, so it can be slightly off.
    """
    path = Path(path)
    if path.is_dir():
        return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return 1
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return stream.frames
        rate = stream.average_rate
        if rate:
            if stream.duration is not None and stream.time_base is not None:
                return round(stream.duration * stream.time_base * rate) or None
            if container.duration is not None:
                return round(Fraction(container.duration, av.time_base) * rate) or None
    return None


def iter_frames(path, *, max_frames=None, start_frame=0):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if start_frame < 0 or (max_frames is not None and max_frames < 1):
        raise ValueError("Require start_frame >= 0 and max_frames >= 1.")

    def images():
        paths = (
            sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if path.is_dir()
            else [path]
        )
        for image_path in paths:
            with Image.open(image_path) as image:
                yield np.asarray(image.convert("RGB")).copy()

    def video():
        with av.open(str(path)) as container:
            stream = _video_stream(container)
            stream.codec_context.thread_count = 1
            for frame in container.decode(stream):
                yield frame.to_ndarray(format="rgb24")

    source = images() if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES else video()
    count = 0
    try:
        for index, rgb in enumerate(source):
            if index < start_frame:
                continue
            yield torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()
            count += 1
            if count == max_frames:
                break
    finally:
        source.close()
    if count == 0:
        raise ValueError("No frames were decoded from the requested input range.")


class VideoWriter:
    def __init__(self, path, fps, threads=1):
        if threads < 0:
            raise ValueError("Encoder threads must be 0 (automatic) or positive.")
        self.path, self.fps, self.threads = Path(path), fps, threads
        self.container = self.stream = None
        self.frames = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix=".refvsr-", suffix=".mp4", dir=self.path.parent, delete=False
        )
        self.temporary = Path(handle.name)
        handle.close()
        self.container = av.open(str(self.temporary), mode="w", format="mp4")
        return self

    def write(self, frame):
        rgb = frame.numpy() if isinstance(frame, torch.Tensor) else frame
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB [H,W,3].")
        h, w = rgb.shape[:2]
        if self.stream is None:
            self.stream = self.container.add_stream("libx264", rate=self.fps)
            self.stream.height, self.stream.width = h, w
            self.stream.pix_fmt = "yuv420p" if h % 2 == w % 2 == 0 else "yuv444p"
            self.stream.options = {"crf": "18", "preset": "fast"}
            self.stream.codec_context.thread_count = self.threads
        if (h, w) != (self.stream.height, self.stream.width):
            raise ValueError("Output dimensions changed.")
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            self.container.mux(packet)
        self.frames += 1

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                if self.stream is None:
                    raise ValueError("Cannot publish an empty video.")
                for packet in self.stream.encode():
                    self.container.mux(packet)
                self.container.close()
                self.container = None
                self.temporary.replace(self.path)
        finally:
            if self.container is not None:
                self.container.close()
            self.temporary.unlink(missing_ok=True)


def _audio_stream(container):
    return next(iter(container.streams.audio), None)


def has_audio(path) -> bool:
    path = Path(path)
    if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES:
        return False
    with av.open(str(path)) as container:
        return _audio_stream(container) is not None


def _video_packets(parts, stream, frame_ticks):
    """Yield (seconds, packet) for the parts played back to back."""
    offset = 0
    for part in parts:
        with av.open(str(part)) as container:
            source = container.streams.video[0]
            frames = 0
            for packet in container.demux(source):
                if packet.dts is None:
                    continue
                frames += 1
                packet.pts += offset
                packet.dts += offset
                packet.stream = stream
                yield float(packet.dts * stream.time_base), packet
        offset += frames * frame_ticks


def _audio_packets(container, source, stream, start, duration):
    """Yield (seconds, packet) for the audio inside [start, start + duration)."""
    # From the very start, keep the timestamps as they are: the encoder priming
    # packet sits before zero and must stay there.
    first = int(start / source.time_base) + (source.start_time or 0) if start > 0 else 0
    last = first + int(duration / source.time_base)
    if start > 0:
        container.seek(first, stream=source, backward=True)
    for packet in container.demux(source):
        if packet.dts is None or packet.pts is None:
            continue
        if start > 0 and packet.pts < first:
            continue
        if packet.pts >= last:
            break
        packet.pts -= first
        packet.dts = packet.pts
        packet.stream = stream
        yield float(packet.pts * stream.time_base), packet


def publish_video(parts, output, fps, *, audio_from=None, audio_start=0.0):
    """Join encoded parts into ``output`` atomically, optionally with the source audio.

    Nothing is re-encoded: video packets are concatenated and audio packets are
    copied for the time range the video covers. Returns whether audio was written;
    audio that the MP4 container cannot carry is dropped with a warning.
    """
    import heapq
    import warnings

    output = Path(output)
    parts = [Path(part) for part in parts]
    if not parts:
        raise ValueError("Cannot publish an empty video.")
    handle = tempfile.NamedTemporaryFile(
        prefix=".refvsr-", suffix=".mp4", dir=output.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()

    def write(with_audio):
        with av.open(str(parts[0])) as probe, av.open(str(temporary), "w", format="mp4") as out:
            template = probe.streams.video[0]
            video = out.add_stream_from_template(template)
            video.time_base = template.time_base
            frame_ticks = int(1 / (Fraction(fps) * template.time_base))
            streams = [_video_packets(parts, video, frame_ticks)]
            source = None
            if with_audio:
                source = av.open(str(audio_from))
            try:
                frames = 0
                if source is not None:
                    for part in parts:
                        with av.open(str(part)) as container:
                            frames += container.streams.video[0].frames
                    audio_in = _audio_stream(source)
                    audio = out.add_stream_from_template(audio_in)
                    audio.time_base = audio_in.time_base
                    streams.append(
                        _audio_packets(source, audio_in, audio, audio_start, frames / Fraction(fps))
                    )
                for _, packet in heapq.merge(*streams, key=lambda item: item[0]):
                    out.mux(packet)
            finally:
                if source is not None:
                    source.close()

    try:
        wrote_audio = audio_from is not None and has_audio(audio_from)
        if wrote_audio:
            try:
                write(True)
            except (av.FFmpegError, ValueError) as error:
                warnings.warn(f"Audio was not copied ({error}).", stacklevel=2)
                wrote_audio = False
        if not wrote_audio:
            write(False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return wrote_audio
