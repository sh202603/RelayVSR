"""Segmented output that survives an interrupted run.

The video is encoded as independent parts that end on keyframe boundaries.
After each part, the stream state at that boundary is saved, so a later run
with the same settings continues from the last finished part and produces the
frames an uninterrupted run would have produced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

FORMAT = 1


def file_fingerprint(path) -> dict:
    """Identity of an input without reading it."""
    path = Path(path).resolve()
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file())
        return {"path": str(path), "files": [[p.name, p.stat().st_size] for p in files]}
    stat = path.stat()
    # Integer nanoseconds: a float mtime does not survive a JSON round trip exactly.
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class SegmentStore:
    """Directory of finished parts, one stream checkpoint and a manifest.

    ``params`` must hold everything that affects the pixels or the part layout;
    a manifest written with different params is discarded. Part 0 holds frames
    ``0..segment_frames`` and every later part the next ``segment_frames``
    frames, so each part ends on the frame a checkpoint describes.
    """

    def __init__(self, directory, params: dict, segment_frames: int):
        if segment_frames < 1:
            raise ValueError("segment_frames must be positive.")
        self.directory = Path(directory)
        self.segment_frames = segment_frames
        self.params = json.loads(json.dumps(params, sort_keys=True))
        self.segments: list[str] = []
        self.checkpoint_name: str | None = None
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = self._read()
        if (
            manifest is not None
            and manifest.get("format") == FORMAT
            and manifest.get("params") == self.params
            and manifest.get("segment_frames") == segment_frames
            and manifest["segments"]
            and all((self.directory / name).is_file() for name in manifest["segments"])
            and (self.directory / manifest["checkpoint"]).is_file()
        ):
            self.segments = list(manifest["segments"])
            self.checkpoint_name = manifest["checkpoint"]
        self._sweep()

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    @property
    def boundary(self) -> int:
        """Index of the last finished frame, which the checkpoint describes; 0 if none."""
        return len(self.segments) * self.segment_frames

    @property
    def parts(self) -> list[Path]:
        return [self.directory / name for name in self.segments]

    @property
    def tail_path(self) -> Path:
        return self.directory / "tail.mp4"

    def next_segment_path(self) -> Path:
        return self.directory / f"part_{len(self.segments):05d}.mp4"

    def _read(self):
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _owned(self, path: Path) -> bool:
        name = path.name
        return (
            name == "manifest.json"
            or name.startswith(".refvsr-")
            or (name.endswith(".mp4") and (name.startswith("part_") or name == "tail.mp4"))
            or (name.endswith(".pt") and name.startswith("state_"))
        )

    def _sweep(self) -> None:
        """Delete leftovers of this store that the manifest does not reference."""
        keep = {*self.segments, self.checkpoint_name}
        if self.segments:
            keep.add("manifest.json")
        for path in self.directory.iterdir():
            if path.is_file() and self._owned(path) and path.name not in keep:
                path.unlink()

    def load_checkpoint(self) -> dict | None:
        if self.checkpoint_name is None:
            return None
        return torch.load(self.directory / self.checkpoint_name, weights_only=True)

    def commit(self, segment: Path, checkpoint: dict) -> None:
        """Record a finished part together with the stream state at its last frame."""
        if segment != self.next_segment_path() or not segment.is_file():
            raise ValueError("Parts must be committed in order.")
        expected = self.boundary + self.segment_frames
        if checkpoint["frame"] != expected:
            raise ValueError("Checkpoint does not describe the end of this part.")
        name = f"state_{expected:09d}.pt"
        temporary = self.directory / f".refvsr-{name}"
        torch.save(checkpoint, temporary)
        temporary.replace(self.directory / name)
        previous = self.checkpoint_name
        self.segments.append(segment.name)
        self.checkpoint_name = name
        manifest = {
            "format": FORMAT,
            "params": self.params,
            "segment_frames": self.segment_frames,
            "segments": self.segments,
            "checkpoint": name,
        }
        temporary = self.directory / ".refvsr-manifest.json"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        # The manifest is the commit point: until it is replaced, the old state stands.
        temporary.replace(self.manifest_path)
        if previous is not None:
            (self.directory / previous).unlink(missing_ok=True)

    def remove(self) -> None:
        """Delete the store after the final video has been published."""
        self.segments, self.checkpoint_name = [], None
        self._sweep()
        try:
            self.directory.rmdir()
        except OSError:
            pass  # the user keeps other files here
