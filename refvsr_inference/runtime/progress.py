"""A one-line progress display: frames done, current and average FPS, elapsed time and ETA."""

from __future__ import annotations

import sys
import time
from collections import deque


def _clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class Progress:
    """Counts emitted frames and redraws one status line on stderr.

    ``total`` is the frame count of the whole run, or None when it is unknown.
    ``done`` frames were finished by an earlier run: they count toward the
    position but not toward the speed.
    """

    def __init__(self, total=None, done=0, *, stream=None, window_seconds=10.0):
        self.total, self.done = total, done
        self.frames = 0
        self.stream = sys.stderr if stream is None else stream
        self.interactive = self.stream.isatty()
        # A terminal line is redrawn in place; a log file gets a new line much less often.
        self.interval = 0.5 if self.interactive else 30.0
        self.window_seconds = window_seconds
        self.started = time.perf_counter()
        self.samples = deque([(self.started, 0)])
        self.last_draw = self.started
        self.width = 0

    def update(self, frames=1):
        self.frames += frames
        now = time.perf_counter()
        if now - self.last_draw >= self.interval:
            self._draw(now)

    def close(self):
        """Draw the final state and end the line, so later output starts on its own line."""
        self._draw(time.perf_counter(), final=True)
        if self.interactive:
            self.stream.write("\n")
            self.stream.flush()

    def _draw(self, now, final=False):
        self.last_draw = now
        self.samples.append((now, self.frames))
        while len(self.samples) > 2 and now - self.samples[1][0] >= self.window_seconds:
            self.samples.popleft()
        elapsed = now - self.started
        average = self.frames / elapsed if elapsed > 0 else 0.0
        then, before = self.samples[0]
        recent = (self.frames - before) / (now - then) if now > then else 0.0
        position = self.done + self.frames
        # The container's frame count can be an estimate; drop it once it is exceeded.
        total = self.total if self.total is not None and position <= self.total else None
        text = f"{position}/{total} {position / total:6.1%}" if total else f"{position} frames"
        if final:
            text += f"  avg {average:.2f} fps  elapsed {_clock(elapsed)}"
        else:
            text += f"  {recent:.2f} fps (avg {average:.2f})  elapsed {_clock(elapsed)}"
            if total and average > 0:
                text += f"  ETA {_clock((total - position) / average)}"
        if self.interactive:
            padding = " " * max(0, self.width - len(text))
            self.width = len(text)
            self.stream.write(f"\r{text}{padding}")
        else:
            self.stream.write(text + "\n")
        self.stream.flush()
