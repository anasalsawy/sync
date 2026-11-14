"""VisionBuffer — a live ring buffer of V lobe tier1 snapshots.

V writes here every pass. All active offspring and main lobes read the
latest frame at inject time. No lock needed: Python dict/list writes are
atomic for single-assignment, and we always read the latest complete frame.

Design:
  - Ring of N frames (default 3). Oldest frame evicted when full.
  - Each frame: {layer_idx: tensor(slots, hidden)} — same shape as tier1.
  - `latest()` returns the most recent complete frame immediately.
  - `push()` called by NDIFChannels after every generate() that includes V.
  - Any number of readers (offspring tasks) call `latest()` concurrently.

This is what makes browser/computer use fast: when R1 is halfway through
its layer loop and V finishes a new screen capture pass, the next layer
iteration picks up the new frame — no waiting for the offspring pass to
complete and restart.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class VisionBuffer:
    """Thread-safe (asyncio) ring buffer of V lobe activation snapshots."""

    def __init__(self, maxframes: int = 3) -> None:
        self._frames: deque[dict[int, Any]] = deque(maxlen=maxframes)
        self._seq: int = 0  # monotonic frame counter

    def push(self, tier1_v: dict[int, Any]) -> None:
        """Write a new V frame. Called from the main pass after generate()."""
        self._frames.append(tier1_v)
        self._seq += 1

    def latest(self) -> dict[int, Any] | None:
        """Read the most recent V frame. Returns None if V has never fired."""
        if not self._frames:
            return None
        return self._frames[-1]

    def seq(self) -> int:
        """Monotonic counter — offspring can detect a new frame by comparing."""
        return self._seq

    def __bool__(self) -> bool:
        return bool(self._frames)
