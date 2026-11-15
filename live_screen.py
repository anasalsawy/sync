"""V lobe live screen/desktop state — shared into worker lanes without screenshots.

V owns the freshest browser/desktop state. Multi-R workers and offspring read
``latest()`` on every prepare/dispatch tick. An explicit observe/screenshot
round-trip is skipped when a live frame is already present: workers act on the
shared state immediately.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ScreenFrame:
    """Compact live surface state (not a PNG)."""

    seq: int
    source: str = "browser"  # browser | desktop | window
    url: str = ""
    title: str = ""
    summary: str = ""
    links: list[dict[str, str]] = field(default_factory=list)
    controls: list[dict[str, str]] = field(default_factory=list)
    focused: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.perf_counter)

    def to_worker_context(self, *, max_chars: int = 1800) -> str:
        """Text injected into R worker prompts / preparations."""
        lines = [
            f"[V live screen seq={self.seq} source={self.source}]",
            f"url={self.url}",
            f"title={self.title}",
        ]
        if self.focused:
            lines.append(f"focused={self.focused}")
        if self.summary:
            lines.append(f"summary={self.summary[:400]}")
        if self.links:
            lines.append("links:")
            for link in self.links[:10]:
                lines.append(
                    f"  - {link.get('text', '')[:60]} -> {link.get('href', '')[:120]}"
                )
        if self.controls:
            lines.append("controls:")
            for c in self.controls[:10]:
                lines.append(
                    f"  - {c.get('tag', '')}#{c.get('id', '')} "
                    f"name={c.get('name', '')} text={c.get('text', '')[:40]}"
                )
        text = "\n".join(lines)
        return text[:max_chars]


class LiveScreenBus:
    """Process-wide live screen bus owned by V, read by all R worker lanes."""

    def __init__(self) -> None:
        self._frame: ScreenFrame | None = None
        self._seq: int = 0
        self.push_count: int = 0
        self.skip_observe_count: int = 0

    @property
    def seq(self) -> int:
        return self._seq

    def has_live(self) -> bool:
        return self._frame is not None

    def push(self, **kwargs: Any) -> ScreenFrame:
        self._seq += 1
        self.push_count += 1
        frame = ScreenFrame(seq=self._seq, **kwargs)
        self._frame = frame
        return frame

    def push_frame(self, frame: ScreenFrame) -> ScreenFrame:
        self._seq = max(self._seq + 1, frame.seq)
        frame.seq = self._seq
        self._frame = frame
        self.push_count += 1
        return frame

    def latest(self) -> ScreenFrame | None:
        return self._frame

    def worker_context(self) -> str:
        if self._frame is None:
            return "[V live screen: none yet]"
        return self._frame.to_worker_context()

    def should_skip_observe(self) -> bool:
        """True when workers already have a live V frame — no screenshot trip."""
        return self._frame is not None

    def note_observe_skipped(self) -> None:
        self.skip_observe_count += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self._seq,
            "has_live": self.has_live(),
            "push_count": self.push_count,
            "skip_observe_count": self.skip_observe_count,
            "frame": asdict(self._frame) if self._frame else None,
        }


# Default shared bus for companion / Multi-R / action runtime
LIVE_SCREEN = LiveScreenBus()
