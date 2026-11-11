"""Time as something the mind lives inside, not a value it is handed once.

The clock never stops. Every pass is stamped with it, and the world's events
arrive on a timeline as they happen, whether or not a lobe is thinking, so the
mind is always inside a present that is still moving.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


class Clock:
    def now(self) -> float:
        return time.time()


class ManualClock(Clock):
    """A clock a test moves forward by hand."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@dataclass(frozen=True)
class Event:
    """Something the world did, stamped with when it happened."""

    at: float
    text: str
