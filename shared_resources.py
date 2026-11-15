"""Cross-task resource coordination for Hands paths that share hardware.

Independent paths (shell file I/O vs browser mouse) overlap freely. Paths that
need the same desktop mouse/keyboard serialize on those locks only.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator


DESKTOP_MOUSE = "desktop.mouse"
DESKTOP_KEYBOARD = "desktop.keyboard"
DESKTOP_DISPLAY = "desktop.display"

# Path → resources it may need for interactive desktop work
PATH_RESOURCES: dict[str, tuple[str, ...]] = {
    "browser": (DESKTOP_MOUSE, DESKTOP_KEYBOARD, DESKTOP_DISPLAY),
    "application": (DESKTOP_MOUSE, DESKTOP_KEYBOARD, DESKTOP_DISPLAY),
    "shell": (),  # file/shell does not need the desktop pointer
}


@dataclass
class SharedResourceBus:
    """Process-wide locks keyed by resource name."""

    locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    holders: dict[str, str] = field(default_factory=dict)
    wait_s: float = 0.0
    acquire_log: list[dict] = field(default_factory=list)

    def _lock(self, name: str) -> asyncio.Lock:
        if name not in self.locks:
            self.locks[name] = asyncio.Lock()
        return self.locks[name]

    @asynccontextmanager
    async def acquire(
        self, resources: tuple[str, ...] | list[str], *, holder: str,
    ) -> AsyncIterator[None]:
        names = sorted(set(resources))
        if not names:
            yield
            return
        # Acquire in sorted order to avoid deadlock across tasks
        acquired: list[asyncio.Lock] = []
        t0 = time.perf_counter()
        try:
            for name in names:
                lock = self._lock(name)
                await lock.acquire()
                acquired.append(lock)
                self.holders[name] = holder
            waited = time.perf_counter() - t0
            self.wait_s += waited
            self.acquire_log.append({
                "holder": holder,
                "resources": names,
                "wait_s": waited,
                "at": time.perf_counter(),
            })
            yield
        finally:
            for name, lock in zip(reversed(names), reversed(acquired)):
                if self.holders.get(name) == holder:
                    self.holders.pop(name, None)
                lock.release()

    def snapshot(self) -> dict:
        return {
            "holders": dict(self.holders),
            "wait_s": self.wait_s,
            "acquires": len(self.acquire_log),
        }
