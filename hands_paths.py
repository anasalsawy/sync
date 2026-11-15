"""Hands execution paths: browser, application, shell — one path per task.

Each authorized task gets its own path instance. Three R workers in that task
group feed this path (native subagents when the path supports them; otherwise
prepare-ahead on a single stream). Cross-task desktop contention goes through
SharedResourceBus so a sequential browser does not block an independent shell.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from host_subagents import PreparedActionMessage
from live_hands_host import HandsHost
from shared_resources import PATH_RESOURCES, SharedResourceBus


PATH_BROWSER = "browser"
PATH_APPLICATION = "application"
PATH_SHELL = "shell"


def detect_hands_path(goal: str) -> str:
    g = goal.lower()
    if any(k in g for k in ("browser", "navigate", "url", "http://", "https://", "tab")):
        return PATH_BROWSER
    if any(k in g for k in ("application", "app ", "notepad", "window", "gui", "click")):
        return PATH_APPLICATION
    if any(k in g for k in ("shell", "command", "terminal", "powershell", "bash")):
        return PATH_SHELL
    # Default file/fix work rides the shell/file Hands path
    return PATH_SHELL


@dataclass
class PathSubagent:
    worker_id: str
    path: str
    delay_s: float
    bus: SharedResourceBus
    resources: tuple[str, ...]
    root: Path
    subagent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    history: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False
    busy_s: float = 0.0

    async def run_action(self, message: PreparedActionMessage) -> dict[str, Any]:
        holder = f"{self.path}:{self.worker_id}"
        async with self.bus.acquire(self.resources, holder=holder):
            t0 = time.perf_counter()
            await asyncio.sleep(self.delay_s)
            # Stamp observation under the task root (real disk I/O)
            stamp = self.root / "path_actions" / f"{message.action_id}.txt"
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text(
                f"{self.path}:{message.effect}@{message.resource} by {self.worker_id}\n",
                encoding="utf-8",
            )
            elapsed = time.perf_counter() - t0
            self.busy_s += elapsed
            obs = {
                "ok": True,
                "result": f"DONE:{message.effect}@{message.resource}",
                "action_id": message.action_id,
                "worker_id": self.worker_id,
                "subagent_id": self.subagent_id,
                "target": message.target,
                "via": f"hands_path:{self.path}:subagent",
                "path": self.path,
                "elapsed_s": elapsed,
            }
            if message.effect == "edit":
                obs["result"] = f"EDITED:{message.resource}"
                obs["file_content"] = message.payload.get(
                    "content", f"patched:{message.action_id}",
                )
            if message.effect == "test":
                failing = bool(message.payload.get("failing", False))
                obs["result"] = "TESTS_FAIL" if failing else "TESTS_PASS"
                obs["ok"] = not failing
            self.history.append(obs)
            return obs

    async def close(self) -> None:
        self.closed = True


@dataclass
class BrowserPathHost:
    """Sequential browser stream — one action at a time; needs desktop I/O."""

    root: Path
    bus: SharedResourceBus
    delay_s: float = 0.12
    supports_subagents: bool = False
    supports_concurrent_dispatch: bool = False
    path: str = PATH_BROWSER
    history: list[dict[str, Any]] = field(default_factory=list)
    in_flight: int = 0
    idle_s: float = 0.0
    busy_s: float = 0.0
    launched: list[str] = field(default_factory=list)
    _last_free: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    async def launch_subagents(self, worker_ids: list[str]) -> dict[str, Any]:
        raise RuntimeError("browser path is a single sequential stream")

    async def execute_single(self, message: PreparedActionMessage) -> dict[str, Any]:
        async with self._lock:
            now = time.perf_counter()
            if self._last_free is not None and self.in_flight == 0:
                self.idle_s += now - self._last_free
            self.in_flight = 1
            resources = PATH_RESOURCES[self.path]
            holder = f"{self.path}:{message.worker_id}"
            async with self.bus.acquire(resources, holder=holder):
                t0 = time.perf_counter()
                await asyncio.sleep(self.delay_s)
                stamp = self.root / "browser" / f"{message.action_id}.nav"
                stamp.parent.mkdir(parents=True, exist_ok=True)
                stamp.write_text(
                    f"nav:{message.effect}@{message.resource}\n", encoding="utf-8",
                )
                elapsed = time.perf_counter() - t0
            self.busy_s += elapsed
            obs = {
                "ok": True,
                "result": f"DONE:{message.effect}@{message.resource}",
                "action_id": message.action_id,
                "worker_id": message.worker_id,
                "via": "hands_path:browser:stream",
                "path": self.path,
                "elapsed_s": elapsed,
            }
            self.history.append(obs)
            self.in_flight = 0
            self._last_free = time.perf_counter()
            return obs


@dataclass
class ApplicationPathHost:
    """Application path with native subagents (one per R worker for this task)."""

    root: Path
    bus: SharedResourceBus
    delay_s: float = 0.10
    supports_subagents: bool = True
    supports_concurrent_dispatch: bool = True
    path: str = PATH_APPLICATION
    launched: list[str] = field(default_factory=list)
    subagents: dict[str, PathSubagent] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    async def launch_subagents(self, worker_ids: list[str]) -> dict[str, PathSubagent]:
        out: dict[str, PathSubagent] = {}
        resources = PATH_RESOURCES[self.path]
        for wid in worker_ids:
            sa = PathSubagent(
                worker_id=wid,
                path=self.path,
                delay_s=self.delay_s,
                bus=self.bus,
                resources=resources,
                root=self.root,
            )
            self.subagents[wid] = sa
            self.launched.append(wid)
            out[wid] = sa
        return out

    async def execute_single(self, message: PreparedActionMessage) -> dict[str, Any]:
        sa = PathSubagent(
            worker_id=message.worker_id,
            path=self.path,
            delay_s=self.delay_s,
            bus=self.bus,
            resources=PATH_RESOURCES[self.path],
            root=self.root,
        )
        obs = await sa.run_action(message)
        self.history.append(obs)
        return obs


@dataclass
class ShellPathHost:
    """Shell/file path: real Hands engine, single stream, no desktop lock."""

    root: Path
    bus: SharedResourceBus
    supports_subagents: bool = False
    supports_concurrent_dispatch: bool = False
    path: str = PATH_SHELL
    history: list[dict[str, Any]] = field(default_factory=list)
    in_flight: int = 0
    idle_s: float = 0.0
    busy_s: float = 0.0
    launched: list[str] = field(default_factory=list)
    hands_engine_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    _hands: Hands | None = None
    _last_free: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        # Reuse HandsHost seeding + engine
        self._inner = HandsHost(root=self.root)
        self._hands = self._inner._hands
        self.hands_engine_id = self._inner.hands_engine_id

    async def launch_subagents(self, worker_ids: list[str]) -> dict[str, Any]:
        raise RuntimeError("shell path is a single Hands stream")

    async def execute_single(self, message: PreparedActionMessage) -> dict[str, Any]:
        # No desktop resources — overlaps freely with browser/app desktop work
        async with self._lock:
            now = time.perf_counter()
            if self._last_free is not None and self.in_flight == 0:
                self.idle_s += now - self._last_free
            self.in_flight = 1
            obs = await self._inner.execute_single(message)
            obs["via"] = "hands_path:shell:hands"
            obs["path"] = self.path
            self.busy_s = self._inner.busy_s
            self.idle_s = self._inner.idle_s
            self.history.append(obs)
            self.in_flight = 0
            self._last_free = time.perf_counter()
            return obs


def build_hands_path(
    path: str,
    *,
    root: Path,
    bus: SharedResourceBus,
    browser_delay_s: float = 0.12,
    app_delay_s: float = 0.10,
) -> Any:
    if path == PATH_BROWSER:
        return BrowserPathHost(root=root / "browser", bus=bus, delay_s=browser_delay_s)
    if path == PATH_APPLICATION:
        return ApplicationPathHost(root=root / "application", bus=bus, delay_s=app_delay_s)
    return ShellPathHost(root=root / "shell", bus=bus)
