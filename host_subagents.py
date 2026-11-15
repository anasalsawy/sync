"""Map Sync's three R workers onto host execution surfaces.

When the host supports subagents: launch one native subagent per active R
worker for the life of the task. The mapping decides *where* a claimed action
runs — not which slice of the task a worker owns.

When the host has one execution path: keep all three R workers active; they
prepare ahead while the single path runs; dispatch the next prepared action
as soon as the host is free.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol

from multi_r_config import DEFAULT_MULTI_R_WORKERS
from task_record import TaskAction


@dataclass
class PreparedActionMessage:
    """Narrow instruction for a host subagent: execute this, return observation."""

    worker_id: str
    action_id: str
    effect: str
    resource: str
    target: str
    expected_result: str
    args: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    def instruction(self) -> str:
        return (
            f"Execute exactly this action and return the observation.\n"
            f"effect: {self.effect}\n"
            f"target: {self.target}\n"
            f"resource: {self.resource}\n"
            f"expected_result: {self.expected_result}\n"
            f"args: {self.args}\n"
            f"Do not plan other work. Do not speak to the user."
        )


class HostSubagent(Protocol):
    """One host execution surface bound to an R worker for the task lifetime."""

    subagent_id: str
    worker_id: str

    async def run_action(self, message: PreparedActionMessage) -> dict[str, Any]:
        ...

    async def close(self) -> None:
        ...


class MultiRHost(Protocol):
    """Host capability probed by the companion Multi-R path."""

    @property
    def supports_subagents(self) -> bool: ...

    @property
    def supports_concurrent_dispatch(self) -> bool: ...

    async def launch_subagents(
        self, worker_ids: list[str],
    ) -> dict[str, HostSubagent]:
        """Native launch: one subagent per R worker. Mapping lasts the task."""
        ...

    async def execute_single(
        self, message: PreparedActionMessage,
    ) -> dict[str, Any]:
        """Single execution path when subagents are unavailable."""
        ...


# ---------------------------------------------------------------------------
# Mock hosts for companion / eval
# ---------------------------------------------------------------------------

@dataclass
class MockHostSubagent:
    worker_id: str
    subagent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    delay_s: float = 0.08
    history: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    async def run_action(self, message: PreparedActionMessage) -> dict[str, Any]:
        await asyncio.sleep(self.delay_s)
        obs = {
            "ok": True,
            "result": f"DONE:{message.effect}@{message.resource}",
            "action_id": message.action_id,
            "worker_id": self.worker_id,
            "subagent_id": self.subagent_id,
            "target": message.target,
            "via": "native_subagent",
        }
        if message.effect == "edit":
            obs["result"] = f"EDITED:{message.resource}"
            obs["file_content"] = message.payload.get("content", "patched")
        if message.effect == "test":
            failing = message.payload.get("failing", False)
            obs["result"] = "TESTS_FAIL" if failing else "TESTS_PASS"
            obs["ok"] = not failing
        self.history.append(obs)
        return obs

    async def close(self) -> None:
        self.closed = True


@dataclass
class NativeSubagentHost:
    """Host that supports native subagents + concurrent independent dispatch."""

    delay_s: float = 0.08
    supports_subagents: bool = True
    supports_concurrent_dispatch: bool = True
    launched: list[str] = field(default_factory=list)
    subagents: dict[str, MockHostSubagent] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    async def launch_subagents(
        self, worker_ids: list[str],
    ) -> dict[str, HostSubagent]:
        out: dict[str, HostSubagent] = {}
        for wid in worker_ids:
            sa = MockHostSubagent(worker_id=wid, delay_s=self.delay_s)
            self.subagents[wid] = sa
            self.launched.append(wid)
            out[wid] = sa
        if len(out) != len(worker_ids):
            raise RuntimeError(
                f"host launched {len(out)} subagents, expected {len(worker_ids)}"
            )
        return out

    async def execute_single(
        self, message: PreparedActionMessage,
    ) -> dict[str, Any]:
        # Fallback path (should not be primary when supports_subagents)
        sa = MockHostSubagent(worker_id=message.worker_id, delay_s=self.delay_s)
        obs = await sa.run_action(message)
        self.history.append(obs)
        return obs


@dataclass
class SinglePathHost:
    """Host with one execution path — no native subagents."""

    delay_s: float = 0.08
    supports_subagents: bool = False
    supports_concurrent_dispatch: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    in_flight: int = 0
    idle_s: float = 0.0
    busy_s: float = 0.0
    _last_free: float | None = None
    launched: list[str] = field(default_factory=list)

    async def launch_subagents(
        self, worker_ids: list[str],
    ) -> dict[str, HostSubagent]:
        raise RuntimeError("host does not support subagents")

    async def execute_single(
        self, message: PreparedActionMessage,
    ) -> dict[str, Any]:
        now = time.perf_counter()
        if self._last_free is not None and self.in_flight == 0:
            self.idle_s += now - self._last_free
        self.in_flight = 1
        t0 = time.perf_counter()
        await asyncio.sleep(self.delay_s)
        obs = {
            "ok": True,
            "result": f"DONE:{message.effect}@{message.resource}",
            "action_id": message.action_id,
            "worker_id": message.worker_id,
            "target": message.target,
            "via": "single_path",
        }
        if message.effect == "edit":
            obs["result"] = f"EDITED:{message.resource}"
            obs["file_content"] = message.payload.get("content", "patched")
        self.history.append(obs)
        self.busy_s += time.perf_counter() - t0
        self.in_flight = 0
        self._last_free = time.perf_counter()
        return obs


def message_from_action(
    action: TaskAction, payload: dict[str, Any], worker_id: str,
) -> PreparedActionMessage:
    target = str(action.args.get("target") or action.resource)
    expected = f"DONE:{action.effect}@{action.resource}"
    return PreparedActionMessage(
        worker_id=worker_id,
        action_id=action.action_id,
        effect=action.effect,
        resource=action.resource,
        target=target,
        expected_result=expected,
        args=dict(action.args),
        payload=dict(payload),
    )


@dataclass
class SubagentMappedExecutor:
    """Dispatch prepared actions onto the R-worker → host-subagent map.

    Independent actions run concurrently when the host allows it. Conflicting
    / dependent actions stay ordered by the shared task record.
    """

    host: MultiRHost
    mapping: dict[str, HostSubagent] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    idle_s: float = 0.0
    busy_s: float = 0.0
    in_flight: int = 0
    concurrent_dispatches: int = 0
    _active: int = 0
    _last_zero: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    async def launch(
        cls, host: MultiRHost, worker_ids: list[str] | None = None,
    ) -> "SubagentMappedExecutor":
        workers = worker_ids or [f"R{i}" for i in range(DEFAULT_MULTI_R_WORKERS)]
        if not host.supports_subagents:
            raise RuntimeError("SubagentMappedExecutor requires supports_subagents")
        mapping = await host.launch_subagents(workers)
        if len(mapping) != len(workers):
            raise RuntimeError(
                f"expected {len(workers)} host subagents, got {len(mapping)}"
            )
        return cls(host=host, mapping=mapping, _last_zero=time.perf_counter())

    async def execute(self, action: TaskAction, payload: dict[str, Any]) -> dict[str, Any]:
        worker_id = action.claimed_by or next(iter(self.mapping))
        if worker_id not in self.mapping:
            # Reuse a stable subagent — mapping is by worker, not by task slice
            worker_id = sorted(self.mapping.keys())[
                hash(action.action_id) % len(self.mapping)
            ]
        sa = self.mapping[worker_id]
        msg = message_from_action(action, payload, worker_id)

        async with self._lock:
            if self._active == 0 and self._last_zero is not None:
                self.idle_s += time.perf_counter() - self._last_zero
            self._active += 1
            self.in_flight = self._active
            if self._active > 1:
                self.concurrent_dispatches += 1

        t0 = time.perf_counter()
        try:
            obs = await sa.run_action(msg)
        finally:
            self.busy_s += time.perf_counter() - t0
            async with self._lock:
                self._active -= 1
                self.in_flight = self._active
                if self._active == 0:
                    self._last_zero = time.perf_counter()
        self.history.append(obs)
        return obs

    async def close(self) -> None:
        for sa in self.mapping.values():
            await sa.close()


@dataclass
class SinglePathMappedExecutor:
    """All three R workers stay active; host runs one action at a time."""

    host: Any
    worker_ids: list[str] = field(
        default_factory=lambda: [f"R{i}" for i in range(DEFAULT_MULTI_R_WORKERS)]
    )
    history: list[dict[str, Any]] = field(default_factory=list)
    idle_s: float = 0.0
    busy_s: float = 0.0
    in_flight: int = 0
    r_workers_active: int = DEFAULT_MULTI_R_WORKERS

    async def execute(self, action: TaskAction, payload: dict[str, Any]) -> dict[str, Any]:
        worker_id = action.claimed_by or self.worker_ids[0]
        msg = message_from_action(action, payload, worker_id)
        self.in_flight = 1
        obs = await self.host.execute_single(msg)
        self.in_flight = 0
        self.idle_s = float(getattr(self.host, "idle_s", self.idle_s) or 0.0)
        self.busy_s = float(getattr(self.host, "busy_s", self.busy_s) or 0.0)
        self.history.append(obs)
        return obs

    async def close(self) -> None:
        return None


async def bind_host_executor(
    host: MultiRHost | None,
    *,
    n_workers: int = DEFAULT_MULTI_R_WORKERS,
    worker_id_prefix: str = "",
) -> tuple[Any, str, dict[str, Any]]:
    """Pick subagent-mapped or single-path executor from host capabilities.

    Returns (executor, mode_label, launch_info).
    """
    pref = f"{worker_id_prefix}:" if worker_id_prefix else ""
    worker_ids = [f"{pref}R{i}" for i in range(n_workers)]
    if host is not None and host.supports_subagents:
        ex = await SubagentMappedExecutor.launch(host, worker_ids)
        info = {
            "host_mode": "native_subagent",
            "subagents_launched": list(ex.mapping.keys()),
            "n_subagents": len(ex.mapping),
            "concurrent_dispatch": host.supports_concurrent_dispatch,
            "host_backend": type(host).__name__,
        }
        # parallel coordinator mode when host can overlap independent actions
        mode = "parallel" if host.supports_concurrent_dispatch else "sequential"
        return ex, mode, info

    # Single execution path — keep three R workers preparing ahead.
    # Prefer the caller's host when it implements execute_single (Hands, etc.).
    if host is not None and hasattr(host, "execute_single") and not host.supports_subagents:
        single: Any = host
        backend = type(host).__name__
    elif isinstance(host, SinglePathHost):
        single = host
        backend = "SinglePathHost"
    else:
        single = SinglePathHost(delay_s=getattr(host, "delay_s", 0.08) if host else 0.08)
        backend = "SinglePathHost"
    ex = SinglePathMappedExecutor(host=single, worker_ids=worker_ids)
    info = {
        "host_mode": "single_path",
        "subagents_launched": [],
        "n_subagents": 0,
        "r_workers_active": n_workers,
        "concurrent_dispatch": False,
        "host_backend": backend,
    }
    return ex, "sequential", info
