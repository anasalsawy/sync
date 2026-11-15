"""Main R coordinates concurrent task groups.

Rule: one authorized task → one group of exactly three R workers in one
inference batch. Existing groups keep running. Each group owns its SharedTaskRecord
and routes prepared actions to that task's Hands execution path.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hands_paths import (
    PATH_APPLICATION,
    PATH_BROWSER,
    PATH_SHELL,
    build_hands_path,
    detect_hands_path,
)
from host_subagents import bind_host_executor
from multi_r_config import DEFAULT_MULTI_R_WORKERS, MultiRLaunchError, multi_r_worker_count
from r_coordinator import CoordinatorResult, RCoordinator
from shared_resources import SharedResourceBus


EventCb = Callable[[dict[str, Any]], None] | None


@dataclass
class TaskGroup:
    """One task ↔ one three-worker R batch ↔ one Hands path."""

    group_id: str
    task_id: str
    goal: str
    path: str
    worker_ids: list[str]
    n_workers: int = DEFAULT_MULTI_R_WORKERS
    status: str = "starting"  # starting|running|done|failed
    started_at: float = 0.0
    finished_at: float | None = None
    inference_started_at: float | None = None
    inference_finished_at: float | None = None
    result: CoordinatorResult | None = None
    host_launch: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    _async_task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def active_r_count(self) -> int:
        if self.status in ("starting", "running"):
            return self.n_workers
        return 0


@dataclass
class MainRCoordinator:
    """Main R lobe: spawn/keep task groups; never steal workers across tasks."""

    bus: SharedResourceBus = field(default_factory=SharedResourceBus)
    groups: dict[str, TaskGroup] = field(default_factory=dict)
    workers_per_group: int = DEFAULT_MULTI_R_WORKERS
    root: Path = field(default_factory=lambda: Path("hands_live_root"))
    crossing: Any = None
    activation_injection: bool = True
    on_event: EventCb = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def snapshot(self) -> dict[str, Any]:
        active = [g for g in self.groups.values() if g.status in ("starting", "running")]
        workers = []
        for g in self.groups.values():
            for wid in g.worker_ids:
                workers.append({
                    "worker_id": wid,
                    "group_id": g.group_id,
                    "task_id": g.task_id,
                    "path": g.path,
                    "active": g.status in ("starting", "running"),
                })
        return {
            "active_task_groups": len(active),
            "total_task_groups": len(self.groups),
            "workers_per_group": self.workers_per_group,
            "active_r_workers_total": sum(g.active_r_count for g in active),
            "groups": [
                {
                    "group_id": g.group_id,
                    "task_id": g.task_id,
                    "path": g.path,
                    "goal": g.goal[:80],
                    "status": g.status,
                    "n_workers": g.n_workers,
                    "worker_ids": g.worker_ids,
                    "host_mode": g.host_launch.get("host_mode"),
                    "inference_s": (
                        None
                        if g.inference_started_at is None or g.inference_finished_at is None
                        else g.inference_finished_at - g.inference_started_at
                    ),
                    "wall_s": (
                        None
                        if g.finished_at is None
                        else g.finished_at - g.started_at
                    ),
                    "satisfied": bool(g.result and g.result.satisfied),
                }
                for g in self.groups.values()
            ],
            "workers": workers,
            "resources": self.bus.snapshot(),
        }

    async def start_task(
        self,
        goal: str,
        *,
        path: str | None = None,
        host: Any | None = None,
        force_duplicate: bool = False,
        wait: bool = True,
    ) -> TaskGroup:
        """L-authorized task → immediately start a new 3-R group; keep others."""
        n = multi_r_worker_count(override=self.workers_per_group)
        if n != DEFAULT_MULTI_R_WORKERS and self.workers_per_group == DEFAULT_MULTI_R_WORKERS:
            n = self.workers_per_group
        path_kind = path or detect_hands_path(goal)
        group_id = f"tg_{uuid.uuid4().hex[:10]}"
        pref = f"{path_kind}:{group_id}"
        worker_ids = [f"{pref}:R{i}" for i in range(n)]

        group = TaskGroup(
            group_id=group_id,
            task_id="",  # filled after SharedTaskRecord create
            goal=goal,
            path=path_kind,
            worker_ids=worker_ids,
            n_workers=n,
            status="starting",
            started_at=time.perf_counter(),
        )

        path_host = host or build_hands_path(
            path_kind, root=self.root / group_id, bus=self.bus,
        )

        async with self._lock:
            self.groups[group_id] = group

        self._emit({
            "type": "task_group_started",
            "group_id": group_id,
            "path": path_kind,
            "n_workers": n,
            "worker_ids": worker_ids,
            "active_groups": self.snapshot()["active_task_groups"],
            "active_r_workers_total": self.snapshot()["active_r_workers_total"],
        })

        async def _run() -> None:
            group.status = "running"
            try:
                bound_executor, bound_mode, host_info = await bind_host_executor(
                    path_host,
                    n_workers=n,
                    worker_id_prefix=pref,
                )
                group.host_launch = dict(host_info)
                group.host_launch["path"] = path_kind
                group.host_launch["group_id"] = group_id
                self._emit({
                    "type": "host_launch",
                    **group.host_launch,
                    "n_workers": n,
                    "worker_ids": worker_ids,
                })

                coord = RCoordinator(
                    n_workers=n,
                    cross_attn=self.activation_injection,
                    mode=bound_mode,
                    crossing=self.crossing,
                    executor=bound_executor,
                    worker_id_prefix=pref,
                )
                group.inference_started_at = time.perf_counter()
                result = await coord.run_task(
                    goal,
                    constraints=(
                        f"silent R group {group_id} on Hands path={path_kind}; "
                        "L owns speech and decisions"
                    ),
                    source_material=f"path={path_kind}",
                    force_duplicate=force_duplicate,
                    influence_probe=self.activation_injection,
                )
                group.inference_finished_at = (
                    group.inference_started_at + float(result.metrics.inference_s or 0)
                )
                close = getattr(bound_executor, "close", None)
                if close is not None:
                    await close()
                result.host_launch = dict(group.host_launch)
                group.result = result
                group.task_id = result.record.task_id
                group.status = "done" if result.satisfied else "failed"
                if not result.satisfied:
                    group.error = "unsatisfied"
            except MultiRLaunchError as e:
                group.status = "failed"
                group.error = str(e)
                raise
            except Exception as e:  # noqa: BLE001
                group.status = "failed"
                group.error = f"{type(e).__name__}: {e}"
                raise
            finally:
                group.finished_at = time.perf_counter()
                self._emit({
                    "type": "task_group_done",
                    "group_id": group_id,
                    "task_id": group.task_id,
                    "path": path_kind,
                    "status": group.status,
                    "error": group.error,
                    "snapshot": self.snapshot(),
                })

        group._async_task = asyncio.create_task(_run(), name=f"task_group:{group_id}")
        if wait:
            await group._async_task
        return group

    async def start_many(
        self,
        specs: list[dict[str, Any]],
        *,
        wait: bool = True,
    ) -> list[TaskGroup]:
        """Start several task groups without waiting between launches."""
        groups: list[TaskGroup] = []
        for spec in specs:
            g = await self.start_task(
                spec["goal"],
                path=spec.get("path"),
                host=spec.get("host"),
                force_duplicate=bool(spec.get("force_duplicate", False)),
                wait=False,
            )
            groups.append(g)
            # Yield so each group reaches 'running' before the next launch returns
            await asyncio.sleep(0)
        if wait:
            await asyncio.gather(*(g._async_task for g in groups if g._async_task))
        return groups

    async def wait_all(self) -> list[TaskGroup]:
        tasks = [g._async_task for g in self.groups.values() if g._async_task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return list(self.groups.values())

    def peak_concurrency_probe(self) -> dict[str, Any]:
        """Best-effort peak while groups are mid-flight (call during wait)."""
        return self.snapshot()
