"""R coordinates multiple silent workers on one shared task.

L owns decisions and speech. R workers prepare ahead while executors run.
Inference/preparation is separated from dispatch.

Executor modes:
  - sequential: one action in flight; prepare upcoming concurrently
  - batch: dispatch compatible prepared actions together
  - parallel: independent actions concurrent; same-resource ordered
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from batch_r import BatchedRTeam, BatchRResult
from crossing import Crossing
from multi_r_config import DEFAULT_MULTI_R_WORKERS, MultiRLaunchError
from task_record import SharedTaskRecord, TaskAction


class ActionExecutor(Protocol):
    async def execute(self, action: TaskAction, payload: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class SequentialExecutor:
    """One action in flight at a time."""

    delay_s: float = 0.05
    history: list[dict[str, Any]] = field(default_factory=list)
    in_flight: int = 0
    idle_s: float = 0.0
    busy_s: float = 0.0
    _last_free: float | None = None

    async def execute(self, action: TaskAction, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.perf_counter()
        if self._last_free is not None:
            self.idle_s += now - self._last_free
        self.in_flight = 1
        t0 = time.perf_counter()
        await asyncio.sleep(self.delay_s)
        result = {
            "ok": True,
            "result": f"DONE:{action.effect}@{action.resource}",
            "action_id": action.action_id,
            "payload": payload,
        }
        # Simulate file content for edit conflicts
        if action.effect == "edit":
            content = payload.get("content", "")
            prior = payload.get("prior_content", "")
            result["result"] = f"EDITED:{action.resource}:{content or prior}"
            result["file_content"] = content or f"patched:{action.args.get('patch', '')}"
        if action.effect == "test":
            failing = payload.get("failing", False)
            result["result"] = "TESTS_FAIL" if failing else "TESTS_PASS"
            result["ok"] = not failing
        self.history.append(result)
        self.busy_s += time.perf_counter() - t0
        self.in_flight = 0
        self._last_free = time.perf_counter()
        return result


@dataclass
class ParallelExecutor:
    """Independent actions concurrently; caller still orders same-resource."""

    delay_s: float = 0.05
    max_parallel: int = 4
    history: list[dict[str, Any]] = field(default_factory=list)
    idle_s: float = 0.0
    busy_s: float = 0.0
    _sem: asyncio.Semaphore | None = None
    _active: int = 0
    _last_zero: float | None = None

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_parallel)
        self._last_zero = time.perf_counter()

    async def execute(self, action: TaskAction, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._sem is not None
        async with self._sem:
            if self._active == 0 and self._last_zero is not None:
                self.idle_s += time.perf_counter() - self._last_zero
            self._active += 1
            t0 = time.perf_counter()
            await asyncio.sleep(self.delay_s)
            result = {
                "ok": True,
                "result": f"DONE:{action.effect}@{action.resource}",
                "action_id": action.action_id,
                "payload": payload,
            }
            if action.effect == "edit":
                result["result"] = f"EDITED:{action.resource}"
                result["file_content"] = payload.get("content", "patched")
            self.history.append(result)
            self.busy_s += time.perf_counter() - t0
            self._active -= 1
            if self._active == 0:
                self._last_zero = time.perf_counter()
            return result


@dataclass
class BatchCapableExecutor:
    """Dispatch compatible prepared actions together."""

    delay_s: float = 0.06
    history: list[dict[str, Any]] = field(default_factory=list)
    idle_s: float = 0.0
    busy_s: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)

    async def execute_batch(
        self, items: list[tuple[TaskAction, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        await asyncio.sleep(self.delay_s)
        out = []
        for action, payload in items:
            r = {
                "ok": True,
                "result": f"DONE:{action.effect}@{action.resource}",
                "action_id": action.action_id,
                "payload": payload,
            }
            out.append(r)
            self.history.append(r)
        self.batch_sizes.append(len(items))
        self.busy_s += time.perf_counter() - t0
        return out

    async def execute(self, action: TaskAction, payload: dict[str, Any]) -> dict[str, Any]:
        rows = await self.execute_batch([(action, payload)])
        return rows[0]


@dataclass
class CoordinatorMetrics:
    total_task_s: float = 0.0
    executor_idle_s: float = 0.0
    executor_busy_s: float = 0.0
    prep_overlap_s: float = 0.0
    duplicate_dispatches: int = 0
    inference_s: float = 0.0
    n_workers: int = 1
    cross_attn: bool = True
    mode: str = "sequential"


@dataclass
class TimelineEvent:
    at: float
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinatorResult:
    record: SharedTaskRecord
    batch: BatchRResult | None
    metrics: CoordinatorMetrics
    dispatch_log: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    satisfied: bool = False
    final_fingerprint: str = ""
    host_launch: dict[str, Any] = field(default_factory=dict)


def _seed_actions_for_goal(record: SharedTaskRecord) -> None:
    """Materialize discoverable actions from the goal (no fixed worker assignment)."""
    g = record.goal.lower()
    if any(k in g for k in ("browser", "navigate", "url", "http://", "https://")):
        # Browser path: sequential nav steps on one page stream
        a, _ = record.propose(
            effect="navigate", resource="page:start", args={"url": "about:blank"},
            worker_id="seed",
        )
        b, _ = record.propose(
            effect="click", resource="page:start", args={"target": "search"},
            depends_on=[a.action_id], worker_id="seed",
        )
        record.propose(
            effect="type", resource="page:start", args={"text": "query"},
            depends_on=[b.action_id], worker_id="seed",
        )
        return
    if any(k in g for k in ("application", "notepad", "gui", "window")):
        for name in ("open", "type", "save"):
            record.propose(
                effect="process", resource=f"app:{name}",
                args={"name": name}, worker_id="seed",
            )
        return
    if "shell" in g or "command" in g or "powershell" in g:
        for name in ("list", "write", "verify"):
            record.propose(
                effect="process", resource=f"shell:{name}",
                args={"name": name}, worker_id="seed",
            )
        return
    if "test" in g and ("fail" in g or "fix" in g or "code" in g):
        # Code task: investigate, edit(s), retest — edits may conflict on same file
        inv, _ = record.propose(
            effect="investigate", resource="tests", args={"suite": "unit"},
            worker_id="seed",
        )
        e1, _ = record.propose(
            effect="edit", resource="src/app.py",
            args={"patch": "fix_a", "line": 10},
            depends_on=[inv.action_id], worker_id="seed",
        )
        # Duplicate proposal (different wording, same effect identity)
        record.propose(
            effect="edit", resource="src/app.py",
            args={"patch": "fix_a", "line": 10, "description": "apply the null check fix"},
            depends_on=[inv.action_id], worker_id="seed",
        )
        # Conflicting edit on same resource
        record.propose(
            effect="edit", resource="src/app.py",
            args={"patch": "fix_b", "line": 12},
            depends_on=[inv.action_id], worker_id="seed",
        )
        record.propose(
            effect="test", resource="tests",
            args={"suite": "unit"},
            depends_on=[e1.action_id], worker_id="seed",
        )
        return
    if "parallel" in g or "independent" in g:
        for name in ("alpha", "beta", "gamma", "delta"):
            record.propose(
                effect="process", resource=f"item:{name}",
                args={"name": name}, worker_id="seed",
            )
        return
    # Default sequential pipeline
    a, _ = record.propose(effect="read", resource="inbox", args={}, worker_id="seed")
    b, _ = record.propose(
        effect="write", resource="outbox", args={"text": "ok"},
        depends_on=[a.action_id], worker_id="seed",
    )
    record.propose(
        effect="send", resource="outbox", args={},
        depends_on=[b.action_id], worker_id="seed",
    )


class RCoordinator:
    """R runs N batched workers; prepares ahead; dispatches to executors."""

    def __init__(
        self,
        *,
        n_workers: int = DEFAULT_MULTI_R_WORKERS,
        cross_attn: bool = True,
        max_new_tokens: int = 12,
        executor: Any | None = None,
        mode: str = "sequential",  # sequential | batch | parallel
        crossing: Crossing | None = None,
        worker_id_prefix: str = "",
    ) -> None:
        self.n_workers = n_workers
        self.cross_attn = cross_attn
        self.mode = mode
        self.worker_id_prefix = worker_id_prefix
        try:
            self.team = BatchedRTeam(
                n_workers=n_workers,
                max_new_tokens=max_new_tokens,
                cross_attn=cross_attn,
                crossing=crossing,
                worker_id_prefix=worker_id_prefix,
            )
            self.team.ensure_launched()
        except MultiRLaunchError:
            raise
        except Exception as e:
            raise MultiRLaunchError(
                n_workers, 0, f"R coordinator failed to launch workers: {e}",
            ) from e
        if executor is not None:
            self.executor = executor
        elif mode == "parallel":
            self.executor = ParallelExecutor()
        elif mode == "batch":
            self.executor = BatchCapableExecutor()
        else:
            self.executor = SequentialExecutor()

    async def run_task(
        self,
        goal: str,
        *,
        constraints: str = "",
        source_material: str = "",
        force_duplicate: bool = True,
        influence_probe: bool = False,
        seed_actions: bool = True,
    ) -> CoordinatorResult:
        record = SharedTaskRecord.create(
            goal, constraints=constraints, source_material=source_material,
        )
        if seed_actions:
            _seed_actions_for_goal(record)
        if force_duplicate:
            # Two workers will both try the same edit — identity collapses
            record.propose(
                effect="edit", resource="src/app.py",
                args={"patch": "fix_a", "line": 10, "wording": "worker says fix null"},
                worker_id="dup_probe",
            )

        t0 = time.perf_counter()
        prep_overlap = 0.0
        dispatch_log: list[dict[str, Any]] = []
        timeline: list[TimelineEvent] = []
        self._timeline = timeline
        self._t0 = t0

        def _ev(kind: str, **detail: Any) -> None:
            timeline.append(TimelineEvent(
                at=time.perf_counter() - t0, kind=kind, detail=detail,
            ))

        self._emit_tl = _ev
        _ev("task_start", goal=goal, n_workers=self.n_workers)

        # Inference pass (batched R) — workers see whole task
        batch = await asyncio.to_thread(
            self.team.run_inference,
            record,
            influence_probe=influence_probe and self.cross_attn,
        )
        inference_s = batch.timing.finished_at - batch.timing.compute_started_at
        _ev("inference_done", s=inference_s, cross_attn=self.cross_attn)

        # First tick: only dependency-ready actions (so follow-ups are prepared
        # during the first execute — visible prep-ahead on the timeline).
        await self._claim_and_prepare_round(record, speculative=False)

        # Dispatch loop: prepare ahead while executor runs
        pending_prep = asyncio.create_task(self._prep_loop(record))
        try:
            if self.mode == "batch" and isinstance(self.executor, BatchCapableExecutor):
                await self._dispatch_batches(record, dispatch_log)
            elif self.mode == "parallel":
                await self._dispatch_parallel(record, dispatch_log)
            else:
                overlap_acc = await self._dispatch_sequential(record, dispatch_log)
                prep_overlap += overlap_acc
        finally:
            pending_prep.cancel()
            try:
                await pending_prep
            except asyncio.CancelledError:
                pass

        total = time.perf_counter() - t0
        idle = float(getattr(self.executor, "idle_s", 0.0) or 0.0)
        busy = float(getattr(self.executor, "busy_s", 0.0) or 0.0)
        # Idle waiting for next action = gaps when executor free but nothing dispatchable
        metrics = CoordinatorMetrics(
            total_task_s=total,
            executor_idle_s=idle,
            executor_busy_s=busy,
            prep_overlap_s=prep_overlap,
            duplicate_dispatches=record.duplicate_dispatches_blocked,
            inference_s=inference_s,
            n_workers=self.n_workers,
            cross_attn=self.cross_attn,
            mode=self.mode,
        )
        satisfied = self._task_satisfied(record)
        fingerprint = self._final_fingerprint(record)
        _ev("task_done", satisfied=satisfied, fingerprint=fingerprint)
        return CoordinatorResult(
            record=record,
            batch=batch,
            metrics=metrics,
            dispatch_log=dispatch_log,
            timeline=timeline,
            satisfied=satisfied,
            final_fingerprint=fingerprint,
        )

    async def _claim_and_prepare_round(
        self, record: SharedTaskRecord, *, speculative: bool = True,
    ) -> None:
        """Each worker claims at most one available action and prepares it.

        No fixed sections — any worker may take any ready action. Speculative
        claims allow preparing follow-ups while an earlier action still executes.
        """
        workers = [f"R{i}" for i in range(self.n_workers)]
        claimed_this_round: set[str] = set()
        emit = getattr(self, "_emit_tl", None)
        for wi, worker_id in enumerate(workers):
            pool = record.actions_for_prepare() if speculative else record.available_actions()
            available = [a for a in pool if a.action_id not in claimed_this_round]
            if not available:
                break
            busy_resources = {
                record.actions[aid].resource
                for aid in claimed_this_round
                if aid in record.actions
            }
            preferred = [a for a in available if a.resource not in busy_resources]
            target = (preferred or available)[0]
            claimed = record.claim(
                target.action_id, worker_id, speculative=speculative,
            )
            if claimed is None:
                continue
            claimed_this_round.add(claimed.action_id)
            wstate = self.team.workers[wi]
            wstate.claimed.append(claimed.action_id)
            payload = self._prepare_payload(claimed, record)
            if record.mark_prepared(claimed.action_id, worker_id, payload):
                wstate.prepared.append(claimed.action_id)
                if emit:
                    deps_ready = all(
                        (record.actions[d].status == "verified")
                        for d in claimed.depends_on
                        if d in record.actions
                    ) if claimed.depends_on else True
                    emit(
                        "prepare",
                        worker=worker_id,
                        action_id=claimed.action_id,
                        effect=claimed.effect,
                        resource=claimed.resource,
                        speculative=not deps_ready,
                        while_executor_busy=bool(
                            getattr(self.executor, "in_flight", 0)
                        ),
                    )
        await asyncio.sleep(0)

    def _prepare_payload(self, action: TaskAction, record: SharedTaskRecord) -> dict[str, Any]:
        payload = {
            "effect": action.effect,
            "resource": action.resource,
            "args": dict(action.args),
            "observations": list(record.observations[-5:]),
        }
        if action.effect == "edit":
            # Recheck against latest observed file content
            prior = ""
            for obs in reversed(record.observations):
                if action.resource in obs and "EDITED" in obs:
                    prior = obs
                    break
            payload["prior_content"] = prior
            payload["content"] = f"patch:{action.args.get('patch', '')}"
        if action.effect == "test":
            payload["failing"] = False
        if action.effect == "investigate":
            payload["findings"] = "null_check_needed"
        return payload

    async def _prep_loop(self, record: SharedTaskRecord) -> None:
        """Background: keep claiming/preparing while dispatch runs."""
        try:
            while not record.all_verified():
                await self._claim_and_prepare_round(record, speculative=True)
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            return

    async def _dispatch_sequential(
        self, record: SharedTaskRecord, log: list[dict[str, Any]],
    ) -> float:
        overlap = 0.0
        safety = 0
        emit = getattr(self, "_emit_tl", None)
        idle_wait = 0.0
        while safety < 100:
            safety += 1
            action = record.next_prepared()
            if action is None:
                if record.all_verified():
                    break
                tw = time.perf_counter()
                await asyncio.sleep(0.02)
                await self._claim_and_prepare_round(record, speculative=True)
                idle_wait += time.perf_counter() - tw
                if emit:
                    emit("executor_idle_wait", s=time.perf_counter() - tw)
                # Prefer prepared-with-deps; also consider available for non-speculative
                if record.next_prepared() is None:
                    stuck = [
                        a for a in record.actions.values()
                        if a.status in ("claimed", "prepared", "available")
                    ]
                    if not stuck and all(
                        a.status in ("verified", "failed", "invalidated", "dispatched")
                        for a in record.actions.values()
                    ):
                        break
                    if not stuck:
                        break
                continue
            if not record.recheck_before_dispatch(action.action_id):
                # Deps not ready yet — keep prepared, try another or wait
                await asyncio.sleep(0.01)
                continue
            if not record.mark_dispatched(action.action_id):
                record.duplicate_dispatches_blocked += 1
                if emit:
                    emit("duplicate_dispatch_blocked", action_id=action.action_id)
                continue
            if emit:
                emit(
                    "dispatch_start",
                    action_id=action.action_id,
                    effect=action.effect,
                    resource=action.resource,
                )
            prep_start = time.perf_counter()

            async def _prep_while() -> None:
                for _ in range(max(2, self.n_workers)):
                    await self._claim_and_prepare_round(record, speculative=True)
                    await asyncio.sleep(0)

            prep_task = asyncio.create_task(_prep_while())
            result = await self.executor.execute(
                action, action.prepared_payload or {},
            )
            await prep_task
            overlap += time.perf_counter() - prep_start
            if emit:
                emit(
                    "dispatch_end",
                    action_id=action.action_id,
                    effect=action.effect,
                    result=result.get("result"),
                    prep_during_exec=True,
                )
            uncertain = bool(result.get("uncertain"))
            if uncertain:
                record.mark_verified(
                    action.action_id, result.get("result", ""), ok=False, uncertain=True,
                )
            else:
                record.mark_verified(
                    action.action_id,
                    result.get("result", ""),
                    ok=bool(result.get("ok", True)),
                )
            log.append({
                "action_id": action.action_id,
                "effect": action.effect,
                "resource": action.resource,
                "result": result.get("result"),
            })
            if action.effect == "edit":
                self._invalidate_stale_edits(record, action)
        # Fold idle waits into executor idle metric when SequentialExecutor
        if hasattr(self.executor, "idle_s"):
            self.executor.idle_s = float(self.executor.idle_s or 0.0) + idle_wait
        return overlap

    def _final_fingerprint(self, record: SharedTaskRecord) -> str:
        parts = []
        for a in sorted(record.actions.values(), key=lambda x: (x.resource, x.effect, x.action_id)):
            if a.status == "verified":
                parts.append(f"{a.effect}@{a.resource}={a.result}")
        return "|".join(parts)

    async def _dispatch_parallel(
        self, record: SharedTaskRecord, log: list[dict[str, Any]],
    ) -> None:
        safety = 0
        emit = getattr(self, "_emit_tl", None)
        while safety < 100:
            safety += 1
            # One prep tick per worker so N workers can fill the prepared queue
            for _ in range(max(1, self.n_workers)):
                await self._claim_and_prepare_round(record, speculative=True)
                if not record.actions_for_prepare():
                    break
            ready: list[TaskAction] = []
            seen_resources: set[str] = set()
            with record._lock:
                for a in sorted(record.actions.values(), key=lambda x: x.updated_at):
                    if a.status != "prepared":
                        continue
                    if not record._deps_met_unlocked(a):
                        continue
                    if a.resource in seen_resources:
                        record.conflicts_ordered += 1
                        continue
                    if record._resource_blocked_unlocked(a):
                        continue
                    ready.append(a)
                    seen_resources.add(a.resource)
            if not ready:
                if record.all_verified():
                    break
                if not record.actions_for_prepare() and not any(
                    a.status in ("claimed", "prepared") for a in record.actions.values()
                ):
                    break
                await asyncio.sleep(0.02)
                continue

            async def _one(act: TaskAction) -> None:
                if not record.recheck_before_dispatch(act.action_id):
                    return
                if not record.mark_dispatched(act.action_id):
                    record.duplicate_dispatches_blocked += 1
                    return
                if emit:
                    emit(
                        "dispatch_start",
                        action_id=act.action_id,
                        effect=act.effect,
                        resource=act.resource,
                        worker=act.claimed_by,
                        concurrent=True,
                    )
                result = await self.executor.execute(act, act.prepared_payload or {})
                if emit:
                    emit(
                        "dispatch_end",
                        action_id=act.action_id,
                        effect=act.effect,
                        result=result.get("result"),
                        worker=act.claimed_by,
                        concurrent=True,
                    )
                record.mark_verified(
                    act.action_id, result.get("result", ""), ok=bool(result.get("ok", True)),
                )
                log.append({
                    "action_id": act.action_id,
                    "effect": act.effect,
                    "resource": act.resource,
                    "result": result.get("result"),
                    "worker": act.claimed_by,
                })

            # Overlap independent actions on distinct host subagents
            if emit:
                emit("parallel_batch", n=len(ready),
                     workers=[a.claimed_by for a in ready])
            await asyncio.gather(*[_one(a) for a in ready])

    async def _dispatch_batches(
        self, record: SharedTaskRecord, log: list[dict[str, Any]],
    ) -> None:
        assert isinstance(self.executor, BatchCapableExecutor)
        safety = 0
        while safety < 50:
            safety += 1
            await self._claim_and_prepare_round(record)
            batch: list[tuple[TaskAction, dict[str, Any]]] = []
            seen: set[str] = set()
            with record._lock:
                for a in sorted(record.actions.values(), key=lambda x: x.updated_at):
                    if a.status != "prepared":
                        continue
                    if not record._deps_met_unlocked(a):
                        continue
                    if a.resource in seen or record._resource_blocked_unlocked(a):
                        continue
                    batch.append((a, a.prepared_payload or {}))
                    seen.add(a.resource)
            if not batch:
                if record.all_verified():
                    break
                await asyncio.sleep(0.02)
                if not record.available_actions():
                    break
                continue
            for a, _ in batch:
                if not record.recheck_before_dispatch(a.action_id):
                    continue
                if not record.mark_dispatched(a.action_id):
                    record.duplicate_dispatches_blocked += 1
            live = [(a, p) for a, p in batch if a.status == "dispatched"]
            if not live:
                continue
            results = await self.executor.execute_batch(live)
            for (a, _), r in zip(live, results):
                record.mark_verified(a.action_id, r.get("result", ""), ok=bool(r.get("ok", True)))
                log.append({
                    "action_id": a.action_id,
                    "effect": a.effect,
                    "resource": a.resource,
                    "result": r.get("result"),
                })

    def _invalidate_stale_edits(self, record: SharedTaskRecord, done: TaskAction) -> None:
        for a in list(record.actions.values()):
            if a.action_id == done.action_id:
                continue
            if a.resource != done.resource:
                continue
            if a.status == "prepared" and a.effect == "edit":
                record.invalidate_prepared(a.action_id, "resource_updated")
                # Re-propose path: set back to available for re-prep against new state
                with record._lock:
                    a.status = "available"
                    a.claimed_by = None
                    a.prepared_payload = None
                    a.touch()

    def _task_satisfied(self, record: SharedTaskRecord) -> bool:
        # At least one verified action and no failed critical path
        verified = [a for a in record.actions.values() if a.status == "verified"]
        if not verified:
            return False
        g = record.goal.lower()
        if "test" in g or "code" in g:
            return any(a.effect == "test" and a.verified for a in verified) or (
                any(a.effect == "edit" and a.verified for a in verified)
                and any(a.effect == "investigate" and a.verified for a in verified)
            )
        if "parallel" in g or "independent" in g:
            return len(verified) >= 3
        return all(
            a.status in ("verified", "invalidated", "failed")
            for a in record.actions.values()
        )
