"""Shared task record for multi-R workers on one task.

One record holds available actions, dependencies, claims, and verified results.
Any worker can discover or propose an action. Before preparing it, the worker
must atomically claim the logical action using an identity tied to its
intended effect — two workers proposing the same update resolve to the same
action even if they describe it differently.

Lifecycle: claimed → prepared → dispatched → verified.
Resource tags order conflicting actions that change the same thing.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def action_identity(
    *,
    effect: str,
    resource: str,
    args: dict[str, Any] | None = None,
) -> str:
    """Stable id from intended effect + resource (not free-form wording)."""
    args = args or {}
    # Canonical arg slice that defines the effect (ignore descriptive fluff)
    keys = sorted(k for k in args if k not in ("description", "note", "rationale", "wording"))
    parts = [effect.strip().lower(), resource.strip().lower()]
    for k in keys:
        parts.append(f"{k}={args[k]!r}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class TaskAction:
    action_id: str
    effect: str
    resource: str
    args: dict[str, Any] = field(default_factory=dict)
    status: str = "available"  # available|claimed|prepared|dispatched|verified|invalidated|failed
    claimed_by: str | None = None
    depends_on: list[str] = field(default_factory=list)
    prepared_payload: dict[str, Any] | None = None
    result: str = ""
    verified: bool = False
    uncertain: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    claim_generation: int = 0

    def touch(self) -> None:
        self.updated_at = time.time()


@dataclass
class SharedTaskRecord:
    """Single shared board for one task — no fixed section assignment."""

    task_id: str
    goal: str
    constraints: str = ""
    source_material: str = ""
    observations: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)
    actions: dict[str, TaskAction] = field(default_factory=dict)
    # resource → ordered action_ids that mutate it (conflict ordering)
    resource_order: dict[str, list[str]] = field(default_factory=dict)
    duplicate_claim_attempts: int = 0
    duplicate_dispatches_blocked: int = 0
    conflicts_ordered: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def create(
        cls,
        goal: str,
        *,
        constraints: str = "",
        source_material: str = "",
        task_id: str | None = None,
    ) -> "SharedTaskRecord":
        return cls(
            task_id=task_id or uuid.uuid4().hex[:12],
            goal=goal,
            constraints=constraints,
            source_material=source_material,
        )

    def snapshot_for_worker(self) -> dict[str, Any]:
        """Whole task view — every worker gets everything."""
        with self._lock:
            return {
                "task_id": self.task_id,
                "goal": self.goal,
                "constraints": self.constraints,
                "source_material": self.source_material,
                "observations": list(self.observations[-40:]),
                "progress": list(self.progress[-40:]),
                "actions": {k: asdict(v) for k, v in self.actions.items()},
                "available": [
                    asdict(a) for a in self.actions.values()
                    if a.status == "available"
                ],
                "ready_to_prepare": [
                    asdict(a) for a in self.actions.values()
                    if a.status in ("available",) and self._deps_met_unlocked(a)
                ],
            }

    def add_observation(self, text: str) -> None:
        with self._lock:
            self.observations.append(text)
            if len(self.observations) > 200:
                self.observations = self.observations[-200:]

    def add_progress(self, text: str) -> None:
        with self._lock:
            self.progress.append(text)
            if len(self.progress) > 200:
                self.progress = self.progress[-200:]

    def propose(
        self,
        *,
        effect: str,
        resource: str,
        args: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
        worker_id: str = "",
    ) -> tuple[TaskAction, bool]:
        """Propose an action. Returns (action, is_new). Same effect → same id."""
        args = dict(args or {})
        aid = action_identity(effect=effect, resource=resource, args=args)
        with self._lock:
            if aid in self.actions:
                self.duplicate_claim_attempts += 1
                return self.actions[aid], False
            action = TaskAction(
                action_id=aid,
                effect=effect,
                resource=resource,
                args=args,
                status="available",
                depends_on=list(depends_on or []),
            )
            self.actions[aid] = action
            self.resource_order.setdefault(resource, []).append(aid)
            self.progress.append(
                f"proposed:{aid}:{effect}@{resource} by={worker_id or '?'}"
            )
            return action, True

    def claim(
        self,
        action_id: str,
        worker_id: str,
        *,
        speculative: bool = False,
    ) -> TaskAction | None:
        """Atomically claim a logical action. Returns None if already claimed.

        ``speculative=True`` allows claiming follow-ups before dependencies are
        verified so workers can prepare ahead while an earlier action executes.
        Dispatch still requires deps via ``recheck_before_dispatch``.
        """
        with self._lock:
            action = self.actions.get(action_id)
            if action is None:
                return None
            if action.status not in ("available",):
                self.duplicate_claim_attempts += 1
                return None
            if not speculative and not self._deps_met_unlocked(action):
                return None
            # Resource conflict: earlier unverified mutation blocks non-speculative
            if self._resource_blocked_unlocked(action):
                self.conflicts_ordered += 1
                if not speculative:
                    return None
                # Speculative prep of a conflicting later edit is OK; dispatch waits
            action.status = "claimed"
            action.claimed_by = worker_id
            action.claim_generation += 1
            action.touch()
            return action

    def actions_for_prepare(self) -> list[TaskAction]:
        """Available actions including those waiting on deps (prep-ahead)."""
        with self._lock:
            return [a for a in self.actions.values() if a.status == "available"]

    def claim_or_propose(
        self,
        *,
        effect: str,
        resource: str,
        args: dict[str, Any] | None = None,
        depends_on: list[str] | None = None,
        worker_id: str,
    ) -> TaskAction | None:
        """Propose (dedupe) then atomically claim. One logical action wins."""
        action, _ = self.propose(
            effect=effect, resource=resource, args=args,
            depends_on=depends_on, worker_id=worker_id,
        )
        return self.claim(action.action_id, worker_id)

    def mark_prepared(self, action_id: str, worker_id: str, payload: dict[str, Any]) -> bool:
        with self._lock:
            action = self.actions.get(action_id)
            if action is None or action.claimed_by != worker_id:
                return False
            if action.status != "claimed":
                return False
            action.prepared_payload = dict(payload)
            action.status = "prepared"
            action.touch()
            return True

    def invalidate_prepared(self, action_id: str, reason: str = "") -> None:
        with self._lock:
            action = self.actions.get(action_id)
            if action is None:
                return
            if action.status == "prepared":
                action.status = "invalidated"
                action.touch()
                self.progress.append(f"invalidated:{action_id}:{reason}")

    def recheck_before_dispatch(self, action_id: str) -> bool:
        """Return True if prepared action's prerequisites still hold.

        Unmet deps do not invalidate — speculative prep waits until deps clear.
        """
        with self._lock:
            action = self.actions.get(action_id)
            if action is None or action.status != "prepared":
                return False
            if not self._deps_met_unlocked(action):
                return False
            if self._resource_blocked_unlocked(action):
                self.conflicts_ordered += 1
                return False
            return True

    def mark_dispatched(self, action_id: str) -> bool:
        with self._lock:
            action = self.actions.get(action_id)
            if action is None or action.status != "prepared":
                if action and action.status == "dispatched":
                    self.duplicate_dispatches_blocked += 1
                return False
            if not self._deps_met_unlocked(action):
                return False
            action.status = "dispatched"
            action.touch()
            return True

    def mark_verified(
        self,
        action_id: str,
        result: str,
        *,
        ok: bool = True,
        uncertain: bool = False,
    ) -> None:
        with self._lock:
            action = self.actions.get(action_id)
            if action is None:
                return
            action.result = result
            action.uncertain = uncertain
            action.verified = ok and not uncertain
            action.status = "verified" if ok else "failed"
            action.touch()
            self.observations.append(f"{action.effect}@{action.resource}: {result}")
            self.progress.append(f"verified:{action_id}:{action.status}")

    def next_prepared(self) -> TaskAction | None:
        """Next prepared action whose prerequisites still hold (dispatch order)."""
        with self._lock:
            candidates = [
                a for a in self.actions.values()
                if a.status == "prepared" and self._deps_met_unlocked(a)
            ]
            if not candidates:
                return None
            # Prefer actions whose resource has no earlier in-flight conflict
            for a in sorted(candidates, key=lambda x: x.updated_at):
                if not self._resource_blocked_unlocked(a):
                    return a
            self.conflicts_ordered += 1
            return None

    def available_actions(self) -> list[TaskAction]:
        with self._lock:
            return [
                a for a in self.actions.values()
                if a.status == "available" and self._deps_met_unlocked(a)
            ]

    def all_verified(self) -> bool:
        with self._lock:
            if not self.actions:
                return False
            return all(
                a.status in ("verified", "failed", "invalidated")
                for a in self.actions.values()
            )

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            by_status: dict[str, int] = {}
            for a in self.actions.values():
                by_status[a.status] = by_status.get(a.status, 0) + 1
            return {
                "task_id": self.task_id,
                "n_actions": len(self.actions),
                "by_status": by_status,
                "duplicate_claim_attempts": self.duplicate_claim_attempts,
                "duplicate_dispatches_blocked": self.duplicate_dispatches_blocked,
                "conflicts_ordered": self.conflicts_ordered,
            }

    def _deps_met_unlocked(self, action: TaskAction) -> bool:
        for dep in action.depends_on:
            other = self.actions.get(dep)
            if other is None or other.status != "verified":
                return False
        return True

    def _resource_blocked_unlocked(self, action: TaskAction) -> bool:
        """True if an earlier action on the same resource is still in flight."""
        order = self.resource_order.get(action.resource) or []
        for aid in order:
            if aid == action.action_id:
                break
            other = self.actions.get(aid)
            if other is None:
                continue
            if other.status in ("claimed", "prepared", "dispatched"):
                return True
        return False
