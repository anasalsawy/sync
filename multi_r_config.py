"""Runtime Multi-R worker count — provisional default is three per task.

When L authorizes a task, main R starts a new task group with exactly three
concurrent R workers in one inference batch. Additional authorized tasks each
get their own three-worker group; groups run concurrently (three tasks → nine
R workers). One-worker mode is an explicit eval control only.
"""
from __future__ import annotations

import os

# Provisional production default — exactly three concurrent R workers per task
DEFAULT_MULTI_R_WORKERS = 3

# Eval control: solo R (no batch sharing). Never the companion default.
EVAL_CONTROL_ONE_WORKER = 1

# Scale candidates for deciding whether to raise the default
SCALE_CANDIDATES = (1, 3, 5)


class MultiRLaunchError(RuntimeError):
    """Raised when the runtime cannot start the requested R worker batch."""

    def __init__(self, requested: int, started: int, reason: str = "") -> None:
        self.requested = requested
        self.started = started
        self.reason = reason or (
            f"requested {requested} R workers, started {started}"
        )
        super().__init__(self.reason)


def multi_r_worker_count(*, override: int | None = None) -> int:
    """Resolve worker count: explicit override → env → provisional default (3)."""
    if override is not None:
        n = int(override)
    else:
        raw = os.environ.get("SYNC_MULTI_R_WORKERS", "").strip()
        n = int(raw) if raw else DEFAULT_MULTI_R_WORKERS
    if n < 1:
        raise MultiRLaunchError(n, 0, f"invalid Multi-R worker count: {n}")
    return n
