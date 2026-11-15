"""Durable Conversation Board and Execution Blueprint for the companion.

Blueprint holds plans and lane state. Sketchpad holds observed tool results.
On resume, both restore. Each proposed action records whether it was
dispatched and whether its result was observed so a restart cannot silently
repeat an uncertain action.
"""
from __future__ import annotations

import json
import sqlite3
import threading
time_module = __import__("time")
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BoardTurn:
    role: str  # user | L
    text: str
    at: float = field(default_factory=time_module.time)


@dataclass
class ConversationBoard:
    session_id: str
    turns: list[BoardTurn] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    commitments: list[str] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.turns.append(BoardTurn(role="user", text=text))

    def add_l(self, text: str) -> None:
        self.turns.append(BoardTurn(role="L", text=text))

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [asdict(t) for t in self.turns[-40:]],
            "open_questions": list(self.open_questions),
            "commitments": list(self.commitments),
        }

    def delta_payload(self) -> dict[str, Any]:
        return {"kind": "board", **self.snapshot()}


@dataclass
class ActionRecord:
    """One proposed host action with dispatch/observe durability."""

    action_id: str
    lane_id: str
    task_id: str
    version: int
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"  # proposed | authorized | rejected | superseded | dispatched | observed
    dispatched: bool = False
    observed: bool = False
    result: str = ""
    authorized_by: str = ""
    verdict_version: int | None = None
    created_at: float = field(default_factory=time_module.time)
    updated_at: float = field(default_factory=time_module.time)

    def mark_dispatched(self) -> None:
        self.dispatched = True
        self.status = "dispatched"
        self.updated_at = time_module.time()

    def mark_observed(self, result: str) -> None:
        self.observed = True
        self.result = result
        self.status = "observed"
        self.updated_at = time_module.time()

    def mark_superseded(self) -> None:
        if self.dispatched and not self.observed:
            # Uncertain — keep dispatched flag; do not re-issue blindly on resume
            self.status = "superseded_uncertain"
        else:
            self.status = "superseded"
        self.updated_at = time_module.time()

    def safe_to_retry(self) -> bool:
        """Restart may only re-issue actions that never dispatched."""
        return not self.dispatched


@dataclass
class LaneStatus:
    lane_id: str
    goal: str
    status: str = "running"  # running | done | cancelled | redirected
    last_result: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list)
    action_version: int = 0
    started_at: float = field(default_factory=time_module.time)
    finished_at: float | None = None


@dataclass
class ExecutionBlueprint:
    session_id: str
    lanes: dict[str, LaneStatus] = field(default_factory=dict)
    sketchpad: list[str] = field(default_factory=list)  # grounded tool facts only
    actions: dict[str, ActionRecord] = field(default_factory=dict)

    def upsert_lane(self, lane: LaneStatus) -> None:
        self.lanes[lane.lane_id] = lane

    def propose_action(
        self,
        *,
        lane_id: str,
        task_id: str,
        name: str,
        args: dict[str, Any] | None = None,
        version: int | None = None,
    ) -> ActionRecord:
        lane = self.lanes.get(lane_id)
        if version is None:
            version = (lane.action_version + 1) if lane else 1
            if lane is not None:
                lane.action_version = version
        action_id = f"{lane_id}:v{version}:{name}"
        rec = ActionRecord(
            action_id=action_id,
            lane_id=lane_id,
            task_id=task_id,
            version=version,
            name=name,
            args=dict(args or {}),
        )
        self.actions[action_id] = rec
        return rec

    def supersede_pending(self, lane_id: str) -> list[ActionRecord]:
        """Invalidate proposed/authorized actions that have not started."""
        out: list[ActionRecord] = []
        for rec in self.actions.values():
            if rec.lane_id != lane_id:
                continue
            if rec.dispatched:
                continue
            if rec.status in ("proposed", "authorized", "rejected"):
                rec.mark_superseded()
                out.append(rec)
        return out

    def uncertain_actions(self) -> list[ActionRecord]:
        """Dispatched but not observed — must not silently repeat on resume."""
        return [a for a in self.actions.values() if a.dispatched and not a.observed]

    def record_observation(self, lane_id: str, fact: str, *, action_id: str | None = None) -> None:
        """Sketchpad accepts only observed tool results, not plans."""
        fact = (fact or "").strip()
        if not fact:
            return
        self.sketchpad.append(fact)
        if len(self.sketchpad) > 200:
            self.sketchpad = self.sketchpad[-200:]
        lane = self.lanes.get(lane_id)
        if lane is not None:
            lane.last_result = fact
            lane.steps.append({"at": time_module.time(), "result": fact, "action_id": action_id})
        if action_id and action_id in self.actions:
            self.actions[action_id].mark_observed(fact)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "lanes": {k: asdict(v) for k, v in self.lanes.items()},
            "sketchpad": list(self.sketchpad[-50:]),
            "actions": {k: asdict(v) for k, v in self.actions.items()},
        }

    def delta_payload(self) -> dict[str, Any]:
        return {"kind": "blueprint", **self.snapshot()}


class BoardStore:
    """SQLite persistence for boards + blueprints."""

    def __init__(self, path: str | Path = "companion.db") -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                board_json TEXT NOT NULL,
                blueprint_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._db.commit()

    def save(self, board: ConversationBoard, blueprint: ExecutionBlueprint) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO sessions (session_id, board_json, blueprint_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  board_json=excluded.board_json,
                  blueprint_json=excluded.blueprint_json,
                  updated_at=excluded.updated_at
                """,
                (
                    board.session_id,
                    json.dumps(board.snapshot()),
                    json.dumps(blueprint.snapshot()),
                    time_module.time(),
                ),
            )
            self._db.commit()

    def restore(self, session_id: str) -> tuple[ConversationBoard, ExecutionBlueprint] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT board_json, blueprint_json FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        board_raw, bp_raw = json.loads(row[0]), json.loads(row[1])
        board = ConversationBoard(
            session_id=session_id,
            turns=[BoardTurn(**t) for t in board_raw.get("turns", [])],
            open_questions=list(board_raw.get("open_questions", [])),
            commitments=list(board_raw.get("commitments", [])),
        )
        lanes = {}
        for lid, raw in (bp_raw.get("lanes") or {}).items():
            raw = dict(raw)
            raw.setdefault("action_version", 0)
            lanes[lid] = LaneStatus(**raw)
        actions = {}
        for aid, raw in (bp_raw.get("actions") or {}).items():
            actions[aid] = ActionRecord(**raw)
        blueprint = ExecutionBlueprint(
            session_id=session_id,
            lanes=lanes,
            sketchpad=list(bp_raw.get("sketchpad", [])),
            actions=actions,
        )
        return board, blueprint

    def new_session_id(self) -> str:
        return uuid.uuid4().hex

    def close(self) -> None:
        with self._lock:
            self._db.close()
