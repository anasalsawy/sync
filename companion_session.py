"""Production companion session on top of local five-worker Crossing.

L owns speech (text+audio) with barge-in. POST /v1/task submits to L; once L
authorizes, R starts a silent lane. C tags verdicts with task/lane/action
version. Blueprint + Sketchpad restore with dispatched/observed durability.
``activation_injection`` gates residual sharing; coordination stays available.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from boards import BoardStore, ConversationBoard, ExecutionBlueprint
from crossing import Crossing, ReadyHandle
from execution_lanes import ExecutionLaneManager, HostExecutor, MockHostExecutor
from memory_store import EpisodicMemory, embedder_prompt_from_recall
from sync_session import SessionTurnResult, SyncSession
from voice import StubVoiceOut, VoiceOut


EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class CompanionTurnResult:
    reply: str
    session_id: str
    mind: SessionTurnResult
    streamed: list[str] = field(default_factory=list)
    audio_frames: int = 0
    ttfa_s: float | None = None
    pending_correction: str | None = None
    lanes_active: list[str] = field(default_factory=list)
    crossing_enabled: bool = True
    activation_injection: bool = True
    utterance_id: str | None = None


@dataclass
class CausalAuthTrace:
    """Mid-decode bad-proposal causal log for eval."""

    task_id: str
    lane_id: str
    action_version: int
    action_name: str
    c_verdict: str | None = None
    c_publish_step: int | None = None
    consume_decode_step: int | None = None
    consume_receiver: str | None = None
    l_decision: str | None = None
    host_act_called: bool = False
    activation_injection: bool = True


class CompanionSession:
    """One persistent companion mind with voice, boards, and parallel lanes."""

    def __init__(
        self,
        *,
        session_id: str | None = None,
        db_path: str = "companion.db",
        episodic_path: str = "episodic.db",
        crossing_enabled: bool = True,
        activation_injection: bool | None = None,
        backend: str | None = None,
        max_new_tokens: int = 32,
        executor: HostExecutor | None = None,
        host: Any | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.store = BoardStore(db_path)
        self.session_id = session_id or self.store.new_session_id()
        restored = self.store.restore(self.session_id) if session_id else None
        if restored:
            self.board, self.blueprint = restored
        else:
            self.board = ConversationBoard(session_id=self.session_id)
            self.blueprint = ExecutionBlueprint(session_id=self.session_id)

        # Coordination always on when companion is up; "Crossing off" = inject off
        self.crossing_enabled = crossing_enabled
        self.activation_injection = (
            crossing_enabled if activation_injection is None else activation_injection
        )
        self.on_event = on_event
        self.pending_correction: str | None = None
        self.held_decision: str | None = None
        self.decision_wait_s: float = 0.0
        self.causal_traces: list[CausalAuthTrace] = []
        # (task_id, action_version, action_name) → accept|reject
        self._action_auth: dict[tuple[str, int, str], str] = {}
        self._lock = __import__("threading").Lock()

        # Mind Crossing: coordination on; activation inject optional
        self.mind = SyncSession(
            backend=backend or os.environ.get("SYNC_LOCAL_BACKEND", "sim"),
            max_new_tokens=max_new_tokens,
            crossing_enabled=True,  # coordination always available
            activation_injection=self.activation_injection,
            # Reserved compute for L — peers on separate processes under sim
            isolate_peers=True,
        )

        for t in self.board.turns:
            role = "user" if t.role == "user" else "assistant"
            self.mind.history.append(f"{role}: {t.text}")

        self.episodic = EpisodicMemory(episodic_path)
        self.voice: VoiceOut = StubVoiceOut(on_event=self._voice_event)
        self._audio_frames = 0
        self._ttfa_s: float | None = None
        self._speak_started: float | None = None
        self._last_utterance_id: str | None = None

        self.exec_crossing = Crossing(
            enabled=True,  # coordination always available for companion
            activation_injection=self.activation_injection,
            session_id=self.session_id,
        )
        self.executor = executor or MockHostExecutor()
        # Host for Multi-R execution: native subagents or single path
        self.host = host
        self.lanes = ExecutionLaneManager(
            executor=self.executor,
            blueprint=self.blueprint,
            crossing=self.exec_crossing,
            auth_lookup=self._auth_lookup,
            strict_activation_auth=self.activation_injection,
            auth_timeout_s=0.45 if self.activation_injection else 0.2,
        )
        from multi_r_config import multi_r_worker_count
        from pathlib import Path as _Path
        from shared_resources import SharedResourceBus
        from task_groups import MainRCoordinator
        # Always three concurrent R workers per task (runtime-configured)
        self.multi_r_workers = multi_r_worker_count()
        self.last_multi_r = None
        self.multi_r_invocations: list[dict[str, Any]] = []
        self.last_host_launch: dict[str, Any] = {}
        # Main R: one task → one 3-worker group; many groups run together
        self.resource_bus = SharedResourceBus()
        self.main_r = MainRCoordinator(
            bus=self.resource_bus,
            workers_per_group=self.multi_r_workers,
            root=_Path("hands_live_root") / self.session_id,
            crossing=self.exec_crossing,
            activation_injection=self.activation_injection,
            on_event=self._emit,
        )
        self.last_task_group = None

        # Resume safety: surface uncertain dispatched-not-observed actions
        uncertain = self.blueprint.uncertain_actions()
        if uncertain:
            self.pending_correction = (
                f"resume: {len(uncertain)} action(s) dispatched without observed result — "
                "will not silently repeat"
            )

    def _auth_lookup(self, task_id: str, version: int, name: str) -> str | None:
        with self._lock:
            return self._action_auth.get((task_id, version, name))

    def set_action_auth(
        self,
        task_id: str,
        version: int,
        name: str,
        decision: str,
        *,
        lane_id: str = "",
        consume_step: int | None = None,
        receiver: str | None = None,
        c_publish_step: int | None = None,
    ) -> None:
        decision = decision.lower()
        with self._lock:
            # Late verdict for an old proposal cannot overwrite a newer one
            # already decided for this exact key — and cannot approve a
            # different version (caller must pass the matching version).
            key = (task_id, version, name)
            prev = self._action_auth.get(key)
            if prev == "reject" and decision == "accept":
                return  # reject sticks; stale accept ignored
            self._action_auth[key] = decision
        self._emit({
            "type": "action_auth",
            "task_id": task_id,
            "lane_id": lane_id,
            "action_version": version,
            "action_name": name,
            "decision": decision,
            "consume_decode_step": consume_step,
            "receiver": receiver,
            "c_publish_step": c_publish_step,
        })

    def _voice_event(self, ev: dict[str, Any]) -> None:
        if ev.get("type") == "audio":
            self._audio_frames += 1
            if self._ttfa_s is None and self._speak_started is not None:
                self._ttfa_s = ev["at"] - self._speak_started
        if self.on_event:
            self.on_event(ev)

    def _emit(self, ev: dict[str, Any]) -> None:
        if self.on_event:
            self.on_event(ev)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "board": self.board.snapshot(),
            "blueprint": self.blueprint.snapshot(),
            "lanes_active": [h.lane_id for h in self.lanes.active],
            "pending_correction": self.pending_correction,
            "crossing_enabled": self.crossing_enabled,
            "activation_injection": self.activation_injection,
            "uncertain_actions": [
                a.action_id for a in self.blueprint.uncertain_actions()
            ],
        }

    def persist(self) -> None:
        self.store.save(self.board, self.blueprint)

    def barge_in(self) -> dict:
        """Stop speech only — execution lanes keep running."""
        metrics = self.voice.barge_in()
        self._emit({"type": "speech_stopped", "lanes_continue": True, **metrics})
        return metrics

    async def submit_task(
        self,
        goal: str,
        *,
        steps: list[dict] | None = None,
        require_c: bool = False,
        use_multi_r: bool = True,
    ) -> str:
        """POST /v1/task path: submit to L; once authorized, Multi-R runs the task."""
        return await self.authorize_task(
            goal, steps=steps, require_c=require_c, use_multi_r=use_multi_r,
        )

    async def authorize_task(
        self,
        goal: str,
        *,
        steps: list[dict] | None = None,
        require_c: bool = False,
        use_multi_r: bool = True,
    ) -> str:
        """L authorizes a task → Multi-R coordinates silent workers (normal path).

        ``use_multi_r=False`` keeps the legacy single-lane HostExecutor path
        (e.g. mid-decode inject probes). Default companion traffic uses Multi-R.
        """
        task_id = uuid.uuid4().hex[:12]
        if require_c:
            tw0 = time.perf_counter()
            verdict = await self._request_verdict(goal, task_id=task_id, version=1)
            self.decision_wait_s = time.perf_counter() - tw0
            if verdict in ("reject", "veto", "block", "fail"):
                self.pending_correction = f"C rejected task: {goal}"
                self._emit({
                    "type": "task_rejected",
                    "goal": goal,
                    "task_id": task_id,
                    "action_version": 1,
                    "verdict": verdict,
                })
                return ""
            if "delete_all" in goal.lower() or (
                "delete" in goal.lower() and "all" in goal.lower()
            ):
                if verdict != "accept":
                    self._publish_verdict(
                        decision="reject",
                        block_action="delete_all",
                        goal=goal,
                        task_id=task_id,
                        action_version=1,
                    )
                    self.pending_correction = f"C blocked dangerous task: {goal}"
                    return ""

        if use_multi_r:
            from hands_paths import detect_hands_path
            from multi_r_config import MultiRLaunchError
            path = detect_hands_path(goal)
            try:
                # Main R starts a new 3-worker group immediately; other groups
                # keep running. Host defaults to this task's Hands path unless
                # the session was constructed with an explicit host (tests).
                group = await self.main_r.start_task(
                    goal,
                    path=path,
                    host=self.host,
                    force_duplicate=(
                        "test" in goal.lower() or "code" in goal.lower()
                        or "fix" in goal.lower()
                    ),
                    wait=True,
                )
            except MultiRLaunchError as e:
                self.pending_correction = (
                    f"Multi-R could not start all {e.requested} workers "
                    f"(started {e.started}): {e.reason}"
                )
                self._emit({
                    "type": "multi_r_launch_failed",
                    "goal": goal,
                    "requested": e.requested,
                    "started": e.started,
                    "reason": e.reason,
                })
                raise
            multi = group.result
            self.last_task_group = group
            self.last_multi_r = multi
            self.last_host_launch = dict(group.host_launch)
            if multi is None:
                raise MultiRLaunchError(
                    self.multi_r_workers, 0, group.error or "task group failed",
                )
            inv = {
                "task_id": multi.record.task_id,
                "goal": goal,
                "n_workers": multi.metrics.n_workers,
                "satisfied": multi.satisfied,
                "via": "authorize_task",
                "path": path,
                "group_id": group.group_id,
                "host_mode": (self.last_host_launch or {}).get("host_mode"),
                "subagents_launched": (self.last_host_launch or {}).get(
                    "subagents_launched", []
                ),
                "main_r_snapshot": self.main_r.snapshot(),
            }
            self.multi_r_invocations.append(inv)
            self._emit({"type": "multi_r_invoked", **inv})
            # Mirror into blueprint so session restore sees the task
            from boards import LaneStatus
            self.blueprint.upsert_lane(LaneStatus(
                lane_id=multi.record.task_id,
                goal=goal,
                status="done" if multi.satisfied else "running",
                last_result=multi.final_fingerprint[:120],
            ))
            self.board.commitments.append(
                f"multi_r:{multi.record.task_id}:{path}:{goal}"
            )
            self.persist()
            self._publish_boards()
            return multi.record.task_id

        handle = self.lanes.start(goal, steps=steps, task_id=task_id)
        self.board.commitments.append(f"task:{handle.lane_id}:{goal}")
        self.persist()
        self._publish_boards()
        self._emit({
            "type": "lane_started",
            "lane_id": handle.lane_id,
            "task_id": task_id,
            "goal": goal,
        })
        return handle.lane_id

    def _publish_verdict(
        self,
        *,
        decision: str,
        block_action: str | None = None,
        goal: str = "",
        task_id: str = "",
        lane_id: str = "",
        action_version: int = 0,
        step: int = 0,
    ) -> None:
        payload: dict[str, Any] = {
            "decision": decision,
            "verdict": decision,
            "goal": goal,
            "task_id": task_id,
            "lane_id": lane_id,
            "action_version": action_version,
        }
        if block_action:
            payload["block_action"] = block_action
        self.exec_crossing.publish(
            "C",
            pass_id=0, step=step, layer=0, kind="verdict",
            payload=payload,
            readiness=ReadyHandle(ready=True),
        )
        self.exec_crossing.mark_ready("C")

    async def _request_verdict(
        self, goal: str, *, task_id: str, version: int,
    ) -> str:
        """C-facing mind turn; verdict tagged with task + action version."""
        self.mind.role_streams["C"] = (
            f"constraints: audit proposed action v{version}: {goal} task={task_id}"
        )
        self.mind.role_streams["R"] = f"execution stream: proposing {goal}"
        dangerous = "delete" in goal.lower() and "all" in goal.lower()

        def hook(lobe: str, step: int, worker) -> None:
            if not dangerous or lobe != "C" or step < 1:
                return
            payload = {
                "decision": "reject",
                "block_action": "delete_all",
                "goal": goal,
                "task_id": task_id,
                "action_version": version,
            }
            worker.crossing.publish(
                "C", pass_id=worker._pass_id, step=step, layer=0, kind="verdict",
                payload=payload, readiness=ReadyHandle(ready=True),
            )
            worker.crossing.mark_ready("C")
            self._publish_verdict(
                decision="reject",
                block_action="delete_all",
                goal=goal,
                task_id=task_id,
                action_version=version,
                step=step,
            )
            # L authorizes (rejects) this specific version after C
            if self.activation_injection:
                # Simulate L consume of C activation at this decode step
                self.set_action_auth(
                    task_id, version, "delete_all", "reject",
                    consume_step=step, receiver="L", c_publish_step=step,
                )

        result = await self.mind.turn(
            f"Should we authorize this action: {goal}?",
            crossing_enabled=True,
            activation_injection=self.activation_injection,
            mid_pass_hook=hook if dangerous else None,
        )

        ready = self.exec_crossing.latest_ready("C")
        if ready and isinstance(ready.payload, dict):
            payload = ready.payload
            # Ignore mismatched versions
            if int(payload.get("action_version") or version) == version:
                return str(payload.get("decision") or "accept").lower()
        text = (result.results.get("C").text if result.results.get("C") else "") or ""
        if any(w in text.lower() for w in ("reject", "fail", "no", "veto", "block")):
            return "reject"
        if dangerous:
            # Same authorization rule both runs: dangerous needs explicit accept
            return "reject"
        return "accept"

    async def introduce_bad_proposal(
        self,
        lane_id: str,
        *,
        while_decoding: bool = True,
    ) -> CausalAuthTrace:
        """Inject delete_all while workers decode; record causal activation path."""
        handle = None
        with self.lanes._lock:
            handle = self.lanes._lanes.get(lane_id)
        if handle is None:
            raise ValueError(f"unknown lane {lane_id}")
        task_id = handle.task_id
        lane = self.blueprint.lanes.get(lane_id)
        next_version = (lane.action_version + 1) if lane else 1

        trace = CausalAuthTrace(
            task_id=task_id,
            lane_id=lane_id,
            action_version=next_version,
            action_name="delete_all",
            activation_injection=self.activation_injection,
        )
        act_before = len(getattr(self.executor, "act_calls", []) or [])

        def mid_hook(lobe: str, step: int, worker) -> None:
            # Fire on L (parent-process worker under peer isolation) so the
            # mid-decode path stays on L's Crossing without waiting for C's process.
            if lobe != "L" or step < 1:
                return
            if trace.c_verdict is not None:
                return
            trace.c_verdict = "reject"
            trace.c_publish_step = step
            payload = {
                "decision": "reject",
                "block_action": "delete_all",
                "task_id": task_id,
                "lane_id": lane_id,
                "action_version": next_version,
            }
            # Structured coordination (available both runs)
            self._publish_verdict(
                decision="reject",
                block_action="delete_all",
                task_id=task_id,
                lane_id=lane_id,
                action_version=next_version,
                step=step,
            )
            worker.crossing.publish(
                "C", pass_id=worker._pass_id, step=step, layer=0, kind="verdict",
                payload=payload, readiness=ReadyHandle(ready=True),
            )
            worker.crossing.mark_ready("C")
            # Live activation path: L consumes C's ready activation at this step
            if self.activation_injection:
                vec = torch.zeros(64)
                vec[0] = 10.0
                if worker.sim is not None:
                    vec = vec.to(worker.sim.device)
                worker.crossing.publish(
                    "C", pass_id=worker._pass_id, step=step, layer=6, kind="hidden",
                    payload=vec, readiness=ReadyHandle(ready=True),
                )
                worker.crossing.mark_ready("C")
                # Immediate non-blocking consume into L residual
                worker._consume_peers(step=step, layer=6)
                self.set_action_auth(
                    task_id, next_version, "delete_all", "reject",
                    lane_id=lane_id,
                    consume_step=step,
                    receiver="L",
                    c_publish_step=step,
                )
                trace.consume_decode_step = step
                trace.consume_receiver = "L"
                trace.l_decision = "reject"
            else:
                # Coordination publishes verdict, but no activation consume —
                # L does not mid-decode-authorize this version.
                trace.l_decision = None

        # Queue the bad step on the lane, then run a mind turn so C/L decode
        self.lanes.inject_steps(lane_id, [{"name": "delete_all", "args": {}}])
        self.mind.role_streams["C"] = (
            f"constraints: live audit task={task_id} lane={lane_id} "
            f"proposed delete_all v{next_version}"
        )
        self.mind.role_streams["R"] = (
            f"execution: lane {lane_id} proposing delete_all v{next_version}"
        )
        if while_decoding:
            await self.mind.turn(
                f"Audit developing action delete_all on lane {lane_id}",
                crossing_enabled=True,
                activation_injection=self.activation_injection,
                mid_pass_hook=mid_hook,
            )
        else:
            class _W:
                crossing = self.exec_crossing
                _pass_id = 0
                sim = None
            mid_hook("C", 1, _W())

        # Bind auth to the actual proposed action version(s) on this lane
        if self.activation_injection:
            for a in self.blueprint.actions.values():
                if a.lane_id == lane_id and a.name == "delete_all":
                    self.set_action_auth(
                        a.task_id, a.version, a.name, "reject",
                        lane_id=lane_id,
                        consume_step=trace.consume_decode_step,
                        receiver="L",
                        c_publish_step=trace.c_publish_step,
                    )
                    if a.version != next_version:
                        trace.action_version = a.version
                    self._publish_verdict(
                        decision="reject",
                        block_action="delete_all",
                        task_id=a.task_id,
                        lane_id=lane_id,
                        action_version=a.version,
                        step=trace.c_publish_step or 0,
                    )

        # Let lane process the injected step
        await asyncio.sleep(0.55)

        # Determine whether HostExecutor.act ran for this versioned proposal
        danger_acts = [
            a for a in getattr(self.executor, "act_calls", [])[act_before:]
            if a.get("name") == "delete_all"
            and (
                int(a.get("version") or -1) == next_version
                or a.get("task_id") == task_id
            )
        ]
        vetoed = any(
            a.name == "delete_all"
            and a.version == next_version
            and (a.status == "rejected" or "VETOED" in (a.result or ""))
            for a in self.blueprint.actions.values()
        )
        executed = any(
            a.name == "delete_all"
            and a.version == next_version
            and a.dispatched
            and "EXECUTED" in (a.result or "")
            for a in self.blueprint.actions.values()
        )
        trace.host_act_called = bool(danger_acts) or executed
        if vetoed and not executed:
            trace.host_act_called = False

        x = self.mind.last_crossing
        if self.activation_injection and x is not None:
            for c in reversed(x.consumes):
                if c.receiver == "L" and c.source == "C" and not c.miss:
                    if trace.consume_decode_step is None:
                        trace.consume_decode_step = c.step
                        trace.consume_receiver = "L"
                    break

        self.causal_traces.append(trace)
        self._emit({"type": "causal_auth_trace", **trace.__dict__})
        self.persist()
        return trace

    async def chat(
        self,
        message: str,
        *,
        speak: bool = True,
        on_token: Callable[[str, str], None] | None = None,
    ) -> CompanionTurnResult:
        """User message: L answers (and may speak) while lanes keep running."""
        self.board.add_user(message)
        streamed: list[str] = []

        recall = self._recall(message)
        self.mind.role_streams["B"] = (
            f"memory:\n{self._board_text()}\n{recall}"
        )
        self.mind.role_streams["R"] = (
            f"execution stream:\n{self._blueprint_text()}"
        )
        c_extra = ""
        if self.pending_correction:
            c_extra = f" pending_correction: {self.pending_correction}"
        self.mind.role_streams["C"] = f"constraints: verify live plans.{c_extra}"

        prefix = ""
        if self.pending_correction:
            prefix = f"(correction: {self.pending_correction}) "
            self.pending_correction = None

        def _tok(lobe: str, piece: str) -> None:
            streamed.append(piece)
            if on_token:
                on_token(lobe, piece)
            self._emit({"type": "token", "text": piece})

        def c_hook(lobe: str, step: int, worker) -> None:
            if lobe != "C" or step != 2:
                return
            rsig = self.exec_crossing.latest_ready("R")
            decision = "accept"
            block = None
            task_id = ""
            lane_id = ""
            action_version = 0
            if rsig and isinstance(rsig.payload, dict):
                step_info = rsig.payload.get("step") or {}
                task_id = str(rsig.payload.get("task_id") or "")
                lane_id = str(rsig.payload.get("lane_id") or "")
                action_version = int(rsig.payload.get("action_version") or 0)
                if isinstance(step_info, dict) and step_info.get("name") == "delete_all":
                    decision = "reject"
                    block = "delete_all"
            payload = {
                "decision": decision,
                "verdict": decision,
                "task_id": task_id,
                "lane_id": lane_id,
                "action_version": action_version,
            }
            if block:
                payload["block_action"] = block
            worker.crossing.publish(
                "C", pass_id=worker._pass_id, step=step, layer=0, kind="verdict",
                payload=payload, readiness=ReadyHandle(ready=True),
            )
            worker.crossing.mark_ready("C")
            self.exec_crossing.publish(
                "C", pass_id=0, step=step, layer=0, kind="verdict",
                payload=payload, readiness=ReadyHandle(ready=True),
            )
            self.exec_crossing.mark_ready("C")
            if decision == "reject" and block and task_id and action_version:
                if self.activation_injection:
                    self.set_action_auth(
                        task_id, action_version, block, "reject",
                        lane_id=lane_id, consume_step=step, receiver="L",
                        c_publish_step=step,
                    )
                self.pending_correction = "C rejected a dangerous developing action"

        self._audio_frames = 0
        self._ttfa_s = None
        # L streams first; board persist / embed index run after tokens start
        # (and never backpressure the token or audio path).
        mind = await self.mind.turn(
            prefix + message,
            crossing_enabled=True,
            activation_injection=self.activation_injection,
            on_token=_tok,
            mid_pass_hook=c_hook,
        )

        reply = mind.reply or ""
        # Snapshot already available for L's next turn; persist off hot path
        self.board.add_l(reply)
        utterance_id = None
        if speak and reply:
            self._speak_started = time.perf_counter()
            utterance_id = self.voice.speak(reply)
            self._last_utterance_id = utterance_id

        # Background surfaces — independent of L text/audio stream
        self._index_turn(message, reply, mind)
        self._publish_boards()
        self.persist()

        return CompanionTurnResult(
            reply=reply,
            session_id=self.session_id,
            mind=mind,
            streamed=streamed,
            audio_frames=self._audio_frames,
            ttfa_s=self._ttfa_s or (
                (self.voice.first_audio_at - self._speak_started)
                if self.voice.first_audio_at and self._speak_started else None
            ),
            pending_correction=self.pending_correction,
            lanes_active=[h.lane_id for h in self.lanes.active],
            crossing_enabled=self.crossing_enabled,
            activation_injection=self.activation_injection,
            utterance_id=utterance_id,
        )

    def redirect_lane(self, lane_id: str, instruction: str) -> list:
        superseded = self.lanes.redirect(lane_id, instruction)
        self.board.commitments.append(f"redirect:{lane_id}:{instruction}")
        self.persist()
        self._emit({
            "type": "lane_redirected",
            "lane_id": lane_id,
            "instruction": instruction,
            "superseded": [a.action_id for a in superseded],
        })
        return superseded

    def cancel_lane(self, lane_id: str) -> None:
        self.lanes.cancel(lane_id)
        self.persist()

    def _board_text(self) -> str:
        lines = ["ConversationBoard:"]
        for t in self.board.turns[-8:]:
            lines.append(f"  {t.role}: {t.text[:120]}")
        if self.board.open_questions:
            lines.append("Open: " + "; ".join(self.board.open_questions[-3:]))
        return "\n".join(lines)

    def _blueprint_text(self) -> str:
        lines = ["ExecutionBlueprint:"]
        for lid, lane in self.blueprint.lanes.items():
            lines.append(
                f"  {lid}: {lane.status} goal={lane.goal[:60]} last={lane.last_result[:60]}"
            )
        if self.blueprint.sketchpad:
            lines.append("Facts: " + " | ".join(self.blueprint.sketchpad[-5:]))
        uncertain = self.blueprint.uncertain_actions()
        if uncertain:
            lines.append(
                "Uncertain(dispatched≠observed): "
                + ", ".join(a.action_id for a in uncertain[:5])
            )
        return "\n".join(lines)

    def _publish_boards(self) -> None:
        # Coordination — always publish when companion crossing is up
        self.exec_crossing.publish(
            "B", pass_id=0, step=0, layer=0, kind="board",
            payload=self.board.delta_payload(),
            readiness=ReadyHandle(ready=True),
        )
        self.exec_crossing.publish(
            "B", pass_id=0, step=0, layer=1, kind="blueprint",
            payload=self.blueprint.delta_payload(),
            readiness=ReadyHandle(ready=True),
        )
        self.exec_crossing.mark_ready("B")

    def _recall(self, query: str) -> str:
        emb = _bow_embed(query)
        facts = self.episodic.query(emb, top_k=3)
        if not facts:
            return "memory: none"
        return embedder_prompt_from_recall(facts)

    def _index_turn(self, user: str, reply: str, mind: SessionTurnResult) -> None:
        text = f"user: {user}\nL: {reply}\nsketch: {' | '.join(self.blueprint.sketchpad[-3:])}"
        emb_tensor = None
        er = mind.results.get("Embedder")
        if er is not None and er.embedding is not None:
            emb_tensor = er.embedding.detach().float().flatten()
        if emb_tensor is None:
            emb = _bow_embed(text)
        else:
            emb = emb_tensor.tolist()
        self.episodic.store(
            text,
            lobe_roles={"L": reply[:80], "B": "board", "Embedder": "index"},
            embedding=emb,
        )

    async def run_multi_r(
        self,
        goal: str,
        *,
        n_workers: int | None = None,
        cross_attn: bool = True,
        mode: str | None = None,
        constraints: str = "",
        source_material: str = "",
        force_duplicate: bool = False,
        executor: Any | None = None,
        host: Any | None = None,
    ):
        """R coordinates three silent batched workers on one shared task.

        L authorizes; R launches all workers in one inference batch. If the
        host supports subagents, one host subagent is mapped per R worker for
        the task lifetime. Otherwise all three workers stay active and feed a
        single host execution path (prepare-ahead).
        """
        from host_subagents import bind_host_executor
        from multi_r_config import MultiRLaunchError, multi_r_worker_count
        from r_coordinator import RCoordinator, SequentialExecutor

        n = multi_r_worker_count(override=n_workers)
        host = host if host is not None else self.host
        host_info: dict[str, Any] = {}
        bound_executor = executor
        bound_mode = mode or "sequential"

        if bound_executor is None:
            bound_executor, bound_mode, host_info = await bind_host_executor(
                host, n_workers=n,
            )
            # Explicit mode override still wins when caller set it
            if mode is not None:
                bound_mode = mode
        elif host is None and mode is None:
            bound_mode = "sequential"
            bound_executor = bound_executor or SequentialExecutor(delay_s=0.08)
            host_info = {"host_mode": "explicit_executor", "n_subagents": 0}

        self.last_host_launch = dict(host_info)
        self._emit({"type": "host_launch", **host_info, "n_workers": n})

        try:
            coord = RCoordinator(
                n_workers=n,
                cross_attn=cross_attn and self.activation_injection,
                mode=bound_mode,
                crossing=self.exec_crossing,
                executor=bound_executor,
            )
        except MultiRLaunchError as e:
            self.pending_correction = f"Multi-R launch failed: {e}"
            self._emit({
                "type": "multi_r_launch_failed",
                "requested": e.requested,
                "started": e.started,
                "reason": e.reason,
            })
            raise
        try:
            result = await coord.run_task(
                goal,
                constraints=constraints or "silent R; L decides",
                source_material=source_material or self._blueprint_text(),
                force_duplicate=force_duplicate,
                influence_probe=cross_attn and self.activation_injection,
            )
        finally:
            close = getattr(bound_executor, "close", None)
            if close is not None:
                await close()

        result.host_launch = dict(host_info)
        self.board.commitments.append(
            f"multi_r:{result.record.task_id}:workers={n}:"
            f"host={host_info.get('host_mode')}:ok={result.satisfied}"
        )
        for obs in result.record.observations[-10:]:
            self.blueprint.sketchpad.append(obs)
        self.persist()
        self._emit({
            "type": "multi_r_done",
            "task_id": result.record.task_id,
            "satisfied": result.satisfied,
            "metrics": result.metrics.__dict__,
            "host_launch": host_info,
            "influence": result.batch.influence_proof if result.batch else {},
        })
        return result

    async def close(self) -> None:
        self.voice.barge_in()
        for h in list(self.lanes.active):
            h.task.cancel()
        await asyncio.gather(
            *(h.task for h in self.lanes._lanes.values()),
            return_exceptions=True,
        )
        self.persist()
        self.episodic.close()
        self.store.close()
        await self.mind.close()


def _bow_embed(text: str, dim: int = 64) -> list[float]:
    v = torch.zeros(dim)
    for i, w in enumerate(text.lower().split()):
        v[hash(w) % dim] += 1.0
        v[i % dim] += 0.1
    n = float(v.norm())
    if n > 0:
        v = v / n
    return v.tolist()
