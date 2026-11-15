"""Sync session — five concurrent local workers + live Crossing.

One user request starts L, R, B, C, and Embedder together without waiting for
peer prefill. Only L streams. Crossing carries activations while inference
runs; never blocks. Session state stays alive for additional L-directed passes.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from crossing import Crossing, ReadyHandle
from local_workers import (
    DEFAULT_VOCAB,
    HFBundle,
    LocalLobeWorker,
    MAIN_WORKERS,
    WorkerResult,
    role_prompt,
)
from peer_placement import IsolatedPeerPool, isolate_peers_default


@dataclass
class SessionTurnResult:
    reply: str
    results: dict[str, WorkerResult]
    crossing_telemetry: dict[str, Any]
    started_at: float
    finished_at: float
    config: str  # solo | five_off | five_on
    streamed: list[str] = field(default_factory=list)
    decision_wait_s: float = 0.0
    compute_starts: dict[str, float] = field(default_factory=dict)
    activation_injection: bool = True
    l_path_ok: bool = True
    l_path_violations: list[str] = field(default_factory=list)

    @property
    def ttft_s(self) -> float | None:
        """Time to first token from L compute start (not turn / peer-boot setup).

        Peers must never make L's next token wait — TTFT is measured on L's
        reserved compute clock, after L's own input is ready.
        """
        l = self.results.get("L")
        if not l or l.timing.first_token_at is None:
            return None
        base = l.timing.compute_started_at or self.started_at
        return l.timing.first_token_at - base

    @property
    def ttft_wall_s(self) -> float | None:
        """Wall-clock TTFT from turn start (includes peer boot overlap)."""
        l = self.results.get("L")
        if not l or l.timing.first_token_at is None:
            return None
        return l.timing.first_token_at - self.started_at

    @property
    def total_s(self) -> float:
        return self.finished_at - self.started_at

    def matched_step_ms(self, n: int | None = None) -> list[float]:
        """Per-step durations for L, truncated to matched length ``n``."""
        l = self.results.get("L")
        if not l:
            return []
        steps = list(l.timing.step_durations)
        if n is not None:
            steps = steps[:n]
        return [d * 1000.0 for d in steps]

    def l_inject_profile(self) -> dict[str, Any]:
        l = self.results.get("L")
        if not l:
            return {}
        inj = list(l.timing.inject_s)
        return {
            "inject_ops": l.timing.inject_ops,
            "inject_count": l.timing.inject_count,
            "inject_s_p50": _pct(inj, 0.5) if inj else 0.0,
            "inject_s_p95": _pct(inj, 0.95) if inj else 0.0,
            "peer_wait_count": l.timing.peer_wait_count,
            "skipped_transfer": l.timing.skipped_transfer,
            "forward_count": l.timing.forward_count,
            "decode_steps": l.timing.decode_steps,
        }


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


class SyncSession:
    """Stateful multi-worker session for local activation crossing."""

    ROSTER = MAIN_WORKERS  # L, R, B, C, Embedder

    def __init__(
        self,
        *,
        model_name: str | None = None,
        backend: str | None = None,
        max_new_tokens: int = 32,
        crossing_enabled: bool = True,
        activation_injection: bool = True,
        solo: bool = False,
        isolate_peers: bool | None = None,
        vocab: dict[str, int] | None = None,
        embedder_model: str | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "NDIF_MODEL", "Qwen/Qwen3-8B",
        )
        self.embedder_model = embedder_model or os.environ.get("EMBEDDER_MODEL")
        self.backend = backend or os.environ.get("SYNC_LOCAL_BACKEND", "sim")
        self.max_new_tokens = max_new_tokens
        self.crossing_enabled = crossing_enabled
        self.activation_injection = activation_injection
        self.solo = solo
        self.isolate_peers = (
            isolate_peers_default() if isolate_peers is None else isolate_peers
        )
        self.vocab = vocab if vocab is not None else dict(DEFAULT_VOCAB)
        self.session_id = uuid.uuid4().hex
        self.history: list[str] = []
        self._hf: HFBundle | None = None
        self._pass = 0
        self._alive_workers: dict[str, LocalLobeWorker] = {}
        self.last_crossing: Crossing | None = None
        self.role_streams = {
            "R": "environment: idle",
            "B": "memory: none",
            "C": "constraints: none",
            "Embedder": "",
        }
        if self.backend == "hf":
            self._hf = HFBundle(self.model_name, embedder_model=self.embedder_model)

    def _conversation(self, message: str) -> str:
        lines = self.history[-6:] + [f"user: {message}"]
        return "\n".join(lines)

    def _workers_for(
        self,
        crossing: Crossing,
        on_token: Callable[[str, str], None] | None,
        mid_pass_hook: Callable | None = None,
        is_solo: bool = False,
    ) -> dict[str, LocalLobeWorker]:
        names = ("L",) if is_solo else self.ROSTER
        workers = {}
        for name in names:
            workers[name] = LocalLobeWorker(
                name,
                crossing,
                backend=self.backend,
                hf=self._hf,
                vocab=self.vocab,
                max_new_tokens=self.max_new_tokens,
                speak=(name == "L"),
                peers=names,
                on_token=on_token if name == "L" else None,
                mid_pass_hook=mid_pass_hook,
            )
        self._alive_workers = workers
        return workers

    async def turn(
        self,
        message: str,
        *,
        crossing_enabled: bool | None = None,
        activation_injection: bool | None = None,
        solo: bool | None = None,
        role_streams: dict[str, str] | None = None,
        need_c_before_commit: bool = False,
        on_token: Callable[[str, str], None] | None = None,
        mid_pass_hook: Callable | None = None,
        isolate_peers: bool | None = None,
    ) -> SessionTurnResult:
        enabled = self.crossing_enabled if crossing_enabled is None else crossing_enabled
        inject = (
            self.activation_injection
            if activation_injection is None
            else activation_injection
        )
        is_solo = self.solo if solo is None else solo
        do_isolate = self.isolate_peers if isolate_peers is None else isolate_peers
        if role_streams:
            self.role_streams.update(role_streams)

        # Peers always get a Crossing when not solo. "five_off" disables only
        # activation injection — coordination kinds still publish.
        if is_solo:
            crossing = Crossing(
                enabled=False, activation_injection=False, session_id=self.session_id,
            )
        else:
            inj = bool(inject and enabled)
            crossing = Crossing(
                enabled=True,
                activation_injection=inj,
                session_id=self.session_id,
            )
        self.last_crossing = crossing
        streamed: list[str] = []

        def _stream(lobe: str, piece: str) -> None:
            streamed.append(piece)
            if on_token:
                on_token(lobe, piece)

        conversation = self._conversation(message)
        self._pass += 1
        pass_id = self._pass
        started = time.perf_counter()
        decision_wait = 0.0
        results: dict[str, WorkerResult] = {}
        compute_starts: dict[str, float] = {}

        # --- Isolated peer placement: L reserved in-parent; peers in children ---
        if (not is_solo) and do_isolate and self.backend == "sim":
            peer_names = [n for n in self.ROSTER if n != "L"]
            peer_prompts = {
                name: role_prompt(name, conversation, self.role_streams.get(name, ""))
                for name in peer_names
            }
            l_prompt = role_prompt("L", conversation, "")
            pool = IsolatedPeerPool(
                crossing,
                max_new_tokens=self.max_new_tokens,
                vocab=self.vocab,
                activation_injection=crossing.activation_injection,
            )
            l_worker = LocalLobeWorker(
                "L", crossing, backend="sim", vocab=self.vocab,
                max_new_tokens=self.max_new_tokens, speak=True,
                peers=self.ROSTER, on_token=_stream, mid_pass_hook=mid_pass_hook,
            )
            self._alive_workers = {"L": l_worker}

            # L begins prefill as soon as its own input is ready — never wait
            # for peer process spawn. Peers start concurrently in the background.
            import threading
            peer_started = threading.Event()

            def _start_peers() -> None:
                pool.start_peers(peer_prompts, pass_id=pass_id)
                peer_started.set()

            peer_boot = threading.Thread(target=_start_peers, daemon=True, name="peer-boot")
            peer_boot.start()
            # Tiny yield so peer boot begins overlapping L's first tokens
            await asyncio.sleep(0)
            l_result = await asyncio.to_thread(l_worker.run_sync, l_prompt, pass_id=pass_id)
            results["L"] = l_result
            compute_starts["L"] = l_result.timing.compute_started_at
            peer_boot.join(timeout=30.0)
            peer_results = await asyncio.to_thread(pool.collect)
            results.update(peer_results)
            for name, res in peer_results.items():
                compute_starts[name] = res.timing.compute_started_at
        else:
            workers = self._workers_for(crossing, _stream, mid_pass_hook, is_solo=is_solo)
            prompts = {
                name: role_prompt(name, conversation, self.role_streams.get(name, ""))
                for name in workers
            }
            secret = self.role_streams.get("_b_secret_token")
            if secret and "B" in workers and workers["B"].sim is not None:
                vocab = workers["B"].vocab
                if secret not in vocab:
                    vocab[secret] = len(vocab)
                workers["B"].sim.set_fact_token(vocab[secret], strength=12.0)
                prompts["B"] = prompts["B"] + f"\nsecret_fact: {secret}"

            # Launch ALL workers together — L begins as soon as its input is ready.
            tasks = {
                name: asyncio.create_task(
                    worker.run(prompts[name], pass_id=pass_id),
                    name=f"worker-{name}",
                )
                for name, worker in workers.items()
            }
            results_list = await asyncio.gather(*tasks.values(), return_exceptions=True)
            if need_c_before_commit and "C" in workers:
                for name, res in zip(tasks.keys(), results_list):
                    if name == "C" and not isinstance(res, Exception):
                        if res.timing.finished_at and res.timing.compute_started_at:
                            decision_wait = (
                                res.timing.finished_at - res.timing.compute_started_at
                            )
                        break
            for name, res in zip(tasks.keys(), results_list):
                if isinstance(res, Exception):
                    results[name] = WorkerResult(
                        lobe=name, text=f"[error: {res}]", token_ids=[],
                        timing=workers[name].timing,
                    )
                else:
                    results[name] = res
                    compute_starts[name] = res.timing.compute_started_at

        finished = time.perf_counter()
        reply = (results.get("L").text if results.get("L") else "") or ""

        self.history.append(f"user: {message}")
        self.history.append(f"assistant: {reply}")
        if len(self.history) > 40:
            self.history = self.history[-40:]

        inj_on = crossing.activation_injection
        if is_solo:
            config = "solo"
        elif inj_on:
            config = "five_on"
        else:
            config = "five_off"

        violations: list[str] = []
        lres = results.get("L")
        if lres is not None:
            if lres.timing.peer_wait_count:
                violations.append(f"peer_wait_count={lres.timing.peer_wait_count}")
            if lres.timing.forward_count > lres.timing.decode_steps:
                lres.timing.extra_forward_count = (
                    lres.timing.forward_count - lres.timing.decode_steps
                )
                violations.append(
                    f"extra_forwards={lres.timing.extra_forward_count}"
                )

        return SessionTurnResult(
            reply=reply,
            results=results,
            crossing_telemetry=crossing.telemetry(),
            started_at=started,
            finished_at=finished,
            config=config,
            streamed=streamed,
            decision_wait_s=decision_wait,
            compute_starts=compute_starts,
            activation_injection=inj_on,
            l_path_ok=not violations,
            l_path_violations=violations,
        )

    async def additional_pass(
        self,
        message: str,
        *,
        on_token: Callable[[str, str], None] | None = None,
    ) -> SessionTurnResult:
        """L-directed extra pass — peers stay in session roster."""
        return await self.turn(message, on_token=on_token)

    async def close(self) -> None:
        self._alive_workers.clear()
        self._hf = None


def mid_pass_publish(
    *,
    when_lobe: str,
    after_step: int,
    publish_as: str,
    kind: str = "hidden",
    token_name: str | None = None,
    decision: str | None = None,
) -> Callable:
    """Causal-test helper: publish onto the worker's Crossing after compute began."""

    fired = {"done": False}

    def hook(lobe: str, step: int, worker: LocalLobeWorker) -> None:
        if fired["done"] or lobe != when_lobe or step < after_step:
            return
        x = worker.crossing
        vocab = worker.vocab
        hidden = worker.sim.hidden if worker.sim is not None else 64
        if kind == "decision":
            tid = vocab.get(token_name or "revise", 34)
            x.publish(
                publish_as, pass_id=worker._pass_id, step=step, layer=0, kind="decision",
                payload={"decision": decision or "objective", "token": tid},
                readiness=ReadyHandle(ready=True),
            )
        else:
            name = token_name or "gamma"
            if name not in vocab:
                vocab[name] = len(vocab)
            vec = torch_zeros_fact(vocab[name], hidden)
            x.publish(
                publish_as, pass_id=worker._pass_id, step=step, layer=2, kind="hidden",
                payload=vec, readiness=ReadyHandle(ready=True),
            )
        x.mark_ready(publish_as)
        fired["done"] = True

    return hook


def torch_zeros_fact(token_id: int, hidden: int):
    import torch
    vec = torch.zeros(hidden)
    vec[token_id % hidden] = 10.0
    return vec


async def run_chat_turn(
    message: str,
    *,
    max_new_tokens: int = 32,
    history: list[str] | None = None,
    role_streams: dict[str, str] | None = None,
    on_token: Callable[[str, str], None] | None = None,
) -> SessionTurnResult:
    session = SyncSession(max_new_tokens=max_new_tokens)
    if history:
        session.history = list(history)
    return await session.turn(message, role_streams=role_streams, on_token=on_token)
