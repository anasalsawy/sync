"""Peer compute placement — keep L's token path off shared-host contention.

When all five workers share one Python process / GPU, L's next token slows
even with activation injection disabled. Host thread priority is not enough.
This module runs peer lobes in child processes so L keeps reserved compute
in the parent. A relay thread merges peer publishes into the in-process
Crossing without ever blocking L.

L begins prefill as soon as its own input is ready; peers start concurrently.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import asdict
from typing import Any

import torch

from crossing import Crossing, ReadyHandle
from local_workers import (
    DEFAULT_VOCAB,
    LocalLobeWorker,
    WorkerResult,
    WorkerTiming,
)


def _peer_entry(
    lobe: str,
    prompt: str,
    pass_id: int,
    max_new_tokens: int,
    vocab: dict,
    activation_injection: bool,
    out_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    """Child process: run one peer; ship publishes + final result to parent."""
    crossing = Crossing(
        enabled=True,
        activation_injection=activation_injection,
        session_id=f"peer-{lobe}",
    )
    # Wrap publish to also enqueue serializable payloads for the parent Crossing
    _orig_publish = crossing.publish

    def _publish(source, **kwargs):
        sig = _orig_publish(source, **kwargs)
        if sig is None:
            return None
        payload = sig.payload
        ser_payload: Any
        if isinstance(payload, torch.Tensor):
            ser_payload = {"_tensor": payload.detach().cpu().tolist(),
                           "_shape": list(payload.shape)}
        else:
            ser_payload = payload
        try:
            out_q.put({
                "source": source,
                "pass_id": sig.pass_id,
                "step": sig.step,
                "layer": sig.layer,
                "kind": sig.kind,
                "payload": ser_payload,
            }, timeout=0.01)
        except Exception:
            pass
        return sig

    crossing.publish = _publish  # type: ignore[method-assign]

    worker = LocalLobeWorker(
        lobe,
        crossing,
        backend="sim",
        vocab=dict(vocab),
        max_new_tokens=max_new_tokens,
        speak=False,
        peers=(lobe,),  # peer-local; parent Crossing carries cross-lobe share
    )
    result = worker.run_sync(prompt, pass_id=pass_id)
    result_q.put({
        "lobe": result.lobe,
        "text": result.text,
        "token_ids": list(result.token_ids),
        "timing": {
            "lobe": result.timing.lobe,
            "task_started_at": result.timing.task_started_at,
            "compute_started_at": result.timing.compute_started_at,
            "first_token_at": result.timing.first_token_at,
            "finished_at": result.timing.finished_at,
            "forward_count": result.timing.forward_count,
            "decode_steps": result.timing.decode_steps,
            "step_durations": list(result.timing.step_durations),
            "inject_count": result.timing.inject_count,
            "miss_count": result.timing.miss_count,
            "inject_s": list(result.timing.inject_s),
            "inject_ops": result.timing.inject_ops,
            "peer_wait_count": result.timing.peer_wait_count,
            "skipped_transfer": result.timing.skipped_transfer,
            "extra_forward_count": result.timing.extra_forward_count,
        },
        "decisions": list(result.decisions),
    })


class IsolatedPeerPool:
    """Start peers in child processes; relay activations into parent Crossing."""

    def __init__(
        self,
        crossing: Crossing,
        *,
        max_new_tokens: int,
        vocab: dict | None = None,
        activation_injection: bool = True,
    ) -> None:
        self.crossing = crossing
        self.max_new_tokens = max_new_tokens
        self.vocab = vocab or dict(DEFAULT_VOCAB)
        self.activation_injection = activation_injection
        self._ctx = mp.get_context("spawn")
        self._out_q: mp.Queue = self._ctx.Queue()
        self._result_q: mp.Queue = self._ctx.Queue()
        self._procs: list[mp.Process] = []
        self._relay_stop = threading.Event()
        self._relay: threading.Thread | None = None

    def start_peers(
        self,
        prompts: dict[str, str],
        *,
        pass_id: int,
    ) -> None:
        self._relay_stop.clear()
        # Relay only needed when activation injection is on — otherwise peers
        # burn compute in child processes without touching L's Crossing.
        if self.activation_injection:
            self._relay = threading.Thread(
                target=self._relay_loop, daemon=True, name="peer-relay",
            )
            self._relay.start()
        for lobe, prompt in prompts.items():
            p = self._ctx.Process(
                target=_peer_entry,
                args=(
                    lobe, prompt, pass_id, self.max_new_tokens,
                    self.vocab, self.activation_injection,
                    self._out_q, self._result_q,
                ),
                name=f"peer-{lobe}",
            )
            p.start()
            self._procs.append(p)

    def _relay_loop(self) -> None:
        while not self._relay_stop.is_set():
            try:
                item = self._out_q.get(timeout=0.02)
            except queue.Empty:
                continue
            except Exception:
                continue
            payload = item.get("payload")
            if isinstance(payload, dict) and "_tensor" in payload:
                payload = torch.tensor(payload["_tensor"]).reshape(payload.get("_shape") or [-1])
            self.crossing.publish(
                item["source"],
                pass_id=item["pass_id"],
                step=item["step"],
                layer=item["layer"],
                kind=item["kind"],
                payload=payload,
                readiness=ReadyHandle(ready=True),
            )
            self.crossing.mark_ready(item["source"])

    def collect(self, timeout: float = 60.0) -> dict[str, WorkerResult]:
        results: dict[str, WorkerResult] = {}
        deadline = time.perf_counter() + timeout
        pending = {p.name.replace("peer-", "") for p in self._procs}
        while pending and time.perf_counter() < deadline:
            try:
                raw = self._result_q.get(timeout=0.05)
            except Exception:
                continue
            lobe = raw["lobe"]
            timing = WorkerTiming(**raw["timing"])
            results[lobe] = WorkerResult(
                lobe=lobe,
                text=raw["text"],
                token_ids=raw["token_ids"],
                timing=timing,
                decisions=raw.get("decisions") or [],
            )
            pending.discard(lobe)
        for p in self._procs:
            p.join(timeout=0.5)
            if p.is_alive():
                p.terminate()
        self._relay_stop.set()
        if self._relay is not None:
            self._relay.join(timeout=1.0)
        return results


def isolate_peers_default() -> bool:
    """Peer isolation is opt-in via SYNC_ISOLATE_PEERS or Companion/latency gate.

    Unit causal tests keep shared-process peers so mid_pass hooks apply.
    Production companion and eval_l_latency enable isolation so L keeps
    reserved compute when five workers would otherwise share one host GIL/GPU.
    """
    env = os.environ.get("SYNC_ISOLATE_PEERS", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    return False
