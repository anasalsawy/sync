"""In-process activation Crossing — zero inference, non-blocking share.

Workers publish activation tensors at layer/decode hooks with an explicit
GPU-readiness handle. Receivers take the newest *ready* peer tensor without
waiting. A Python lock protects routing metadata only; CUDA events (or an
immediate-ready flag on CPU/sim) gate tensor use.
"""
from __future__ import annotations

import threading
time_module = __import__("time")
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReadyHandle:
    """GPU (or sim) readiness for a published activation.

    On CUDA, ``event`` is a recorded ``torch.cuda.Event``. On CPU/sim,
    ``ready`` is set True immediately after the producing work finished.
    """
    ready: bool = False
    event: Any = None  # torch.cuda.Event | None

    def mark_ready(self, event: Any = None) -> None:
        self.event = event
        self.ready = True if event is None else False
        if event is not None:
            # Event already recorded by publisher; query() decides readiness.
            self.ready = False

    def is_ready(self) -> bool:
        if self.event is None:
            return self.ready
        try:
            return bool(self.event.query())
        except Exception:
            return self.ready


@dataclass
class Signal:
    session_id: str
    source: str
    pass_id: int
    step: int
    layer: int
    kind: str
    payload: Any
    produced_at: float = field(default_factory=time_module.perf_counter)
    seq: int = 0
    dtype: str = "float32"
    device: str = "cpu"
    shape: tuple[int, ...] = ()
    readiness: ReadyHandle = field(default_factory=ReadyHandle)


@dataclass
class ConsumeRecord:
    receiver: str
    source: str
    signal_seq: int
    step: int
    layer: int
    consumed_at: float
    miss: bool = False
    reason: str = ""


@dataclass
class MissRecord:
    receiver: str
    source: str
    step: int
    layer: int
    at: float
    reason: str = "absent"


# Tensor residual inject kinds. Structured coordination kinds always publish
# when ``enabled`` — "Crossing off" for eval means ``activation_injection=False``.
ACTIVATION_KINDS = frozenset({"hidden", "embed"})
COORDINATION_KINDS = frozenset({
    "tool_result", "board", "blueprint", "task", "verdict", "decision", "index",
})


class Crossing:
    """Non-blocking pub/sub for live activation signals.

    ``publish`` never waits on peers. ``latest_ready`` returns the newest
    signal whose producing GPU work has completed, else the previous ready
    value, else None (caller records a miss and skips injection).

    ``activation_injection`` gates residual inject of hidden/embed tensors.
    Tool results, boards, task coordination, and verdicts keep flowing when
    injection is off so comparisons isolate activation sharing.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        activation_injection: bool = True,
        session_id: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.activation_injection = activation_injection
        self.session_id = session_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._latest: dict[str, Signal] = {}
        self._latest_ready: dict[str, Signal] = {}
        self._seq = 0
        self.publishes: list[Signal] = []
        self.consumes: list[ConsumeRecord] = []
        self.misses: list[MissRecord] = []

    def allows_activation_inject(self, kind: str) -> bool:
        if not self.enabled:
            return False
        if kind in ACTIVATION_KINDS:
            return self.activation_injection
        return True

    def publish(
        self,
        source: str,
        *,
        pass_id: int,
        step: int,
        layer: int,
        kind: str,
        payload: Any,
        readiness: ReadyHandle | None = None,
        dtype: str | None = None,
        device: str | None = None,
    ) -> Signal | None:
        if not self.enabled:
            return None
        # Activation tensors are not published when injection is disabled —
        # coordination kinds still publish under the same authorization rules.
        if kind in ACTIVATION_KINDS and not self.activation_injection:
            return None
        now = time_module.perf_counter()
        shape: tuple[int, ...] = ()
        pay_dtype = dtype or "float32"
        pay_device = device or "cpu"
        if hasattr(payload, "shape"):
            try:
                shape = tuple(int(x) for x in payload.shape)
            except Exception:
                shape = ()
        if hasattr(payload, "dtype") and dtype is None:
            pay_dtype = str(payload.dtype).replace("torch.", "")
        if hasattr(payload, "device") and device is None:
            pay_device = str(payload.device)

        handle = readiness or ReadyHandle(ready=True)
        with self._lock:
            self._seq += 1
            sig = Signal(
                session_id=self.session_id,
                source=source,
                pass_id=pass_id,
                step=step,
                layer=layer,
                kind=kind,
                payload=payload,
                produced_at=now,
                seq=self._seq,
                dtype=pay_dtype,
                device=pay_device,
                shape=shape,
                readiness=handle,
            )
            self._latest[source] = sig
            if handle.is_ready():
                self._latest_ready[source] = sig
            self.publishes.append(sig)
            return sig

    def mark_ready(self, source: str, seq: int | None = None) -> None:
        """Publisher calls this after GPU work for the signal has finished."""
        with self._lock:
            sig = self._latest.get(source)
            if sig is None:
                return
            if seq is not None and sig.seq != seq:
                return
            if sig.readiness.event is not None:
                # Leave query() to decide; still promote if already done.
                if sig.readiness.is_ready():
                    self._latest_ready[source] = sig
                return
            sig.readiness.ready = True
            self._latest_ready[source] = sig

    def latest(self, sources: list[str] | tuple[str, ...] | None = None) -> dict[str, Signal]:
        with self._lock:
            if sources is None:
                return dict(self._latest)
            return {s: self._latest[s] for s in sources if s in self._latest}

    def latest_one(self, source: str) -> Signal | None:
        with self._lock:
            return self._latest.get(source)

    def latest_ready(self, source: str) -> Signal | None:
        """Newest compatible tensor whose producing GPU work has completed."""
        with self._lock:
            cur = self._latest.get(source)
            if cur is not None and cur.readiness.is_ready():
                self._latest_ready[source] = cur
                return cur
            return self._latest_ready.get(source)

    def record_consume(
        self,
        receiver: str,
        source: str,
        *,
        step: int,
        layer: int,
        signal: Signal | None,
        reason: str = "",
    ) -> None:
        now = time_module.perf_counter()
        with self._lock:
            if signal is None:
                why = reason or "absent"
                self.misses.append(MissRecord(
                    receiver=receiver, source=source, step=step, layer=layer,
                    at=now, reason=why,
                ))
                self.consumes.append(ConsumeRecord(
                    receiver=receiver, source=source, signal_seq=-1,
                    step=step, layer=layer, consumed_at=now, miss=True,
                    reason=why,
                ))
                return
            self.consumes.append(ConsumeRecord(
                receiver=receiver, source=source, signal_seq=signal.seq,
                step=step, layer=layer, consumed_at=now, miss=False,
                reason=reason,
            ))

    def miss_count(self) -> int:
        with self._lock:
            return len(self.misses)

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            latencies = []
            for c in self.consumes:
                if c.miss:
                    continue
                for p in self.publishes:
                    if p.seq == c.signal_seq:
                        latencies.append(c.consumed_at - p.produced_at)
                        break
            return {
                "session_id": self.session_id,
                "enabled": self.enabled,
                "activation_injection": self.activation_injection,
                "publish_count": len(self.publishes),
                "consume_count": len(self.consumes),
                "miss_count": len(self.misses),
                "sources": sorted(self._latest.keys()),
                "ready_sources": sorted(self._latest_ready.keys()),
                "latency_s": latencies,
                "miss_reasons": {
                    r.reason: sum(1 for m in self.misses if m.reason == r.reason)
                    for r in self.misses
                } if self.misses else {},
            }
