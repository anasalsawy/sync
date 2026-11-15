"""Batched R workers with explicit cross-worker attention.

Five (or N) R workers share one inference batch. Each keeps independent
decoding state. At defined layers, every worker attends over live intermediate
activations from every other worker in the same forward. Changing one worker's
activation can change another's behavior within the same run.

All execution workers stay silent — L owns decisions and speech.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from crossing import Crossing, ReadyHandle
from live_screen import LIVE_SCREEN, LiveScreenBus
from local_workers import DEFAULT_VOCAB, HIDDEN, INJECT_SCALE, PUBLISH_LAYERS
from multi_r_config import DEFAULT_MULTI_R_WORKERS, MultiRLaunchError
from task_record import SharedTaskRecord


# Layers where cross-worker attention is applied inside the batched forward
CROSS_ATTN_LAYERS = (3, 5, 7)


@dataclass
class RWorkerState:
    worker_id: str
    index: int
    state: torch.Tensor
    token_ids: list[int] = field(default_factory=list)
    text_pieces: list[str] = field(default_factory=list)
    proposed: list[str] = field(default_factory=list)
    claimed: list[str] = field(default_factory=list)
    prepared: list[str] = field(default_factory=list)


@dataclass
class BatchRTiming:
    compute_started_at: float = 0.0
    finished_at: float = 0.0
    forward_count: int = 0
    decode_steps: int = 0
    cross_attn_ops: int = 0
    step_durations: list[float] = field(default_factory=list)
    cross_attn_influence: bool = False


@dataclass
class BatchRResult:
    worker_results: dict[str, dict[str, Any]]
    timing: BatchRTiming
    cross_attn_enabled: bool
    n_workers: int
    influence_proof: dict[str, Any] = field(default_factory=dict)


class BatchedRSim:
    """N R workers, one batched residual stack + cross-worker attention."""

    def __init__(
        self,
        n_workers: int = DEFAULT_MULTI_R_WORKERS,
        *,
        hidden: int = HIDDEN,
        vocab: dict[str, int] | None = None,
        cross_attn: bool = True,
        cross_attn_scale: float = 0.35,
    ) -> None:
        if n_workers < 1:
            raise MultiRLaunchError(n_workers, 0, "n_workers must be >= 1")
        self.n_workers = n_workers
        self.hidden = hidden
        self.vocab = vocab if vocab is not None else dict(DEFAULT_VOCAB)
        self.inv = {i: t for t, i in self.vocab.items()}
        self.cross_attn = cross_attn
        self.cross_attn_scale = cross_attn_scale
        g = torch.Generator()
        g.manual_seed(0x50B07)
        self.W = torch.randn(hidden, max(64, len(self.vocab) + 32), generator=g)
        self.states: list[torch.Tensor] = [
            torch.zeros(hidden) for _ in range(n_workers)
        ]
        self._influence_probe: torch.Tensor | None = None

    def reset(self, prompt_ids_per_worker: list[list[int]]) -> None:
        self.states = []
        for ids in prompt_ids_per_worker:
            s = torch.zeros(self.hidden)
            for tid in ids:
                s = 0.9 * s
                s[tid % self.hidden] += 1.0
            self.states.append(s)

    def _expand_w(self) -> None:
        if self.W.shape[1] < len(self.vocab):
            extra = len(self.vocab) - self.W.shape[1]
            self.W = torch.cat([self.W, torch.randn(self.hidden, extra)], dim=1)

    def cross_attend(self, layer: int) -> list[torch.Tensor]:
        """Explicit cross-worker attention over live activations at ``layer``."""
        if not self.cross_attn or self.n_workers < 2:
            return [s.clone() for s in self.states]
        # Stack live residuals as K/V; each worker is a query
        stacked = torch.stack(self.states, dim=0)  # [N, H]
        # Layer-conditioned roll so different layers mix differently
        keys = torch.stack(
            [torch.roll(stacked[j], shifts=layer + j, dims=0) for j in range(self.n_workers)],
            dim=0,
        )
        # Attention scores: Q_i · K_j
        scores = stacked @ keys.T / (self.hidden ** 0.5)  # [N, N]
        # Mask self slightly so peers matter, but keep self
        eye = torch.eye(self.n_workers)
        scores = scores - 0.25 * eye
        weights = F.softmax(scores, dim=-1)
        mixed = weights @ stacked  # [N, H]
        out = []
        for i in range(self.n_workers):
            out.append(
                stacked[i] + self.cross_attn_scale * (mixed[i] - stacked[i])
            )
        return out

    def forward_step(self) -> tuple[list[int], list[torch.Tensor]]:
        """One batched decode step with cross-attn at CROSS_ATTN_LAYERS."""
        self._expand_w()
        # Apply cross-attn at each defined layer boundary during this step
        for layer in CROSS_ATTN_LAYERS:
            if self.cross_attn:
                self.states = self.cross_attend(layer)
        tids: list[int] = []
        snaps: list[torch.Tensor] = []
        for i, state in enumerate(self.states):
            w = self.W[:, : len(self.vocab)]
            logits = state @ w
            probs = F.softmax(logits, dim=-1)
            tid = int(torch.argmax(probs).item())
            tids.append(tid)
            snaps.append(state.detach().clone())
            self.states[i] = 0.85 * state
            self.states[i][tid % self.hidden] += 1.0
        return tids, snaps

    def perturb_worker(self, index: int, strength: float = 12.0) -> None:
        """Causal probe: change one worker's activation mid-run."""
        self.states[index] = self.states[index] + strength
        self.states[index][0] = self.states[index][0] + strength


def _tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    toks = []
    for w in text.lower().replace("\n", " ").split():
        w = "".join(ch for ch in w if ch.isalnum() or ch in ("_", "-"))
        if not w:
            continue
        if w not in vocab:
            vocab[w] = len(vocab)
        toks.append(vocab[w])
    return toks or [vocab.get("plan", 4)]


def _task_prompt(
    record: SharedTaskRecord,
    worker_id: str,
    live_screen: LiveScreenBus | None = None,
) -> str:
    snap = record.snapshot_for_worker()
    obs = " | ".join(snap["observations"][-5:]) or "none"
    prog = " | ".join(snap["progress"][-5:]) or "none"
    avail = ", ".join(
        f"{a['effect']}@{a['resource']}" for a in snap["available"][:8]
    ) or "discover"
    bus = live_screen if live_screen is not None else LIVE_SCREEN
    v_ctx = bus.worker_context() if bus.has_live() else "[V live screen: none yet]"
    # When V already shared live state, workers act next — no observe/screenshot.
    skip = (
        "skip_observe: V live screen already shared — do not request screenshot"
        if bus.should_skip_observe()
        else "observe_ok: no live V frame yet"
    )
    return (
        f"R worker {worker_id} silent execution.\n"
        f"goal: {snap['goal']}\n"
        f"constraints: {snap['constraints']}\n"
        f"source: {snap['source_material'][:200]}\n"
        f"{v_ctx}\n"
        f"{skip}\n"
        f"observations: {obs}\n"
        f"progress: {prog}\n"
        f"available_actions: {avail}\n"
        f"propose_or_claim:"
    )


class BatchedRTeam:
    """Coordinates N silent R workers in one batched inference pass."""

    def __init__(
        self,
        n_workers: int = DEFAULT_MULTI_R_WORKERS,
        *,
        max_new_tokens: int = 16,
        cross_attn: bool = True,
        vocab: dict[str, int] | None = None,
        crossing: Crossing | None = None,
        worker_id_prefix: str = "",
        live_screen: LiveScreenBus | None = None,
    ) -> None:
        if n_workers < 1:
            raise MultiRLaunchError(n_workers, 0, "n_workers must be >= 1")
        self.n_workers = n_workers
        self.max_new_tokens = max_new_tokens
        self.cross_attn = cross_attn
        self.vocab = vocab if vocab is not None else dict(DEFAULT_VOCAB)
        self.crossing = crossing
        self.worker_id_prefix = worker_id_prefix
        self.live_screen = live_screen if live_screen is not None else LIVE_SCREEN
        try:
            self.sim = BatchedRSim(
                n_workers, vocab=self.vocab, cross_attn=cross_attn,
            )
            pref = f"{worker_id_prefix}:" if worker_id_prefix else ""
            self.workers = [
                RWorkerState(
                    worker_id=f"{pref}R{i}", index=i, state=torch.zeros(HIDDEN),
                )
                for i in range(n_workers)
            ]
        except Exception as e:
            raise MultiRLaunchError(
                n_workers, 0, f"failed to allocate R batch: {e}",
            ) from e
        if len(self.workers) != n_workers:
            raise MultiRLaunchError(
                n_workers, len(self.workers),
                f"requested {n_workers} R workers, started {len(self.workers)}",
            )

    def ensure_launched(self) -> None:
        """Surface failure if the live batch is short of the requested count."""
        live = len(self.workers)
        if live != self.n_workers:
            raise MultiRLaunchError(self.n_workers, live)
        if len(self.sim.states) != self.n_workers:
            raise MultiRLaunchError(
                self.n_workers, len(self.sim.states),
                "batched residual states missing for some R workers",
            )

    def run_inference(
        self,
        record: SharedTaskRecord,
        *,
        influence_probe: bool = False,
        on_step: Callable[[int, "BatchedRTeam"], None] | None = None,
    ) -> BatchRResult:
        """Batched decode; workers see the whole task; stay silent."""
        self.ensure_launched()
        timing = BatchRTiming(compute_started_at=time.perf_counter())
        prompts = [
            _task_prompt(record, w.worker_id, self.live_screen)
            for w in self.workers
        ]
        prompt_ids = [_tokenize(p, self.vocab) for p in prompts]
        self.sim.reset(prompt_ids)
        if len(self.sim.states) != self.n_workers:
            raise MultiRLaunchError(
                self.n_workers, len(self.sim.states),
                "could not start all R workers in the inference batch",
            )
        for w, s in zip(self.workers, self.sim.states):
            w.state = s
            w.token_ids.clear()
            w.text_pieces.clear()

        influence_proof: dict[str, Any] = {}
        baseline_other: list[int] | None = None

        for step in range(self.max_new_tokens):
            t0 = time.perf_counter()
            if influence_probe and step == 1 and self.cross_attn:
                # Snapshot worker-1 next token tendency before perturbing worker-0
                before = [float(s[0].item()) for s in self.sim.states]
                self.sim.perturb_worker(0, strength=15.0)
                influence_proof["perturbed_worker"] = "R0"
                influence_proof["before_peer_state0"] = before[1] if len(before) > 1 else None
            if on_step is not None:
                on_step(step, self)
            tids, snaps = self.sim.forward_step()
            timing.forward_count += 1
            timing.decode_steps += 1
            if self.cross_attn:
                timing.cross_attn_ops += len(CROSS_ATTN_LAYERS)
            for i, (w, tid, snap) in enumerate(zip(self.workers, tids, snaps)):
                w.state = self.sim.states[i]
                w.token_ids.append(tid)
                piece = self.sim.inv.get(tid, f"t{tid}")
                w.text_pieces.append(piece)
                if self.crossing is not None and self.crossing.enabled:
                    self.crossing.publish(
                        w.worker_id,
                        pass_id=0, step=step, layer=CROSS_ATTN_LAYERS[0],
                        kind="hidden", payload=snap,
                        readiness=ReadyHandle(ready=True),
                    )
                    self.crossing.mark_ready(w.worker_id)
            if influence_probe and step == 1 and self.cross_attn:
                after = float(self.sim.states[1][0].item()) if self.n_workers > 1 else 0.0
                influence_proof["after_peer_state0"] = after
                influence_proof["peer_changed"] = (
                    influence_proof.get("before_peer_state0") is not None
                    and abs(after - float(influence_proof["before_peer_state0"])) > 1e-4
                )
                timing.cross_attn_influence = bool(influence_proof.get("peer_changed"))
            timing.step_durations.append(time.perf_counter() - t0)

        timing.finished_at = time.perf_counter()
        worker_results = {
            w.worker_id: {
                "text": " ".join(w.text_pieces),
                "token_ids": list(w.token_ids),
                "proposed": list(w.proposed),
                "claimed": list(w.claimed),
                "prepared": list(w.prepared),
            }
            for w in self.workers
        }
        return BatchRResult(
            worker_results=worker_results,
            timing=timing,
            cross_attn_enabled=self.cross_attn,
            n_workers=self.n_workers,
            influence_proof=influence_proof,
        )
