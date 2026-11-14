"""runtime_sync.py — Thin orchestrator for the NDIF multi-lobe system.

Design principles:
  1. No role-play, no stage directions, no system prompts.
     Every lobe receives the raw user input identically. Character and routing
     emerge from the asymmetric wiring + norm-gated channels, not from prompts.

  2. Spawn signal comes from channel divergence, not L's entropy.
     When R and C activate different expert paths on the same input, that is
     genuine disagreement — spawn parallel exploration. L's entropy is a
     side-effect; R↔C reliance divergence is the cause.

  3. Offspring state is live for all lobes, not just V.
     SharedState is updated incrementally. Non-V lobes get a new snapshot
     pushed after each pass so the next offspring inject sees fresh state,
     not a frozen pre-pass snapshot.

  4. MoE target: Qwen3-30B-A3B (30B params, ~3B active/token).
     Two levels of emergent routing stack: expert routing within each lobe,
     channel routing between lobes. They compound — specialised lobes emit
     more precise channel signals, activating more targeted expert subsets
     in receiving lobes.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ndif import (
    NDIFChannels,
    NDIFOffspring,
    LobePrompt,
    SharedState,
    OffspringHandle,
    Generation,
)


# ---------------------------------------------------------------------------
# Spawn policy — driven by R↔C channel divergence
# ---------------------------------------------------------------------------

# Cosine distance between R and C's tier1 at the mid-stack snapshot layer
# that triggers spawning a parallel exploration lane.
DIVERGENCE_SPAWN_THRESHOLD = 0.25   # tunable; 0 = identical, 1 = orthogonal

# Each additional DIVERGENCE_STEP above threshold spawns one more lane.
DIVERGENCE_STEP = 0.10

MAX_OFFSPRING = 10

# The snapshot layer used to measure R↔C divergence (must be in _SNAP_LAYERS).
DIVERGENCE_SNAP_LAYER = 18


def _cosine_distance(a, b) -> float:
    """1 - cosine_similarity, in [0, 2]. Returns 1.0 on zero vectors."""
    try:
        import torch
        a = a.float().flatten()
        b = b.float().flatten()
        denom = (a.norm() * b.norm()).item()
        if denom < 1e-8:
            return 1.0
        return float(1.0 - (a @ b).item() / denom)
    except Exception:
        return 0.0


def _spawn_count(tier1: dict) -> int:
    """Decide how many lanes to spawn from R↔C divergence in tier1 snapshots."""
    r_snap = tier1.get("R", {}).get(DIVERGENCE_SNAP_LAYER)
    c_snap = tier1.get("C", {}).get(DIVERGENCE_SNAP_LAYER)
    if r_snap is None or c_snap is None:
        return 0
    # Mean across slots → representative vector for each lobe at this layer
    r_vec = r_snap.float().mean(dim=0)
    c_vec = c_snap.float().mean(dim=0)
    dist = _cosine_distance(r_vec, c_vec)
    if dist <= DIVERGENCE_SPAWN_THRESHOLD:
        return 0
    excess = dist - DIVERGENCE_SPAWN_THRESHOLD
    return min(1 + int(excess / DIVERGENCE_STEP), MAX_OFFSPRING)


# ---------------------------------------------------------------------------
# TurnResult
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    text: str                        # L's spoken reply
    generation: Generation           # full main-pass result
    offspring_texts: dict[str, str]  # completed offspring outputs this turn
    spawned: list[str]               # lobe names spawned this turn
    cancelled: list[str]             # lobe names cancelled (C veto / error)
    rc_divergence: float             # R↔C cosine distance that drove spawning


# ---------------------------------------------------------------------------
# SyncRuntime
# ---------------------------------------------------------------------------

class SyncRuntime:
    """Stateful async runtime — thin wrapper, emergent routing does the work.

    The only job of this class is:
      - feed raw input to all 5 lobes equally
      - measure R↔C divergence after each pass
      - spawn/kill offspring lanes accordingly
      - harvest completed offspring and fold their text into the next pass
        as plain additional input (not role-tagged, not stage-directed)
    """

    # MoE upgrade path: 235B-A22B has 22B active/token (3× the current 7B dense)
    # Stay on Qwen3-8B until you can test on 235B-A22B — do not use 30B-A3B,
    # its 3B active/token is a regression from 7B dense.
    DEFAULT_MODEL = "Qwen/Qwen3-8B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 64,
        max_offspring: int = MAX_OFFSPRING,
    ) -> None:
        self.channels = NDIFChannels(model_name=model_name)
        self.offspring_runner = NDIFOffspring(model_name=model_name)
        self.shared = SharedState()
        self.max_new_tokens = max_new_tokens
        self.max_offspring = max_offspring
        self._handles: list[OffspringHandle] = []
        self._turn: int = 0

    # ------------------------------------------------------------------
    # Harvest + C-veto
    # ------------------------------------------------------------------

    def _harvest_and_veto(self) -> tuple[dict[str, str], list[str]]:
        """Collect done offspring; cancel errored ones. Returns (results, cancelled)."""
        results: dict[str, str] = {}
        cancelled: list[str] = []
        still_active: list[OffspringHandle] = []
        for h in self._handles:
            if h.task.done():
                if h.task.cancelled() or h.task.exception() is not None:
                    cancelled.append(h.lobe)
                else:
                    results[h.lobe] = h.task.result().text
            else:
                still_active.append(h)
        self._handles = still_active
        return results, cancelled

    # ------------------------------------------------------------------
    # Build prompts — raw input, no role labels
    # ------------------------------------------------------------------

    def _prompts(self, user_input: str, offspring_ctx: dict[str, str]) -> list[LobePrompt]:
        """All lobes get the same raw input. Offspring results appended as plain text."""
        text = user_input
        if offspring_ctx:
            # Append without role labels — let the lobes interpret freely
            extra = "\n\n" + "\n\n".join(offspring_ctx.values())
            text = text + extra
        return [
            LobePrompt("L", text),
            LobePrompt("R", text),
            LobePrompt("B", text),
            LobePrompt("V", text),
            LobePrompt("C", text),
        ]

    # ------------------------------------------------------------------
    # Single turn
    # ------------------------------------------------------------------

    async def turn(self, user_input: str) -> TurnResult:
        self._turn += 1

        offspring_ctx, cancelled = self._harvest_and_veto()

        generation = await self.channels.generate(
            self._prompts(user_input, offspring_ctx),
            max_new_tokens=self.max_new_tokens,
            shared=self.shared,
        )

        # Spawn based on R↔C divergence
        n = _spawn_count(generation.tier1)
        spawned: list[str] = []
        for i in range(n):
            lane = f"R{len(self._handles) + i + 1}"
            handle = self.offspring_runner.spawn(
                LobePrompt(lane, user_input),   # same raw input, no framing
                self.shared,
                main_lobe_names=list(generation.tier1.keys()),
                max_new_tokens=self.max_new_tokens,
            )
            self._handles.append(handle)
            spawned.append(lane)

        rc_dist = _cosine_distance(
            generation.tier1.get("R", {}).get(DIVERGENCE_SNAP_LAYER,
                generation.tier1.get("R", {}).get(next(iter(generation.tier1.get("R", {1: None})), 1))),
            generation.tier1.get("C", {}).get(DIVERGENCE_SNAP_LAYER,
                generation.tier1.get("C", {}).get(next(iter(generation.tier1.get("C", {1: None})), 1))),
        ) if generation.tier1.get("R") and generation.tier1.get("C") else 0.0

        return TurnResult(
            text=generation.text,
            generation=generation,
            offspring_texts=offspring_ctx,
            spawned=spawned,
            cancelled=cancelled,
            rc_divergence=rc_dist,
        )

    # ------------------------------------------------------------------
    # Session loop
    # ------------------------------------------------------------------

    async def session(self, messages: list[str]):
        for msg in messages:
            yield await self.turn(msg)
        for h in self._handles:
            h.task.cancel()
        await asyncio.gather(*(h.task for h in self._handles), return_exceptions=True)
        self._handles.clear()
