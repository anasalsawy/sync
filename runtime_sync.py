"""runtime_sync.py — Thin orchestrator for the NDIF multi-lobe system.

Primary chat path is now ``sync_session.SyncSession`` (five concurrent local
workers + activation Crossing). This module keeps the NDIF batched + offspring
spawn path for research / ``SYNC_RUNTIME=ndif_batch``.

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
from action_runtime import ActionRuntime, EnvironmentState
from sync_session import SyncSession, SessionTurnResult  # noqa: F401 — companion entry


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
    env_state: EnvironmentState | None = None  # parallel action results this turn


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
        self._action_runtime = ActionRuntime()

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

    # Two independently-paced groups. L's claims must be aided by B and
    # verified by C every call, so they stay in one coupled batch together.
    # R's execution depends on V's live state, so they stay in a separate
    # coupled batch. The two batches run fully concurrently — neither's
    # model.generate() loop waits on the other's step count or token budget —
    # and Crossing (via SharedState) carries each group's latest tier1 into
    # the other's trace. This is what makes L's reply pace independent of
    # how long R+V's execution pass runs, and vice versa.
    GROUP_LBC: tuple[str, ...] = ("L", "B", "C")
    GROUP_RV: tuple[str, ...] = ("R", "V")

    def _prompts_for(self, names: tuple[str, ...], user_input: str,
                     offspring_ctx: dict[str, str]) -> list[LobePrompt]:
        """Lobes in one group get the same raw input. Offspring results appended as plain text."""
        text = user_input
        if offspring_ctx:
            # Append without role labels — let the lobes interpret freely
            extra = "\n\n" + "\n\n".join(offspring_ctx.values())
            text = text + extra
        return [LobePrompt(name, text) for name in names]

    # ------------------------------------------------------------------
    # Single turn
    # ------------------------------------------------------------------

    async def turn(self, user_input: str) -> TurnResult:
        self._turn += 1

        offspring_ctx, cancelled = self._harvest_and_veto()

        # Three concurrent things, nothing serialized:
        # - L+B+C: conversation + aid + verification, coupled, own pace.
        # - R+V: execution + live screen, coupled, own pace — can run longer
        #   than L+B+C without holding L's reply up, since it's a separate
        #   model.generate() call, not the same lockstep decode loop.
        # - action round: completed offspring outputs execute their actions
        #   NOW, latency hidden inside whichever group pass runs longest.
        lbc_prompts = self._prompts_for(self.GROUP_LBC, user_input, offspring_ctx)
        rv_prompts = self._prompts_for(self.GROUP_RV, user_input, offspring_ctx)
        lbc_gen, rv_gen, env_state = await asyncio.gather(
            self.channels.generate_group(
                lbc_prompts, self.max_new_tokens, self.shared,
                peer_group_names=self.GROUP_RV, speaker="L",
            ),
            self.channels.generate_group(
                rv_prompts, self.max_new_tokens, self.shared,
                peer_group_names=self.GROUP_LBC, speaker="R",
            ),
            self._action_runtime.run(offspring_ctx),
        )

        # Merge both groups' tier1 — spawn/divergence below reads R (from the
        # R+V group) and C (from the L+B+C group) out of the same dict.
        generation = lbc_gen
        generation.tier1 = {**lbc_gen.tier1, **rv_gen.tier1}

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
            env_state=env_state,
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


# ---------------------------------------------------------------------------
# Preferred entry: local multi-worker activation Crossing (incl. V)
# ---------------------------------------------------------------------------

class LocalSyncRuntime:
    """Session entry point for SYNC_RUNTIME=local_crossing.

    Wraps ``SyncSession`` so callers can import a single runtime module.
    Roster is fixed to L, R, B, V, C, Embedder. V owns live screen state.
    """

    def __init__(self, **kwargs) -> None:
        self.session = SyncSession(**kwargs)

    @property
    def ROSTER(self):
        return self.session.ROSTER

    async def turn(self, message: str, **kwargs) -> SessionTurnResult:
        return await self.session.turn(message, **kwargs)

    async def run(self, messages: list[str]):
        for msg in messages:
            yield await self.session.turn(msg)

    async def close(self) -> None:
        await self.session.close()
