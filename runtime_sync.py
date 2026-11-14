"""runtime_sync.py — Spawn-aware orchestrator for the NDIF multi-lobe system.

Ties NDIFChannels (main 5-lobe pass) + NDIFOffspring (concurrent second-tier
lanes) + SharedState (bidirectional live feed) into one async loop.

The main mind (L, R, B, V, C) runs every turn. After each main pass:
  1. L's output entropy is checked. High entropy → ambiguous, spawn more lanes.
  2. R's demand flags are checked. demand.split → spawn parallel exploration.
  3. Active offspring results are harvested and injected as extra context for L.
  4. C can veto in-flight offspring tasks before the next main pass.

Usage — single-turn helper:
    from runtime_sync import SyncRuntime
    rt = SyncRuntime()
    reply = await rt.turn(user_message)
    print(reply)

Usage — long-running session:
    rt = SyncRuntime()
    async for reply in rt.session(messages):
        print(reply)
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import AsyncIterator

from ndif import (
    NDIFChannels,
    NDIFOffspring,
    LobePrompt,
    SharedState,
    OffspringHandle,
    Generation,
    ENTROPY_DEMAND_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Spawn policy
# ---------------------------------------------------------------------------

# Maximum concurrent offspring lanes at any one time.
MAX_OFFSPRING = 10

# How many tokens of entropy headroom before we spawn another lane.
# (entropy > ENTROPY_DEMAND_THRESHOLD already triggers one; each additional
#  ENTROPY_STEP above that threshold spawns one more, up to MAX_OFFSPRING)
ENTROPY_STEP = 0.5

# Lobe name prefix for auto-spawned exploration lanes.
EXPLORE_PREFIX = "R"


@dataclass
class TurnResult:
    """Everything the caller needs after one full turn."""
    text: str                        # L's spoken reply
    generation: Generation           # full main-pass result
    offspring_texts: dict[str, str]  # {lobe: text} for all completed lanes
    spawned: list[str]               # lobe names spawned this turn
    cancelled: list[str]             # lobe names cancelled (C veto)


# ---------------------------------------------------------------------------
# SyncRuntime
# ---------------------------------------------------------------------------

class SyncRuntime:
    """Stateful async runtime for the 5+N lobe system.

    Manages:
      - One NDIFChannels instance (main 5-lobe mind)
      - One NDIFOffspring instance (shared; re-used across turns)
      - One SharedState hub (lives for the whole session)
      - Active offspring handle table (killed or awaited each turn)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.8-27B",
        max_new_tokens: int = 64,
        max_offspring: int = MAX_OFFSPRING,
    ) -> None:
        self.channels = NDIFChannels(model_name=model_name)
        self.offspring_runner = NDIFOffspring(model_name=model_name)
        self.shared = SharedState()
        self.max_new_tokens = max_new_tokens
        self.max_offspring = max_offspring

        # Active handles from the previous turn (may still be running)
        self._handles: list[OffspringHandle] = []
        self._turn_counter: int = 0

    # ------------------------------------------------------------------
    # Spawn-count policy
    # ------------------------------------------------------------------

    def _how_many_to_spawn(self, generation: Generation) -> int:
        """Decide how many new offspring lanes to spawn based on L's entropy."""
        # Compute output entropy from L's logits
        logits = generation.logits  # [vocab]
        if logits is None:
            return 0
        probs = logits.softmax(dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum().item() / math.log(2)

        if entropy <= ENTROPY_DEMAND_THRESHOLD:
            return 0

        # One lane per ENTROPY_STEP above the threshold
        excess = entropy - ENTROPY_DEMAND_THRESHOLD
        n = 1 + int(excess / ENTROPY_STEP)
        # Also check demand flags from the main pass
        if generation.auto_demands:
            for lobe_name, widened_peers in generation.auto_demands.items():
                if lobe_name == "R" and widened_peers:
                    n = max(n, 2)   # R flagged split → at least 2 lanes
        return min(n, self.max_offspring)

    # ------------------------------------------------------------------
    # C-veto: cancel lanes whose slot_act energy is suspiciously low
    # ------------------------------------------------------------------

    def _apply_c_veto(self, generation: Generation) -> list[str]:
        """Cancel any active offspring that C's attention is ignoring.

        Heuristic: if C's output logit for a lobe's slot band is below the
        mean by >2σ, C is not attending to that lane → cancel it.
        Returns list of cancelled lobe names.
        """
        cancelled = []
        if not self._handles:
            return cancelled
        # Simple stub — in production, read C's tier1 slot energies and compare
        # against offspring slot_act values captured in SharedState.
        # For now: cancel any task that raised an exception (failed jobs).
        still_active = []
        for h in self._handles:
            if h.task.done():
                exc = h.task.exception() if not h.task.cancelled() else None
                if exc is not None:
                    cancelled.append(h.lobe)
            else:
                still_active.append(h)
        self._handles = still_active
        return cancelled

    # ------------------------------------------------------------------
    # Harvest completed offspring results
    # ------------------------------------------------------------------

    def _harvest(self) -> dict[str, str]:
        """Collect results from done offspring tasks. Remove from handle list."""
        results: dict[str, str] = {}
        still_active = []
        for h in self._handles:
            if h.task.done() and not h.task.cancelled():
                try:
                    gen = h.task.result()
                    results[h.lobe] = gen.text
                except Exception:
                    pass   # C-veto or error — already removed above
            else:
                still_active.append(h)
        self._handles = still_active
        return results

    # ------------------------------------------------------------------
    # Build main-pass prompts
    # ------------------------------------------------------------------

    def _build_main_prompts(
        self,
        user_message: str,
        offspring_context: dict[str, str],
    ) -> list[LobePrompt]:
        """Build the 5 main-lobe prompts, injecting any offspring results."""
        ctx = ""
        if offspring_context:
            parts = [f"[{lobe}]: {text}" for lobe, text in offspring_context.items()]
            ctx = "\nParallel exploration results:\n" + "\n".join(parts) + "\n"

        return [
            LobePrompt("L", user_message + ctx),
            LobePrompt("R", f"Execute: {user_message}"),
            LobePrompt("B", f"Context for: {user_message}"),
            LobePrompt("V", "Observe current visual state."),
            LobePrompt("C", f"Verify: {user_message}"),
        ]

    # ------------------------------------------------------------------
    # Single turn
    # ------------------------------------------------------------------

    async def turn(self, user_message: str) -> TurnResult:
        """Run one full turn: main pass → spawn/cancel offspring → reply."""
        self._turn_counter += 1

        # 1. Cancel any timed-out offspring before the main pass
        cancelled = self._apply_c_veto(generation=None)  # type: ignore[arg-type]

        # 2. Harvest offspring that finished during the previous turn
        offspring_ctx = self._harvest()

        # 3. Main 5-lobe pass
        prompts = self._build_main_prompts(user_message, offspring_ctx)
        generation = await self.channels.generate(
            prompts,
            max_new_tokens=self.max_new_tokens,
            shared=self.shared,
        )

        # 4. C-veto pass: cancel offspring that C is ignoring
        cancelled += self._apply_c_veto(generation)

        # 5. Spawn new offspring lanes based on entropy / R demand
        n_spawn = self._how_many_to_spawn(generation)
        spawned: list[str] = []
        for i in range(n_spawn):
            lane_id = len(self._handles) + i + 1
            lobe_name = f"{EXPLORE_PREFIX}{lane_id}"
            goal = (
                f"Explore branch {lane_id} for: {user_message[:200]}"
                f"  [turn={self._turn_counter}]"
            )
            handle = self.offspring_runner.spawn(
                LobePrompt(lobe_name, goal),
                self.shared,
                main_lobe_names=list(generation.tier1.keys()),
                max_new_tokens=self.max_new_tokens,
            )
            self._handles.append(handle)
            spawned.append(lobe_name)

        return TurnResult(
            text=generation.text,
            generation=generation,
            offspring_texts=offspring_ctx,
            spawned=spawned,
            cancelled=cancelled,
        )

    # ------------------------------------------------------------------
    # Session loop (async generator)
    # ------------------------------------------------------------------

    async def session(self, messages: list[str]) -> AsyncIterator[TurnResult]:
        """Drive a sequence of turns, yielding results as they complete."""
        for msg in messages:
            result = await self.turn(msg)
            yield result
        # Let any in-flight offspring finish or cancel them
        for h in self._handles:
            h.task.cancel()
        await asyncio.gather(*(h.task for h in self._handles), return_exceptions=True)
        self._handles.clear()

    # ------------------------------------------------------------------
    # Convenience: simple blocking call for one-shot use
    # ------------------------------------------------------------------

    @classmethod
    def run_once(cls, user_message: str, **kwargs) -> TurnResult:
        """Synchronous wrapper for single-turn use from non-async contexts."""
        async def _go():
            rt = cls(**kwargs)
            return await rt.turn(user_message)
        return asyncio.run(_go())
