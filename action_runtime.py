"""action_runtime.py — Parallel action execution across offspring lobes.

Design:
  - Each offspring independently decides its next action from the full task context.
  - No action assignment. Wiring asymmetry drives natural specialization:
      R  → execution / environment manipulation
      B  → retrieval / lookup
      V  → visual / spatial / coordinate actions
      R1+ → parallel exploration of the same action space
  - Offspring emit structured ActionProposals, not free text.
  - An optimistic lock prevents two offspring targeting the same resource
    simultaneously — first to claim wins, others skip that target.
  - asyncio.gather runs all claimed actions in parallel.
  - Results collapse into one EnvironmentState for the next main pass.

The host sees one unified state update, not 5 streams.
Latency = slowest single action, not sum of all actions.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


# ---------------------------------------------------------------------------
# Action schema — what an offspring emits
# ---------------------------------------------------------------------------

@dataclass
class ActionProposal:
    """Structured action emitted by an offspring lobe."""
    lobe: str
    action_type: str          # "click", "fill", "scroll", "retrieve", "observe", "wait"
    target: str               # CSS selector, URL, field name, query — whatever the action acts on
    payload: Any = None       # value to fill, scroll delta, search query, etc.
    priority: float = 1.0     # higher = claim wins on conflict (driven by lobe confidence)
    raw: str = ""             # raw offspring output for debugging

    def lock_key(self) -> str:
        """Two proposals conflict if they share the same (action_type, target)."""
        return f"{self.action_type}::{self.target}"


@dataclass
class ActionResult:
    lobe: str
    action_type: str
    target: str
    success: bool
    output: Any = None        # what the action returned (screenshot, retrieved text, etc.)
    error: str = ""
    elapsed_s: float = 0.0


@dataclass
class EnvironmentState:
    """Unified environment snapshot after one parallel action round."""
    turn: int
    actions_attempted: int
    actions_succeeded: int
    results: list[ActionResult] = field(default_factory=list)
    aggregated: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_context(self) -> str:
        """Compact summary injected into the next main pass as plain text."""
        lines = [f"[env t={self.turn} actions={self.actions_succeeded}/{self.actions_attempted}]"]
        for r in self.results:
            status = "✓" if r.success else "✗"
            out = str(r.output)[:120] if r.output else ""
            lines.append(f"  {status} {r.lobe} {r.action_type}({r.target}): {out}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parser — extracts ActionProposal from offspring free text
# ---------------------------------------------------------------------------

def parse_action(lobe: str, text: str) -> ActionProposal | None:
    """
    Parse a structured action from offspring output.

    Offspring are expected to emit JSON anywhere in their output:
        {"action": "click", "target": "#submit-btn"}
        {"action": "fill",  "target": "#email", "value": "user@example.com"}
        {"action": "retrieve", "target": "latest price of AAPL"}
        {"action": "observe", "target": "full page"}

    Falls back to heuristic extraction if no JSON found.
    """
    # Try JSON block first
    import re
    for match in re.finditer(r'\{[^{}]+\}', text):
        try:
            obj = json.loads(match.group())
            action_type = obj.get("action") or obj.get("type") or obj.get("action_type")
            target = obj.get("target") or obj.get("selector") or obj.get("query") or ""
            if action_type and target:
                return ActionProposal(
                    lobe=lobe,
                    action_type=action_type.lower(),
                    target=target,
                    payload=obj.get("value") or obj.get("payload") or obj.get("text"),
                    priority=float(obj.get("priority", 1.0)),
                    raw=text,
                )
        except (json.JSONDecodeError, ValueError):
            continue

    # Heuristic: look for action keywords
    text_lower = text.lower()
    for keyword in ("click", "fill", "scroll", "retrieve", "observe", "navigate", "wait"):
        if keyword in text_lower:
            # Extract the target as whatever follows the keyword
            idx = text_lower.find(keyword)
            snippet = text[idx + len(keyword):idx + len(keyword) + 100].strip().strip(":()\n")
            if snippet:
                return ActionProposal(
                    lobe=lobe,
                    action_type=keyword,
                    target=snippet[:80],
                    raw=text,
                )

    return None


# ---------------------------------------------------------------------------
# Optimistic action lock — first claim wins
# ---------------------------------------------------------------------------

class ActionLock:
    """Prevents two offspring from acting on the same target simultaneously."""

    def __init__(self):
        self._claimed: dict[str, str] = {}   # lock_key → lobe name
        self._lock = asyncio.Lock()

    async def try_claim(self, proposal: ActionProposal) -> bool:
        async with self._lock:
            key = proposal.lock_key()
            if key in self._claimed:
                return False
            self._claimed[key] = proposal.lobe
            return True

    def reset(self):
        self._claimed.clear()


# ---------------------------------------------------------------------------
# Executor registry — maps action_type → async handler
# ---------------------------------------------------------------------------

# Handlers are async callables: (proposal) -> ActionResult
# Register your environment's real handlers here.
_HANDLERS: dict[str, Callable[[ActionProposal], Awaitable[ActionResult]]] = {}


def register_handler(action_type: str):
    """Decorator to register an action handler."""
    def decorator(fn):
        _HANDLERS[action_type] = fn
        return fn
    return decorator


async def _default_handler(proposal: ActionProposal) -> ActionResult:
    """Fallback: log the action but don't execute it (for environments not yet wired)."""
    return ActionResult(
        lobe=proposal.lobe,
        action_type=proposal.action_type,
        target=proposal.target,
        success=False,
        error="no handler registered",
    )


async def execute_action(proposal: ActionProposal) -> ActionResult:
    t0 = time.time()
    handler = _HANDLERS.get(proposal.action_type, _default_handler)
    try:
        result = await handler(proposal)
        result.elapsed_s = time.time() - t0
        return result
    except Exception as e:
        return ActionResult(
            lobe=proposal.lobe,
            action_type=proposal.action_type,
            target=proposal.target,
            success=False,
            error=str(e),
            elapsed_s=time.time() - t0,
        )


# ---------------------------------------------------------------------------
# Parallel action round
# ---------------------------------------------------------------------------

async def run_action_round(
    offspring_outputs: dict[str, str],   # lobe_name → raw offspring text
    turn: int = 0,
) -> EnvironmentState:
    """
    Parse proposals from all offspring, claim targets optimistically,
    execute all claimed actions in parallel, return unified EnvironmentState.

    offspring_outputs: {lobe_name: raw_text} from SharedState offspring results
    """
    lock = ActionLock()

    # Parse proposals from each offspring
    proposals: list[ActionProposal] = []
    for lobe, text in offspring_outputs.items():
        proposal = parse_action(lobe, text)
        if proposal:
            proposals.append(proposal)

    if not proposals:
        return EnvironmentState(
            turn=turn,
            actions_attempted=0,
            actions_succeeded=0,
        )

    # Sort by priority descending — highest confidence claims first
    proposals.sort(key=lambda p: p.priority, reverse=True)

    # Claim targets optimistically
    claimed: list[ActionProposal] = []
    for proposal in proposals:
        if await lock.try_claim(proposal):
            claimed.append(proposal)

    # Execute all claimed actions in parallel
    results = await asyncio.gather(*[execute_action(p) for p in claimed])

    succeeded = sum(1 for r in results if r.success)

    # Aggregate outputs by action type for easy consumption
    aggregated: dict[str, Any] = {}
    for r in results:
        if r.success and r.output is not None:
            aggregated.setdefault(r.action_type, []).append(r.output)

    return EnvironmentState(
        turn=turn,
        actions_attempted=len(claimed),
        actions_succeeded=succeeded,
        results=list(results),
        aggregated=aggregated,
    )


# ---------------------------------------------------------------------------
# Integration with SyncRuntime
# ---------------------------------------------------------------------------

class ActionRuntime:
    """
    Drop-in companion to SyncRuntime.

    After each main pass, call .run(shared, turn) to:
      1. Read completed offspring outputs from SharedState
      2. Parse their action proposals
      3. Execute in parallel
      4. Return EnvironmentState for injection into the next turn
    """

    def __init__(self):
        self._turn = 0

    async def run(self, shared_offspring_results: dict[str, str]) -> EnvironmentState:
        self._turn += 1
        state = await run_action_round(shared_offspring_results, turn=self._turn)
        if state.actions_attempted:
            print(
                f"[action] turn={self._turn} "
                f"{state.actions_succeeded}/{state.actions_attempted} actions succeeded "
                f"in parallel | types: {list(state.aggregated.keys())}",
                flush=True,
            )
        return state


# ---------------------------------------------------------------------------
# Example handler registrations (replace with real environment bindings)
# ---------------------------------------------------------------------------

@register_handler("observe")
async def _handle_observe(p: ActionProposal) -> ActionResult:
    """Placeholder — replace with real screenshot / DOM read."""
    return ActionResult(lobe=p.lobe, action_type=p.action_type,
                        target=p.target, success=True,
                        output=f"[observed: {p.target}]")


@register_handler("wait")
async def _handle_wait(p: ActionProposal) -> ActionResult:
    delay = float(p.payload or 0.5)
    await asyncio.sleep(min(delay, 5.0))
    return ActionResult(lobe=p.lobe, action_type=p.action_type,
                        target=p.target, success=True,
                        output=f"waited {delay}s")
