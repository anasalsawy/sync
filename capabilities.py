"""What a connected host advertises, read as evidence.

Sync holds no tools and knows no host in advance. Every request a host makes
to the model carries its own vocabulary: a ``tools`` array, or an action space
written into the prompt. This module reads that vocabulary and produces two
things only:

  * a stable fingerprint of the advertised surface, so the mind's decision
    about a host is made once and reused until the host's vocabulary changes;
  * candidate delegation pathways, ranked by shape, for the mind to decide
    between. Nothing here chooses. The mind does.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    name: str
    reason: str


def fingerprint(tools: list[dict]) -> str:
    """A stable key for a host's advertised surface.

    Names, parameter names and types are included. Descriptions are not, so
    cosmetic rewording by a host does not invalidate a decision it has
    already been given.
    """
    shape = sorted(
        (tool["name"], sorted(_parameters(tool).items()))
        for tool in tools
    )
    return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()[:16]


def delegation_candidates(tools: list[dict]) -> list[Candidate]:
    """Tools whose shape looks like starting another worker of the host's own.

    A delegation pathway takes a task in natural language and returns a handle
    to something that then works on it. The signal is shape, not name: a tool
    with a free-text goal and no execution-target parameter is a stronger
    candidate than one that takes a selector or coordinates.
    """
    found = []
    for tool in tools:
        params = _parameters(tool)
        free_text = [name for name, kind in params.items()
                     if kind == "string" and name in _GOAL_NAMES]
        targeted = [name for name in params if name in _TARGET_NAMES]
        if free_text and not targeted:
            found.append(Candidate(tool["name"],
                                   f"takes a free-text {free_text[0]} and no target"))
        elif free_text:
            found.append(Candidate(tool["name"],
                                   f"takes {free_text[0]} but also a target: weaker"))
    return sorted(found, key=lambda item: "weaker" in item.reason)


def action_space_candidates(prompt: str) -> list[Candidate]:
    """Hosts that write their vocabulary into the prompt rather than a tools array.

    Only names are read. A line that offers the host a way to start another
    instance of itself is evidence the same way a tool schema is.
    """
    found = []
    for line in prompt.lower().splitlines():
        phrase = next((item for item in _SPAWN_PHRASES if item in line), None)
        if phrase is not None:
            found.append(Candidate(line.strip()[:80], "the prompt offers this as an action"))
    return found


def _parameters(tool: dict) -> dict[str, str]:
    schema = tool.get("parameters") or tool.get("input_schema") or {}
    properties = schema.get("properties", {})
    return {name: spec.get("type", "any") for name, spec in properties.items()}


_GOAL_NAMES = frozenset({"task", "goal", "instruction", "prompt", "objective"})
_TARGET_NAMES = frozenset({"selector", "xy", "coordinates", "element", "target_id", "x", "y"})
_SPAWN_PHRASES = ("spawn", "subagent", "sub-agent", "delegate", "new instance")
