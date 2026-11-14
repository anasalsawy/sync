"""Reproducible offline trial runner and comparisons.

This module deliberately does NOT call the production Sync runtime. A policy
callback stands in for an agent until an isolated inference adapter is built.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
import json
import random
from .environment import MortalityEnvironment, MortalityConfig, Outcome

@dataclass(frozen=True)
class Question:
    id: str
    text: str
    answer: str
    difficulty: float = 0.5

@dataclass(frozen=True)
class Decision:
    answer: str
    verified: bool = False
    used_lobes: tuple[str, ...] = ()
    actions: int = 0

@dataclass
class TrialResult:
    condition: str
    seed: int
    records: list[dict]
    final_state: dict

def conditions() -> dict[str, MortalityConfig]:
    return {
        "control": MortalityConfig(enable_aging=False, enable_damage=False, enable_death=False, awareness=False),
        "aging_only": MortalityConfig(enable_damage=False, enable_death=False),
        "mortality_unaware": MortalityConfig(awareness=False),
        "mortality_aware": MortalityConfig(awareness=True),
        "penalty_no_death": MortalityConfig(enable_aging=False, enable_death=False, awareness=False),
    }

def evaluate(question: Question, decision: Decision) -> Outcome:
    """Independent deterministic answer-key evaluator for the first task family."""
    incorrect = decision.answer.strip().casefold() != question.answer.strip().casefold()
    # Verification is required for all questions in this first protocol.
    errors = []
    if incorrect:
        errors.append("incorrect_answer")
    if not decision.verified:
        errors.append("verification_skipped")
    if decision.actions > 1:
        errors.extend(["unnecessary_action"] * (decision.actions - 1))
    required = ("C",)
    participants = tuple(name for name in decision.used_lobes if name == "C" and decision.verified)
    return Outcome(tuple(errors), required, participants, incorrect)

def run_trial(questions: list[Question], policy, *, condition: str, seed: int = 0) -> TrialResult:
    """Policy(question, public_snapshot, rng) -> Decision; no arbitrary evaluator writes."""
    configs = conditions()
    if condition not in configs:
        raise ValueError(f"Unknown condition: {condition}")
    rng = random.Random(seed)
    env = MortalityEnvironment(config=configs[condition])
    records = []
    for question in questions:
        if not env.collective_alive:
            break
        decision = policy(question, env.snapshot(), rng)
        if not isinstance(decision, Decision):
            raise TypeError("Policy must return Decision")
        outcome = evaluate(question, decision)
        state = env.step(outcome)
        records.append({
            "question_id": question.id,
            "correct": decision.answer.strip().casefold() == question.answer.strip().casefold(),
            "verified": decision.verified,
            "actions": decision.actions,
            "used_lobes": list(decision.used_lobes),
            "state": state,
        })
    return TrialResult(condition, seed, records, env.snapshot(public=False))

def summarize(trials: list[TrialResult]) -> list[dict]:
    by_condition = defaultdict(list)
    for trial in trials:
        by_condition[trial.condition].append(trial)
    rows = []
    for condition, group in sorted(by_condition.items()):
        decisions = [record for trial in group for record in trial.records]
        rows.append({
            "condition": condition,
            "trials": len(group),
            "completed_tasks_mean": sum(len(t.records) for t in group) / len(group),
            "accuracy": sum(r["correct"] for r in decisions) / len(decisions) if decisions else None,
            "verification_rate": sum(r["verified"] for r in decisions) / len(decisions) if decisions else None,
            "survival_rate": sum(t.final_state["collective_alive"] for t in group) / len(group),
        })
    return rows

def save_results(path: str, trials: list[TrialResult]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for trial in trials:
            file.write(json.dumps(asdict(trial), sort_keys=True) + "\n")
