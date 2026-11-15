"""Rubric scoring for Sync efficacy runs."""
from __future__ import annotations

from dataclasses import dataclass

from eval_bank import EvalItem, Rubric


@dataclass
class Score:
    hit_required: int
    need_required: int
    hit_any: int
    need_any: int
    hit_forbidden: int
    need_forbidden: int
    pass_: bool
    detail: str

    @property
    def recall(self) -> float:
        parts = []
        if self.need_required:
            parts.append(self.hit_required / self.need_required)
        if self.need_any:
            parts.append(1.0 if self.hit_any else 0.0)
        return sum(parts) / len(parts) if parts else 1.0

    @property
    def clean(self) -> float:
        if self.need_forbidden == 0:
            return 1.0
        return 1.0 - (self.hit_forbidden / self.need_forbidden)


def score_reply(text: str, rubric: Rubric) -> Score:
    low = (text or "").lower()
    req = rubric.must_include
    any_req = rubric.must_include_any
    bad = rubric.must_not_include
    hit_req = sum(1 for s in req if s.lower() in low)
    hit_any = sum(1 for s in any_req if s.lower() in low)
    hit_bad = sum(1 for s in bad if s.lower() in low)

    req_ok = hit_req == len(req)
    any_ok = (hit_any >= 1) if any_req else True
    clean_ok = hit_bad == 0
    if req or any_req or bad:
        passed = req_ok and any_ok and clean_ok
    else:
        passed = bool(text.strip())

    detail = (
        f"all {hit_req}/{len(req)}; any {hit_any}/{len(any_req)}; "
        f"forbid_hits {hit_bad}/{len(bad)}"
    )
    return Score(
        hit_required=hit_req,
        need_required=len(req),
        hit_any=hit_any,
        need_any=len(any_req),
        hit_forbidden=hit_bad,
        need_forbidden=len(bad),
        pass_=passed,
        detail=detail,
    )


def score_item(text: str, item: EvalItem) -> Score:
    return score_reply(text, item.rubric)


def connected_helps(solo: Score, connected: Score, item: EvalItem) -> bool | None:
    """Did connecting peers improve the outcome when peers were needed?"""
    if not item.rubric.needs_peers:
        return connected.pass_ or (
            connected.recall >= solo.recall and connected.clean >= solo.clean
        )
    if connected.pass_ and not solo.pass_:
        return True
    if connected.recall > solo.recall and connected.clean >= solo.clean:
        return True
    if connected.pass_ and solo.pass_:
        return True
    return False
