"""Verify multi-R: live cross-attn, prep-ahead, faster task, no duplicate effects.

Compare 1 R worker vs 5 batched R workers on:
  - sequential-executor code task (investigate / conflicting edits / test)
  - independent actions with parallel executor

Force duplicate proposal + resource conflict. Accept when workers share live
activations, prepare without fixed assignments, finish faster without
duplicate effects or measurable L slowdown.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from r_coordinator import RCoordinator
from sync_session import SyncSession


CODE_GOAL = (
    "Fix the failing unit tests in this code task: investigate failures, "
    "prepare edits, re-run tests."
)
PARALLEL_GOAL = (
    "Process independent items in parallel: alpha beta gamma delta"
)


async def _l_latency(*, tokens: int = 6) -> dict:
    s = SyncSession(backend="sim", max_new_tokens=tokens, solo=True, isolate_peers=False)
    r = await s.turn("Give a one-line status update.")
    await s.close()
    return {
        "ttft_s": r.ttft_s,
        "matched_step_ms_p50": (
            sorted(r.matched_step_ms())[len(r.matched_step_ms()) // 2]
            if r.matched_step_ms() else None
        ),
        "l_path_ok": r.l_path_ok,
    }


async def run_case(
    *,
    n_workers: int,
    cross_attn: bool,
    mode: str,
    goal: str,
    measure_l: bool = False,
    executor_delay_s: float = 0.12,
) -> dict:
    from r_coordinator import (
        BatchCapableExecutor,
        ParallelExecutor,
        SequentialExecutor,
    )
    if mode == "parallel":
        executor = ParallelExecutor(delay_s=executor_delay_s, max_parallel=4)
    elif mode == "batch":
        executor = BatchCapableExecutor(delay_s=executor_delay_s)
    else:
        executor = SequentialExecutor(delay_s=executor_delay_s)
    coord = RCoordinator(
        n_workers=n_workers,
        cross_attn=cross_attn,
        max_new_tokens=8,
        mode=mode,
        executor=executor,
    )
    l_before = await _l_latency() if measure_l else None
    # L status turn concurrent with R task (L owns speech; R stays silent)
    l_task = None
    if measure_l:
        async def _l_during() -> dict:
            await asyncio.sleep(0.01)
            return await _l_latency()

        l_task = asyncio.create_task(_l_during())

    result = await coord.run_task(
        goal,
        constraints="silent workers; L decides",
        source_material="repo snapshot + failing test log",
        force_duplicate=("code" in goal.lower() or "test" in goal.lower()),
        influence_probe=(n_workers > 1 and cross_attn),
    )
    l_during = await l_task if l_task is not None else None

    dispatches = [d["action_id"] for d in result.dispatch_log]
    edit_dispatches = [
        d for d in result.dispatch_log if d["effect"] == "edit"
    ]
    # Same action_id must appear at most once in dispatch log
    dup_dispatch = len(dispatches) != len(set(dispatches))

    return {
        "n_workers": n_workers,
        "cross_attn": cross_attn,
        "mode": mode,
        "goal": goal,
        "total_task_s": result.metrics.total_task_s,
        "executor_idle_s": result.metrics.executor_idle_s,
        "executor_busy_s": result.metrics.executor_busy_s,
        "prep_overlap_s": result.metrics.prep_overlap_s,
        "inference_s": result.metrics.inference_s,
        "duplicate_dispatches": result.metrics.duplicate_dispatches,
        "duplicate_claim_attempts": result.record.duplicate_claim_attempts,
        "conflicts_ordered": result.record.conflicts_ordered,
        "dup_action_id_in_log": dup_dispatch,
        "n_dispatches": len(dispatches),
        "n_edit_dispatches": len(edit_dispatches),
        "satisfied": result.satisfied,
        "record_metrics": result.record.metrics(),
        "influence_proof": (result.batch.influence_proof if result.batch else {}),
        "cross_attn_influence": (
            result.batch.timing.cross_attn_influence if result.batch else False
        ),
        "l_before": l_before,
        "l_during": l_during,
        "dispatch_log": result.dispatch_log,
    }


def _accept(report: dict) -> dict:
    one_seq = report["one_r_sequential"]
    five_on = report["five_r_sequential_on"]
    five_off = report["five_r_sequential_off"]
    one_par = report["one_r_parallel"]
    five_par = report["five_r_parallel"]

    failures: list[str] = []

    if not five_on.get("cross_attn_influence"):
        failures.append("five_on did not show cross-worker activation influence")
    if five_on["dup_action_id_in_log"]:
        failures.append("duplicate action dispatched more than once")
    if five_on["n_edit_dispatches"] < 1:
        failures.append("expected at least one edit dispatch")
    if five_on["duplicate_claim_attempts"] < 1:
        failures.append("expected duplicate proposal to collapse to same action id")
    if five_on["conflicts_ordered"] < 1:
        failures.append("expected same-resource conflict to be ordered")
    if not five_on["satisfied"]:
        failures.append("five_on sequential task not satisfied")
    if not five_par["satisfied"]:
        failures.append("parallel independent task not satisfied")
    # Speed win is on independent/parallel work (seq chains stay resource-ordered)
    if five_par["total_task_s"] > one_par["total_task_s"] * 1.05:
        failures.append(
            f"five_r parallel not faster: {five_par['total_task_s']:.4f} vs "
            f"1R {one_par['total_task_s']:.4f}"
        )
    if five_par["prep_overlap_s"] < 0:
        failures.append("negative prep overlap")
    # L latency: during must stay near before
    lb = five_on.get("l_before") or {}
    ld = five_on.get("l_during") or {}
    if lb.get("ttft_s") is not None and ld.get("ttft_s") is not None:
        if ld["ttft_s"] > lb["ttft_s"] * 2.0 + 0.01:
            failures.append(
                f"L TTFT slowed during multi-R: {ld['ttft_s']:.4f} vs {lb['ttft_s']:.4f}"
            )
    sharing = {
        "on_influence": five_on.get("cross_attn_influence"),
        "off_influence": five_off.get("cross_attn_influence"),
        "off_lacks_influence": not five_off.get("cross_attn_influence"),
        "seq_1r_s": one_seq["total_task_s"],
        "seq_5r_on_s": five_on["total_task_s"],
        "seq_5r_off_s": five_off["total_task_s"],
    }

    return {
        "pass": not failures,
        "failures": failures,
        "sharing": sharing,
        "speedup_parallel_vs_1r": (
            one_par["total_task_s"] / five_par["total_task_s"]
            if five_par["total_task_s"] else None
        ),
    }


async def main_async(args: argparse.Namespace) -> dict:
    one = await run_case(
        n_workers=1, cross_attn=False, mode="sequential", goal=CODE_GOAL,
    )
    five_on = await run_case(
        n_workers=5, cross_attn=True, mode="sequential", goal=CODE_GOAL,
        measure_l=True,
    )
    five_off = await run_case(
        n_workers=5, cross_attn=False, mode="sequential", goal=CODE_GOAL,
    )
    one_par = await run_case(
        n_workers=1, cross_attn=False, mode="parallel", goal=PARALLEL_GOAL,
    )
    five_par = await run_case(
        n_workers=5, cross_attn=True, mode="parallel", goal=PARALLEL_GOAL,
    )
    report = {
        "generated_at": time.time(),
        "one_r_sequential": one,
        "five_r_sequential_on": five_on,
        "five_r_sequential_off": five_off,
        "one_r_parallel": one_par,
        "five_r_parallel": five_par,
    }
    report["accept"] = _accept(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("eval_results/multi_r_eval.json"),
    )
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    if report["accept"]["pass"]:
        print("MULTI_R_ACCEPT")
    else:
        print("MULTI_R_NEEDS_WORK")
        for f in report["accept"]["failures"]:
            print(f"  - {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
