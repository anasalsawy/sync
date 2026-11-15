"""Benchmark 1 / 3 / 5 R workers on sequential and parallel executors.

One-worker is an explicit eval control. Three is the provisional default.
Raise the default to five only if five consistently finishes faster on both
executor modes while preserving L's solo-like TTFT and matched-step pace.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from multi_r_config import (
    DEFAULT_MULTI_R_WORKERS,
    EVAL_CONTROL_ONE_WORKER,
    SCALE_CANDIDATES,
)
from r_coordinator import ParallelExecutor, RCoordinator, SequentialExecutor
from sync_session import SyncSession


SEQ_GOAL = (
    "Pipeline: read the inbox, write a reply to outbox, then send it."
)
PAR_GOAL = (
    "Process independent items in parallel: alpha beta gamma delta"
)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


async def _l_latency(*, tokens: int = 6) -> dict:
    s = SyncSession(
        backend="sim", max_new_tokens=tokens, solo=True, isolate_peers=False,
    )
    r = await s.turn("Brief status while R works.")
    steps = r.matched_step_ms()
    await s.close()
    return {
        "ttft_s": r.ttft_s,
        "matched_step_ms_p50": _pct(steps, 0.5) if steps else None,
        "matched_step_ms_p95": _pct(steps, 0.95) if steps else None,
        "l_path_ok": r.l_path_ok,
    }


async def run_once(
    n_workers: int,
    *,
    mode: str,
    goal: str,
    delay_s: float,
    measure_l: bool,
) -> dict:
    if mode == "parallel":
        executor = ParallelExecutor(delay_s=delay_s, max_parallel=4)
    else:
        executor = SequentialExecutor(delay_s=delay_s)
    coord = RCoordinator(
        n_workers=n_workers,
        cross_attn=(n_workers > 1),
        max_new_tokens=6,
        mode=mode,
        executor=executor,
    )
    # Confirm full batch launched
    assert len(coord.team.workers) == n_workers

    l_before = await _l_latency() if measure_l else None
    l_during_holder: dict = {}

    async def _l_bg() -> None:
        await asyncio.sleep(0.02)
        l_during_holder["v"] = await _l_latency()

    l_task = asyncio.create_task(_l_bg()) if measure_l else None
    result = await coord.run_task(
        goal,
        constraints=f"mode={mode}; n={n_workers}",
        source_material="bench",
        force_duplicate=False,
    )
    if l_task is not None:
        await l_task

    ids = [d["action_id"] for d in result.dispatch_log]
    return {
        "n_workers": n_workers,
        "mode": mode,
        "total_task_s": result.metrics.total_task_s,
        "executor_idle_s": result.metrics.executor_idle_s,
        "executor_busy_s": result.metrics.executor_busy_s,
        "prep_overlap_s": result.metrics.prep_overlap_s,
        "duplicate_dispatches": result.metrics.duplicate_dispatches,
        "dup_ids_in_log": len(ids) != len(set(ids)),
        "final_fingerprint": result.final_fingerprint,
        "satisfied": result.satisfied,
        "workers_launched": len(coord.team.workers),
        "l_before": l_before,
        "l_during": l_during_holder.get("v"),
    }


async def collect(
    n_workers: int, mode: str, goal: str, *, repeats: int, delay_s: float,
) -> dict:
    runs = []
    for i in range(repeats):
        runs.append(await run_once(
            n_workers, mode=mode, goal=goal, delay_s=delay_s,
            measure_l=(i == 0),
        ))
    totals = [r["total_task_s"] for r in runs]
    idles = [r["executor_idle_s"] for r in runs]
    fps = {r["final_fingerprint"] for r in runs}
    dups = any(r["dup_ids_in_log"] or r["duplicate_dispatches"] for r in runs)
    return {
        "n_workers": n_workers,
        "mode": mode,
        "repeats": repeats,
        "total_task_s": {"p50": _pct(totals, 0.5), "p95": _pct(totals, 0.95), "mean": statistics.mean(totals)},
        "executor_idle_s": {"p50": _pct(idles, 0.5), "mean": statistics.mean(idles)},
        "final_fingerprint": next(iter(fps)) if len(fps) == 1 else sorted(fps),
        "fingerprint_stable": len(fps) == 1,
        "zero_duplicate_dispatches": not dups,
        "all_satisfied": all(r["satisfied"] for r in runs),
        "workers_launched": runs[0]["workers_launched"],
        "l_before": runs[0].get("l_before"),
        "l_during": runs[0].get("l_during"),
        "runs": runs,
    }


def _l_ok(row: dict, solo: dict, *, tol_ttft: float = 2.0, tol_step: float = 2.5) -> bool:
    """Preserve L solo pace: during must stay within tolerance of solo baseline."""
    ld = row.get("l_during") or {}
    if ld.get("ttft_s") is None or solo.get("ttft_s") is None:
        return True
    if ld["ttft_s"] > solo["ttft_s"] * tol_ttft + 0.005:
        return False
    if (
        ld.get("matched_step_ms_p50") is not None
        and solo.get("matched_step_ms_p50") is not None
        and ld["matched_step_ms_p50"] > solo["matched_step_ms_p50"] * tol_step + 0.5
    ):
        return False
    return True


def decide_default(report: dict) -> dict:
    """Keep three unless five consistently beats three and preserves L pace."""
    seq3 = report["sequential"]["3"]
    seq5 = report["sequential"]["5"]
    par3 = report["parallel"]["3"]
    par5 = report["parallel"]["5"]
    solo = report["l_solo_baseline"]

    five_faster_seq = seq5["total_task_s"]["p50"] < seq3["total_task_s"]["p50"] * 0.95
    five_faster_par = par5["total_task_s"]["p50"] < par3["total_task_s"]["p50"] * 0.95
    five_preserves_l = _l_ok(seq5, solo) and _l_ok(par5, solo)
    consistently_faster = five_faster_seq and five_faster_par

    recommend = 5 if (consistently_faster and five_preserves_l) else DEFAULT_MULTI_R_WORKERS
    return {
        "current_default": DEFAULT_MULTI_R_WORKERS,
        "recommend_default": recommend,
        "keep_three": recommend == 3,
        "five_faster_sequential": five_faster_seq,
        "five_faster_parallel": five_faster_par,
        "five_preserves_l_solo_pace": five_preserves_l,
        "reason": (
            "five consistently faster on seq+par while preserving L pace — raise default"
            if recommend == 5
            else "keep three: five does not consistently beat three on sequential "
                 "and parallel while preserving L solo pace"
        ),
        "seq_p50": {"3": seq3["total_task_s"]["p50"], "5": seq5["total_task_s"]["p50"]},
        "par_p50": {"3": par3["total_task_s"]["p50"], "5": par5["total_task_s"]["p50"]},
    }


async def main_async(args: argparse.Namespace) -> dict:
    l_solo = await _l_latency()
    sequential = {}
    parallel = {}
    for n in SCALE_CANDIDATES:
        sequential[str(n)] = await collect(
            n, "sequential", SEQ_GOAL, repeats=args.repeats, delay_s=args.delay,
        )
        parallel[str(n)] = await collect(
            n, "parallel", PAR_GOAL, repeats=args.repeats, delay_s=args.delay,
        )

    # Cross-config: same final result within each mode across worker counts
    seq_fps = {sequential[str(n)]["final_fingerprint"] for n in SCALE_CANDIDATES}
    par_fps = {parallel[str(n)]["final_fingerprint"] for n in SCALE_CANDIDATES}
    zero_dups = all(
        sequential[str(n)]["zero_duplicate_dispatches"]
        and parallel[str(n)]["zero_duplicate_dispatches"]
        for n in SCALE_CANDIDATES
    )
    decision = decide_default({
        "sequential": sequential,
        "parallel": parallel,
        "l_solo_baseline": l_solo,
    })

    accept = {
        "pass": (
            len(seq_fps) == 1
            and len(par_fps) == 1
            and zero_dups
            and all(sequential[str(n)]["all_satisfied"] for n in SCALE_CANDIDATES)
            and all(parallel[str(n)]["all_satisfied"] for n in SCALE_CANDIDATES)
            and all(
                sequential[str(n)]["workers_launched"] == n
                and parallel[str(n)]["workers_launched"] == n
                for n in SCALE_CANDIDATES
            )
        ),
        "same_final_result_sequential": len(seq_fps) == 1,
        "same_final_result_parallel": len(par_fps) == 1,
        "zero_duplicate_dispatches": zero_dups,
        "default_remains_three": decision["keep_three"],
    }

    return {
        "generated_at": time.time(),
        "default_multi_r_workers": DEFAULT_MULTI_R_WORKERS,
        "eval_control_one_worker": EVAL_CONTROL_ONE_WORKER,
        "scale_candidates": list(SCALE_CANDIDATES),
        "l_solo_baseline": l_solo,
        "sequential": sequential,
        "parallel": parallel,
        "decision": decision,
        "accept": accept,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument(
        "--out", type=Path, default=Path("eval_results/multi_r_scale.json"),
    )
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Multi-R scale: 1 (control) / 3 (default) / 5 ===\n")
    for mode in ("sequential", "parallel"):
        print(f"-- {mode} --")
        print(f"{'n':>3}  {'total_p50':>10}  {'idle_p50':>10}  {'L_ttft':>10}  {'L_step':>10}  dups")
        for n in SCALE_CANDIDATES:
            row = report[mode][str(n)]
            ld = row.get("l_during") or {}
            print(
                f"{n:3d}  {row['total_task_s']['p50']:10.4f}  "
                f"{row['executor_idle_s']['p50']:10.4f}  "
                f"{(ld.get('ttft_s') or 0):10.4f}  "
                f"{(ld.get('matched_step_ms_p50') or 0):10.4f}  "
                f"{'ok' if row['zero_duplicate_dispatches'] else 'FAIL'}"
            )
        print()
    d = report["decision"]
    print(f"default={d['current_default']}  recommend={d['recommend_default']}")
    print(f"  five_faster_seq={d['five_faster_sequential']}  "
          f"five_faster_par={d['five_faster_parallel']}  "
          f"preserves_L={d['five_preserves_l_solo_pace']}")
    print(f"  {d['reason']}")
    print(f"\nwrote {args.out}")
    print("MULTI_R_SCALE_PASS" if report["accept"]["pass"] else "MULTI_R_SCALE_NEEDS_WORK")
    if report["accept"]["default_remains_three"]:
        print("DEFAULT_REMAINS_THREE")
    else:
        print("DEFAULT_CANDIDATE_FIVE")
    if not report["accept"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
