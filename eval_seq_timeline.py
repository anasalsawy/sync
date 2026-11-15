"""1R vs 5R on one sequential-executor task with prep-ahead timeline.

Reports total task time, executor idle waiting for the next action, and L's
first-token + matched-step latency. Confirms identical final fingerprint and
zero duplicate dispatches. Also verifies companion submit_task invokes Multi-R.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from companion_session import CompanionSession
from r_coordinator import RCoordinator, SequentialExecutor
from sync_session import SyncSession


GOAL = (
    "Pipeline: read the inbox, write a reply to outbox, then send it. "
    "Workers may prepare later steps while an earlier step executes."
)


async def _l_metrics(*, tokens: int = 6) -> dict:
    s = SyncSession(
        backend="sim", max_new_tokens=tokens, solo=True, isolate_peers=False,
    )
    r = await s.turn("Brief status: task is running.")
    steps = r.matched_step_ms()
    await s.close()
    return {
        "ttft_s": r.ttft_s,
        "matched_step_ms": steps,
        "matched_step_ms_p50": sorted(steps)[len(steps) // 2] if steps else None,
        "l_path_ok": r.l_path_ok,
    }


def _timeline_rows(timeline: list) -> list[dict]:
    return [
        {"t_s": round(ev.at, 4), "kind": ev.kind, **ev.detail}
        for ev in timeline
    ]


def _prep_during_exec(timeline: list) -> list[dict]:
    """Prepare events that fired while the sequential executor was busy."""
    busy_until = 0.0
    in_dispatch = False
    dispatch_start = 0.0
    out = []
    for ev in timeline:
        if ev.kind == "dispatch_start":
            in_dispatch = True
            dispatch_start = ev.at
        elif ev.kind == "dispatch_end":
            busy_until = ev.at
            in_dispatch = False
        elif ev.kind == "prepare":
            during = bool(ev.detail.get("while_executor_busy")) or (
                in_dispatch and ev.at >= dispatch_start
            )
            if during or (ev.detail.get("speculative") and busy_until > 0):
                # Also count speculative prepares that overlap open dispatch window
                if during or (in_dispatch and ev.detail.get("speculative")):
                    out.append({
                        "t_s": round(ev.at, 4),
                        "worker": ev.detail.get("worker"),
                        "effect": ev.detail.get("effect"),
                        "action_id": ev.detail.get("action_id"),
                        "speculative": ev.detail.get("speculative"),
                        "while_executor_busy": True,
                    })
    # Prefer explicit while_executor_busy prepares
    explicit = [
        {
            "t_s": round(ev.at, 4),
            "worker": ev.detail.get("worker"),
            "effect": ev.detail.get("effect"),
            "action_id": ev.detail.get("action_id"),
            "speculative": ev.detail.get("speculative"),
            "while_executor_busy": True,
        }
        for ev in timeline
        if ev.kind == "prepare" and ev.detail.get("while_executor_busy")
    ]
    return explicit or out


async def run_workers(n: int, *, delay_s: float = 0.12) -> dict:
    ex = SequentialExecutor(delay_s=delay_s)
    coord = RCoordinator(
        n_workers=n,
        cross_attn=(n > 1),
        max_new_tokens=6,
        mode="sequential",
        executor=ex,
    )
    # L latency sampled during the task
    l_holder: dict = {}

    async def _l_during() -> None:
        await asyncio.sleep(0.02)
        l_holder["during"] = await _l_metrics()

    l_task = asyncio.create_task(_l_during())
    l_holder["before"] = await _l_metrics()
    result = await coord.run_task(
        GOAL,
        constraints="strictly sequential executor; silent R",
        source_material="inbox messages",
        force_duplicate=False,
        influence_probe=False,
    )
    await l_task

    ids = [d["action_id"] for d in result.dispatch_log]
    return {
        "n_workers": n,
        "total_task_s": result.metrics.total_task_s,
        "executor_idle_s": result.metrics.executor_idle_s,
        "executor_busy_s": result.metrics.executor_busy_s,
        "prep_overlap_s": result.metrics.prep_overlap_s,
        "duplicate_dispatches": result.metrics.duplicate_dispatches,
        "dup_ids_in_log": len(ids) != len(set(ids)),
        "final_fingerprint": result.final_fingerprint,
        "satisfied": result.satisfied,
        "dispatch_log": result.dispatch_log,
        "timeline": _timeline_rows(result.timeline),
        "prep_while_executing": _prep_during_exec(result.timeline),
        "l_before": l_holder.get("before"),
        "l_during": l_holder.get("during"),
        "record_metrics": result.record.metrics(),
    }


async def verify_companion_path() -> dict:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        events: list[dict] = []
        session = CompanionSession(
            db_path=str(root / "c.db"),
            episodic_path=str(root / "e.db"),
            backend="sim",
            max_new_tokens=4,
            on_event=events.append,
        )
        session.multi_r_workers = 3  # provisional default
        task_id = await session.submit_task(GOAL, require_c=False)
        invoked = [e for e in events if e.get("type") == "multi_r_invoked"]
        out = {
            "submit_returned": task_id,
            "multi_r_invoked_event": bool(invoked),
            "invocations": list(session.multi_r_invocations),
            "via_authorize_task": bool(
                session.multi_r_invocations
                and session.multi_r_invocations[-1].get("via") == "authorize_task"
            ),
            "last_fingerprint": (
                session.last_multi_r.final_fingerprint if session.last_multi_r else ""
            ),
            "satisfied": (
                session.last_multi_r.satisfied if session.last_multi_r else False
            ),
        }
        await session.close()
        return out


async def main_async(args: argparse.Namespace) -> dict:
    one = await run_workers(1, delay_s=args.delay)
    five = await run_workers(5, delay_s=args.delay)
    companion = await verify_companion_path()

    same_result = one["final_fingerprint"] == five["final_fingerprint"]
    zero_dups = (
        not one["dup_ids_in_log"]
        and not five["dup_ids_in_log"]
        and one["duplicate_dispatches"] == 0
        and five["duplicate_dispatches"] == 0
    )
    prep_shown = len(five["prep_while_executing"]) >= 1
    # Also accept speculative prepares whose timestamps fall inside a dispatch window
    if not prep_shown:
        windows = []
        cur = None
        for row in five["timeline"]:
            if row["kind"] == "dispatch_start":
                cur = row["t_s"]
            elif row["kind"] == "dispatch_end" and cur is not None:
                windows.append((cur, row["t_s"]))
                cur = None
        overlapped = []
        for row in five["timeline"]:
            if row["kind"] != "prepare":
                continue
            for a, b in windows:
                if a <= row["t_s"] <= b:
                    overlapped.append(row)
                    break
        if overlapped:
            five["prep_while_executing"] = [
                {
                    "t_s": r["t_s"],
                    "worker": r.get("worker"),
                    "effect": r.get("effect"),
                    "action_id": r.get("action_id"),
                    "speculative": r.get("speculative"),
                    "while_executor_busy": True,
                }
                for r in overlapped
            ]
            prep_shown = True

    accept = {
        "pass": same_result and zero_dups and prep_shown and companion["via_authorize_task"],
        "same_final_result": same_result,
        "zero_duplicate_dispatches": zero_dups,
        "prep_ahead_visible_in_timeline": prep_shown,
        "companion_submit_invokes_multi_r": companion["via_authorize_task"],
    }
    return {
        "generated_at": time.time(),
        "goal": GOAL,
        "executor": "sequential",
        "one_r": one,
        "five_r": five,
        "companion_path": companion,
        "accept": accept,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=0.12)
    parser.add_argument(
        "--out", type=Path, default=Path("eval_results/seq_timeline.json"),
    )
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Human-readable summary
    one, five = report["one_r"], report["five_r"]
    print("=== 1R vs 5R sequential executor ===")
    print(f"goal: {GOAL[:80]}...")
    print()
    print(f"{'metric':<32} {'1R':>12} {'5R':>12}")
    print(f"{'total_task_s':<32} {one['total_task_s']:12.4f} {five['total_task_s']:12.4f}")
    print(f"{'executor_idle_s':<32} {one['executor_idle_s']:12.4f} {five['executor_idle_s']:12.4f}")
    print(f"{'executor_busy_s':<32} {one['executor_busy_s']:12.4f} {five['executor_busy_s']:12.4f}")
    print(f"{'prep_overlap_s':<32} {one['prep_overlap_s']:12.4f} {five['prep_overlap_s']:12.4f}")
    lb, ld = five["l_before"], five["l_during"]
    print()
    print("L latency (sampled during 5R run):")
    print(f"  ttft_s before={lb['ttft_s']:.4f} during={ld['ttft_s']:.4f}")
    print(
        f"  matched_step_ms p50 before={lb['matched_step_ms_p50']:.4f} "
        f"during={ld['matched_step_ms_p50']:.4f}"
    )
    print()
    print(f"final_fingerprint 1R: {one['final_fingerprint']}")
    print(f"final_fingerprint 5R: {five['final_fingerprint']}")
    print(f"same_result={report['accept']['same_final_result']}  "
          f"zero_dups={report['accept']['zero_duplicate_dispatches']}")
    print()
    print("5R timeline (prep while earlier action executes):")
    for row in five["prep_while_executing"][:12]:
        print(
            f"  t={row['t_s']:.4f}s  {row['worker']} prepared {row['effect']} "
            f"({'speculative' if row.get('speculative') else 'ready'}) "
            f"while executor busy"
        )
    if not five["prep_while_executing"]:
        print("  (scanning full timeline for prepare∩dispatch windows)")
        for row in five["timeline"]:
            if row["kind"] in ("prepare", "dispatch_start", "dispatch_end"):
                print(f"  t={row['t_s']:.4f}s  {row['kind']}  { {k:v for k,v in row.items() if k not in ('t_s','kind')} }")
    print()
    print(
        f"companion submit_task → Multi-R: "
        f"{report['accept']['companion_submit_invokes_multi_r']}"
    )
    print(f"\nwrote {args.out}")
    print("SEQ_TIMELINE_PASS" if report["accept"]["pass"] else "SEQ_TIMELINE_NEEDS_WORK")
    if not report["accept"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
