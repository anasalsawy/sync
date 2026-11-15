"""Live Multi-R numbers: 3 R workers → one Hands engine → one shared task.

Hands is a single execution path. R workers claim/prepare ahead; dispatch is
serialized through one `sawii_hands.Hands` instance. Not 1 R : 1 Hands.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from companion_session import CompanionSession
from live_hands_host import HandsHost
from sync_session import SyncSession

PAR_GOAL = "Process independent items in parallel: alpha beta gamma delta"
SEQ_GOAL = "Pipeline: read the inbox, write a reply to outbox, then send it."
FIX_GOAL = (
    "Investigate the failing unit tests, edit src/app.py to fix them, "
    "then re-run tests."
)


async def _l_latency() -> dict:
    s = SyncSession(backend="sim", max_new_tokens=6, solo=True, isolate_peers=False)
    r = await s.turn("Status while Hands executes.")
    steps = r.matched_step_ms()
    await s.close()
    return {
        "ttft_s": r.ttft_s,
        "matched_step_ms_p50": sorted(steps)[len(steps) // 2] if steps else None,
    }


def _prep_while_busy(timeline) -> int:
    windows = []
    cur = None
    for ev in timeline:
        if ev.kind == "dispatch_start":
            cur = ev.at
        elif ev.kind == "dispatch_end" and cur is not None:
            windows.append((cur, ev.at))
            cur = None
    n = 0
    for ev in timeline:
        if ev.kind != "prepare":
            continue
        if ev.detail.get("while_executor_busy"):
            n += 1
            continue
        for a, b in windows:
            if a <= ev.at <= b:
                n += 1
                break
    return n


def _max_concurrent_dispatch(timeline) -> int:
    open_d = 0
    max_open = 0
    for ev in timeline:
        if ev.kind == "dispatch_start":
            open_d += 1
            max_open = max(max_open, open_d)
        elif ev.kind == "dispatch_end":
            open_d = max(0, open_d - 1)
    return max_open


async def run_task(host: HandsHost, goal: str, *, label: str, work: Path) -> dict:
    events: list[dict] = []
    session = CompanionSession(
        db_path=str(work / f"{label}_c.db"),
        episodic_path=str(work / f"{label}_e.db"),
        backend="sim",
        max_new_tokens=4,
        host=host,
        on_event=events.append,
    )
    l_before = await _l_latency()
    l_box: dict = {}

    async def _l() -> None:
        await asyncio.sleep(0.02)
        l_box["v"] = await _l_latency()

    l_task = asyncio.create_task(_l())
    t0 = time.perf_counter()
    task_id = await session.submit_task(goal, require_c=False)
    wall_s = time.perf_counter() - t0
    await l_task

    multi = session.last_multi_r
    launch = session.last_host_launch or {}
    timeline = multi.timeline if multi else []
    ids = [d["action_id"] for d in (multi.dispatch_log if multi else [])]
    hands_calls = list(host.history)

    engine_ids = {c.get("hands_engine_id") for c in hands_calls}
    workers_seen = sorted({c.get("worker_id") for c in hands_calls if c.get("worker_id")})

    written = sorted(
        str(p.relative_to(host.root))
        for p in Path(host.root).rglob("*")
        if p.is_file() and p.stat().st_mtime >= t0 - 1
    )

    out = {
        "label": label,
        "task_id": task_id,
        "goal": goal,
        "host_mode": launch.get("host_mode"),
        "host_backend": launch.get("host_backend"),
        "n_subagents": launch.get("n_subagents", 0),
        "r_workers_active": launch.get("r_workers_active") or (
            multi.metrics.n_workers if multi else 0
        ),
        "hands_engines": len(engine_ids),
        "hands_engine_ids": sorted(engine_ids),
        "workers_that_dispatched": workers_seen,
        "total_task_s": multi.metrics.total_task_s if multi else None,
        "wall_submit_s": wall_s,
        "executor_idle_s": multi.metrics.executor_idle_s if multi else None,
        "hands_busy_s": host.busy_s,
        "hands_idle_s": host.idle_s,
        "final_fingerprint": multi.final_fingerprint if multi else "",
        "satisfied": multi.satisfied if multi else False,
        "duplicate_dispatches": multi.metrics.duplicate_dispatches if multi else 0,
        "dup_ids_in_log": len(ids) != len(set(ids)),
        "prep_while_executing": _prep_while_busy(timeline),
        "max_concurrent_dispatches": _max_concurrent_dispatch(timeline),
        "hands_tool_calls": len(hands_calls),
        "hands_tools": [c.get("hands_tool") for c in hands_calls],
        "hands_elapsed_s": [round(float(c.get("elapsed_s") or 0), 4) for c in hands_calls],
        "hands_ok": all(c.get("ok", True) for c in hands_calls) if hands_calls else False,
        "files_touched": written[:40],
        "n_files_touched": len(written),
        "l_before": l_before,
        "l_during": l_box.get("v"),
        "via_authorize_task": bool(
            session.multi_r_invocations
            and session.multi_r_invocations[-1].get("via") == "authorize_task"
        ),
    }
    await session.close()
    return out


async def main_async(args: argparse.Namespace) -> dict:
    work = Path(tempfile.mkdtemp(prefix="live_hands_1hand_3r_"))

    seq = await run_task(
        HandsHost(root=work / "seq"), SEQ_GOAL, label="seq", work=work,
    )
    par = await run_task(
        HandsHost(root=work / "par"), PAR_GOAL, label="par", work=work,
    )
    fix = await run_task(
        HandsHost(root=work / "fix"), FIX_GOAL, label="fix", work=work,
    )

    accept = {
        "one_hand_per_task": (
            seq["hands_engines"] == 1
            and par["hands_engines"] == 1
            and fix["hands_engines"] == 1
            and seq["n_subagents"] == 0
            and seq["host_mode"] == "single_path"
        ),
        "three_r_active": (
            seq["r_workers_active"] == 3
            and par["r_workers_active"] == 3
            and fix["r_workers_active"] == 3
        ),
        "prep_ahead": seq["prep_while_executing"] >= 1,
        "serial_dispatch": (
            seq["max_concurrent_dispatches"] <= 1
            and par["max_concurrent_dispatches"] <= 1
            and fix["max_concurrent_dispatches"] <= 1
        ),
        "real_hands_tools": seq["hands_tool_calls"] >= 1 and seq["hands_ok"],
        "files_on_disk": seq["n_files_touched"] >= 1,
        "zero_dups": (
            seq["duplicate_dispatches"] == 0
            and par["duplicate_dispatches"] == 0
            and fix["duplicate_dispatches"] == 0
            and not seq["dup_ids_in_log"]
            and not par["dup_ids_in_log"]
            and not fix["dup_ids_in_log"]
        ),
        "satisfied": seq["satisfied"] and par["satisfied"] and fix["satisfied"],
    }
    accept["pass"] = all(accept.values())

    return {
        "generated_at": time.time(),
        "design": "3_R_workers_one_Hands_per_task",
        "work_root": str(work),
        "sequential_pipeline": seq,
        "parallel_items_serialized_on_one_hand": par,
        "fix_edit_test": fix,
        "accept": accept,
    }


def _row(title: str, r: dict) -> None:
    print(f"\n[{title}] {r['goal'][:72]}")
    print(
        f"  host={r['host_backend']} mode={r['host_mode']} "
        f"r_workers={r['r_workers_active']} hands_engines={r['hands_engines']} "
        f"engine={r['hands_engine_ids']}"
    )
    print(
        f"  total_task_s={r['total_task_s']:.4f}  wall_s={r['wall_submit_s']:.4f}  "
        f"hands_busy_s={r['hands_busy_s']:.4f}  idle_s={r['executor_idle_s']:.4f}"
    )
    print(
        f"  prep_while_exec={r['prep_while_executing']}  "
        f"max_concurrent_dispatch={r['max_concurrent_dispatches']}  "
        f"dups={r['duplicate_dispatches']}"
    )
    print(
        f"  hands_calls={r['hands_tool_calls']} tools={r['hands_tools']} "
        f"elapsed_s={r['hands_elapsed_s']}"
    )
    print(f"  workers_dispatched={r['workers_that_dispatched']}")
    print(f"  files={r['n_files_touched']} sample={r['files_touched'][:6]}")
    print(f"  fingerprint={r['final_fingerprint']}")
    print(
        f"  L ttft before={r['l_before']['ttft_s']:.4f} "
        f"during={r['l_during']['ttft_s']:.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("eval_results/live_hands.json"))
    args = p.parse_args()
    report = asyncio.run(main_async(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== LIVE: 3 R workers → 1 Hands → 1 shared task ===")
    print(f"work_root={report['work_root']}")
    _row("seq pipeline", report["sequential_pipeline"])
    _row("par items (still 1 hand)", report["parallel_items_serialized_on_one_hand"])
    _row("fix/edit/test", report["fix_edit_test"])
    print(f"\nACCEPT: {report['accept']}")
    print("LIVE_HANDS_PASS" if report["accept"]["pass"] else "LIVE_HANDS_FAIL")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
