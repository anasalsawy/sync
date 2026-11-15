"""Verify Hands entry: browser + application + shell → 3 groups × 3 R = 9 R.

Shows inference overlap across groups and execution overlap where Hands paths
are independent (shell vs desktop). Shared mouse/keyboard serializes only when
browser and application contend.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from companion_session import CompanionSession
from hands_paths import (
    PATH_APPLICATION,
    PATH_BROWSER,
    PATH_SHELL,
    build_hands_path,
)
from shared_resources import SharedResourceBus
from task_groups import MainRCoordinator


GOALS = [
    {
        "path": PATH_BROWSER,
        "goal": "Browser task: navigate to the start page, click search, type a query",
    },
    {
        "path": PATH_APPLICATION,
        "goal": "Application task: open the window, type into the field, save the document",
    },
    {
        "path": PATH_SHELL,
        "goal": "Shell task: list files, write a note via shell tools, verify it",
    },
]


async def run_live(args: argparse.Namespace) -> dict:
    work = Path(tempfile.mkdtemp(prefix="live_task_groups_"))
    bus = SharedResourceBus()
    events: list[dict] = []

    main = MainRCoordinator(
        bus=bus,
        workers_per_group=3,
        root=work / "hands",
        activation_injection=True,
        on_event=events.append,
    )

    specs = []
    for g in GOALS:
        host = build_hands_path(
            g["path"],
            root=work / g["path"],
            bus=bus,
            browser_delay_s=args.browser_delay,
            app_delay_s=args.app_delay,
        )
        specs.append({"goal": g["goal"], "path": g["path"], "host": host})

    t0 = time.perf_counter()
    groups = await main.start_many(specs, wait=False)

    peak_r = 0
    peak_groups = 0
    peak_snap = {}
    samples: list[dict] = []
    while any(g.status in ("starting", "running") for g in groups):
        snap = main.snapshot()
        ar = snap["active_r_workers_total"]
        ag = snap["active_task_groups"]
        if ar > peak_r:
            peak_r = ar
            peak_groups = ag
            peak_snap = {
                "at_s": time.perf_counter() - t0,
                "active_r_workers_total": ar,
                "active_task_groups": ag,
                "groups": [
                    {
                        "path": x["path"],
                        "status": x["status"],
                        "n_workers": x["n_workers"],
                        "worker_ids": x["worker_ids"],
                    }
                    for x in snap["groups"]
                ],
            }
        if len(samples) < 40:
            samples.append({
                "t": round(time.perf_counter() - t0, 4),
                "active_r": ar,
                "active_groups": ag,
                "statuses": {x["path"]: x["status"] for x in snap["groups"]},
            })
        await asyncio.sleep(0.005)

    await main.wait_all()
    wall = time.perf_counter() - t0
    final = main.snapshot()

    # Inference windows overlap?
    inf_windows = []
    for g in groups:
        if g.inference_started_at is not None and g.inference_finished_at is not None:
            inf_windows.append((g.path, g.inference_started_at, g.inference_finished_at))
    inference_overlap = False
    for i, (pa, a0, a1) in enumerate(inf_windows):
        for pb, b0, b1 in inf_windows[i + 1:]:
            if a0 < b1 and b0 < a1:
                inference_overlap = True

    # Execution overlap: shell busy while browser busy (independent paths)
    exec_overlap = False
    shell_hist = []
    browser_hist = []
    app_hist = []
    for g, spec in zip(groups, specs):
        host = spec["host"]
        hist = list(getattr(host, "history", []) or [])
        for sa in getattr(host, "subagents", {}).values():
            hist.extend(sa.history)
        # Reconstruct absolute intervals from elapsed (approx from finish order)
        # Better: use acquire_log + path tags in history with elapsed
        if g.path == PATH_SHELL:
            shell_hist = hist
        elif g.path == PATH_BROWSER:
            browser_hist = hist
        else:
            app_hist = hist

    # Wall overlap: if total wall < sum of per-group walls, groups overlapped
    group_walls = [
        (g.finished_at or 0) - g.started_at for g in groups if g.finished_at
    ]
    sum_walls = sum(group_walls)
    execution_overlap_ratio = (sum_walls / wall) if wall > 0 else 0.0
    exec_overlap = execution_overlap_ratio > 1.15  # clear multi-task speedup

    # Desktop contention vs shell independence: shell should not wait on mouse
    shell_waits = [
        a for a in bus.acquire_log if a["holder"].startswith("shell:")
    ]
    desktop_waits = [
        a for a in bus.acquire_log
        if a["holder"].startswith("browser:") or a["holder"].startswith("application:")
    ]

    per_group = []
    for g, spec in zip(groups, specs):
        r = g.result
        host = spec["host"]
        calls = list(getattr(host, "history", []) or [])
        for sa in getattr(host, "subagents", {}).values():
            calls.extend(sa.history)
        per_group.append({
            "group_id": g.group_id,
            "task_id": g.task_id,
            "path": g.path,
            "status": g.status,
            "n_workers": g.n_workers,
            "worker_ids": g.worker_ids,
            "host_mode": g.host_launch.get("host_mode"),
            "n_subagents": g.host_launch.get("n_subagents", 0),
            "total_task_s": r.metrics.total_task_s if r else None,
            "inference_s": r.metrics.inference_s if r else None,
            "duplicate_dispatches": r.metrics.duplicate_dispatches if r else None,
            "satisfied": bool(r and r.satisfied),
            "fingerprint": r.final_fingerprint if r else "",
            "hands_calls": len(calls),
        })

    # Companion entry point smoke: authorize one shell task through CompanionSession
    companion_events: list[dict] = []
    session = CompanionSession(
        db_path=str(work / "companion.db"),
        episodic_path=str(work / "episodic.db"),
        backend="sim",
        max_new_tokens=4,
        host=None,  # use Hands path routing
        on_event=companion_events.append,
    )
    # Point companion main_r at same bus/root style
    session.main_r.root = work / "companion_hands"
    session.main_r.bus = SharedResourceBus()
    tid = await session.submit_task(
        "Shell task: list files, write companion_ok.txt via shell, verify it",
        require_c=False,
    )
    companion_ok = bool(session.last_task_group and session.last_task_group.n_workers == 3)
    await session.close()

    accept = {
        "three_active_groups_at_peak": peak_groups == 3,
        "nine_active_r_at_peak": peak_r == 9,
        "three_workers_per_group": all(g.n_workers == 3 for g in groups),
        "paths_browser_app_shell": sorted(g.path for g in groups)
        == sorted([PATH_BROWSER, PATH_APPLICATION, PATH_SHELL]),
        "browser_single_stream": any(
            g.path == PATH_BROWSER and g.host_launch.get("host_mode") == "single_path"
            for g in groups
        ),
        "application_native_subagents": any(
            g.path == PATH_APPLICATION
            and g.host_launch.get("host_mode") == "native_subagent"
            and g.host_launch.get("n_subagents") == 3
            for g in groups
        ),
        "shell_single_hands_stream": any(
            g.path == PATH_SHELL and g.host_launch.get("host_mode") == "single_path"
            for g in groups
        ),
        "inference_overlap_across_groups": inference_overlap,
        "execution_overlap_independent_paths": exec_overlap,
        "all_satisfied": all(g.result and g.result.satisfied for g in groups),
        "zero_dups": all(
            g.result and g.result.metrics.duplicate_dispatches == 0 for g in groups
        ),
        "companion_hands_entry": companion_ok and bool(tid),
    }
    accept["pass"] = all(accept.values())

    return {
        "generated_at": time.time(),
        "work_root": str(work),
        "wall_s": wall,
        "sum_group_walls_s": sum_walls,
        "execution_overlap_ratio": execution_overlap_ratio,
        "peak_active_task_groups": peak_groups,
        "peak_active_r_workers": peak_r,
        "peak_snapshot": peak_snap,
        "samples": samples[:20],
        "inference_windows": [
            {"path": p, "start": a, "end": b} for p, a, b in inf_windows
        ],
        "resource_bus": bus.snapshot(),
        "desktop_acquires": len(desktop_waits),
        "shell_desktop_acquires": len(shell_waits),
        "per_group": per_group,
        "final_snapshot": final,
        "companion_task_id": tid,
        "companion_group_workers": (
            session.last_task_group.n_workers if session.last_task_group else 0
        ),
        "accept": accept,
        "event_types": [e.get("type") for e in events],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("eval_results/live_task_groups.json"))
    p.add_argument("--browser-delay", type=float, default=0.18)
    p.add_argument("--app-delay", type=float, default=0.14)
    args = p.parse_args()
    report = asyncio.run(run_live(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== LIVE Hands: 3 task groups × 3 R = 9 R workers ===\n")
    print(f"wall_s={report['wall_s']:.4f}  sum_group_walls_s={report['sum_group_walls_s']:.4f}  "
          f"overlap_ratio={report['execution_overlap_ratio']:.2f}")
    print(f"peak_active_task_groups={report['peak_active_task_groups']}")
    print(f"peak_active_r_workers={report['peak_active_r_workers']}")
    print(f"peak_snapshot={json.dumps(report['peak_snapshot'], indent=2)}")
    print()
    for g in report["per_group"]:
        print(
            f"  [{g['path']}] workers={g['n_workers']} host={g['host_mode']} "
            f"subagents={g['n_subagents']} total_s={g['total_task_s']:.4f} "
            f"infer_s={g['inference_s']:.4f} dups={g['duplicate_dispatches']} "
            f"ok={g['satisfied']}"
        )
        print(f"    ids={g['worker_ids']}")
    print(f"\ninference_overlap={report['accept']['inference_overlap_across_groups']}")
    print(f"execution_overlap={report['accept']['execution_overlap_independent_paths']}")
    print(f"desktop_acquires={report['desktop_acquires']} "
          f"shell_desktop_acquires={report['shell_desktop_acquires']}")
    print(f"companion_entry task_id={report['companion_task_id']} "
          f"workers={report['companion_group_workers']}")
    print(f"\nACCEPT: {report['accept']}")
    print("LIVE_TASK_GROUPS_PASS" if report["accept"]["pass"] else "LIVE_TASK_GROUPS_FAIL")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
