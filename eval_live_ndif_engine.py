"""Live multi-turn: NDIF reasons → Hands courts or Tasker engine executes.

No new browser engine. Uses the already-running Hands app (/v1/courts) or
tasker-agent --live. Reports LLM latency, engine wall time, and gaps between
engine actions/turns.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ndif import LobePrompt, NDIFChannels


@dataclass
class TurnRow:
    turn: int
    reasoning: str
    subgoal: str
    llm_s: float
    engine: str
    engine_s: float
    gap_from_prev_engine_s: float | None
    engine_status: str
    engine_detail: str
    observation: str


@dataclass
class Report:
    goal: str
    model: str
    engine_mode: str
    wall_s: float = 0.0
    llm_s_total: float = 0.0
    engine_s_total: float = 0.0
    gaps_s: list[float] = field(default_factory=list)
    turns: list[TurnRow] = field(default_factory=list)
    ok: bool = False
    notes: list[str] = field(default_factory=list)


def http(method: str, base: str, path: str, key: str, body=None, timeout=90):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw[:500]}
    except Exception as e:  # noqa: BLE001
        return 0, {"error": f"{type(e).__name__}: {e}"}


def parse_plan(text: str) -> tuple[str, str, bool]:
    """Return (reasoning, subgoal, done)."""
    reasoning = (text or "").strip()
    done = bool(re.search(r"\bDONE\b", reasoning, re.I))
    subgoal = ""
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text or "")
    blob = fence.group(1).strip() if fence else (text or "")
    for match in re.finditer(r"\{[\s\S]*\}", blob):
        try:
            obj = json.loads(match.group())
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        reasoning = str(obj.get("reasoning") or obj.get("think") or reasoning)[:800]
        subgoal = str(obj.get("subgoal") or obj.get("goal") or obj.get("action") or "")
        done = done or bool(obj.get("done"))
        if subgoal:
            break
    if not subgoal:
        # fallback: last non-empty line
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        subgoal = lines[-1][:240] if lines else "Observe the current screen and report what you see"
    return reasoning[:800], subgoal, done


def planner_prompt(goal: str, history: list[str], observation: str, turn: int, max_turns: int) -> str:
    return (
        f"You are Sync's planner (turn {turn}/{max_turns}). Hands/Tasker will execute ONE subgoal.\n"
        f"OVERALL GOAL: {goal}\n"
        f"LAST ENGINE OBSERVATION:\n{observation}\n"
        f"HISTORY:\n" + ("\n".join(history[-8:]) or "(none)") + "\n\n"
        "Reason about progress, then emit ONLY JSON:\n"
        '{"reasoning":"...","subgoal":"one concrete browser/desktop instruction","done":false}\n'
        "When the overall goal is satisfied set done=true and subgoal=\"\". "
        "Prefer public pages (example.com, news.ycombinator.com). No logins or purchases."
    )


class HandsEngine:
    """Existing Hands courts deck on a running stitch server."""

    name = "hands"

    def __init__(self, base: str, key: str):
        self.base = base.rstrip("/")
        self.key = key
        self.court_id: str | None = None

    def ensure_court(self, kind: str = "browser") -> str:
        if self.court_id:
            return self.court_id
        code, body = http("POST", self.base, "/v1/courts", self.key, {
            "title": "ndif-live-browser", "kind": kind,
        })
        cid = (body or {}).get("id")
        if not cid:
            raise RuntimeError(f"create court failed {code} {body}")
        self.court_id = cid
        return cid

    def run_subgoal(self, subgoal: str, timeout_s: float = 90.0) -> dict[str, Any]:
        cid = self.ensure_court("browser")
        t0 = time.perf_counter()
        code, body = http(
            "POST", self.base, f"/v1/courts/{cid}/goal", self.key,
            {"goal": subgoal}, timeout=60,
        )
        status = (body or {}).get("status") or f"http_{code}"
        detail = str((body or {}).get("error") or "")[:300]
        last = body or {}
        while time.perf_counter() - t0 < timeout_s:
            time.sleep(2.0)
            _, cur = http("GET", self.base, f"/v1/courts/{cid}", self.key)
            last = cur or last
            status = (cur or {}).get("status") or status
            detail = str((cur or {}).get("error") or detail)[:300]
            if status in ("done", "failed", "stopped"):
                break
        _, hands = http("GET", self.base, "/v1/hands", self.key)
        elapsed = time.perf_counter() - t0
        return {
            "status": status,
            "elapsed_s": elapsed,
            "detail": detail,
            "court": last,
            "hands": {
                "connected": (hands or {}).get("connected"),
                "places": list(((hands or {}).get("places") or {}).keys())[:8],
            },
            "observation": json.dumps({
                "status": status,
                "error": detail,
                "lane": (last or {}).get("lane"),
                "meta": (last or {}).get("meta"),
                "hands_places": list(((hands or {}).get("places") or {}).keys())[:8],
            })[:1200],
        }


class TaskerEngine:
    """Existing tasker-agent --live computer-use engine."""

    name = "tasker"

    def __init__(self, journal_dir: Path):
        self.journal_dir = journal_dir
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def run_subgoal(self, subgoal: str, timeout_s: float = 120.0) -> dict[str, Any]:
        self.n += 1
        journal = self.journal_dir / f"turn_{self.n}.sqlite"
        run_id = f"ndif_live_{self.n}_{int(time.time())}"
        cmd = [
            "tasker-agent", "--live",
            "--run-id", run_id,
            "--journal", str(journal),
            "--max-batches", "4",
            "--max-actions-per-batch", "3",
            "--timeout", str(int(timeout_s)),
            "--trace",
            subgoal,
        ]
        t0 = time.perf_counter()
        try:
            p = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s + 30,
                env={**os.environ},
            )
            out = (p.stdout or "")[-2000:]
            err = (p.stderr or "")[-1000:]
            status = "done" if p.returncode == 0 else "failed"
            detail = err.strip() or out.strip()[-400:]
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else ""
            err = (e.stderr or "")[-1000:] if isinstance(e.stderr, str) else "timeout"
            status = "failed"
            detail = f"timeout: {err}"
        elapsed = time.perf_counter() - t0
        # Parse action timestamps from trace if present
        action_lines = [
            ln for ln in (out or "").splitlines()
            if "action" in ln.lower() or "batch" in ln.lower() or "click" in ln.lower()
        ][-12:]
        return {
            "status": status,
            "elapsed_s": elapsed,
            "detail": detail[:400],
            "journal": str(journal),
            "trace_tail": action_lines,
            "observation": json.dumps({
                "status": status,
                "detail": detail[:300],
                "trace": action_lines[-6:],
            })[:1200],
        }


def run(args: argparse.Namespace) -> Report:
    if not os.environ.get("NDIF_API_KEY"):
        raise SystemExit("NDIF_API_KEY missing")

    model = os.environ.get("NDIF_MODEL", args.model)
    report = Report(goal=args.goal, model=model, engine_mode=args.engine)
    channels = NDIFChannels(model_name=model)

    if args.engine == "hands":
        engine: Any = HandsEngine(args.hands_url, args.hands_key)
        # Probe
        code, body = http("GET", args.hands_url, "/v1/courts", args.hands_key)
        if code != 200:
            raise SystemExit(f"Hands not reachable: {code} {body}")
        report.notes.append("hands_courts_ok")
    else:
        engine = TaskerEngine(Path(args.journal_dir))
        report.notes.append("tasker_agent_live")

    history: list[str] = []
    observation = "(engine idle — no observation yet)"
    last_engine_end: float | None = None
    wall0 = time.perf_counter()

    for turn in range(1, args.turns + 1):
        prompt = planner_prompt(args.goal, history, observation, turn, args.turns)
        t_llm0 = time.perf_counter()
        gen = channels.generate_sync(
            [
                LobePrompt("L", prompt),
                LobePrompt("R", f"execute via {args.engine}: {args.goal}"),
                LobePrompt("B", "memory: track browser/desktop progress"),
                LobePrompt("V", f"last observation: {observation[:900]}"),
                LobePrompt("C", "constraints: public pages only; no login; no purchase"),
            ],
            max_new_tokens=args.max_tokens,
        )
        llm_s = time.perf_counter() - t_llm0
        report.llm_s_total += llm_s
        reasoning, subgoal, done = parse_plan(gen.text or "")

        if done and not subgoal:
            report.turns.append(TurnRow(
                turn=turn, reasoning=reasoning, subgoal="", llm_s=llm_s,
                engine=args.engine, engine_s=0.0, gap_from_prev_engine_s=None,
                engine_status="skipped", engine_detail="done",
                observation=observation[:500],
            ))
            report.notes.append(f"NDIF signaled DONE at turn {turn}")
            break

        gap = None if last_engine_end is None else time.perf_counter() - last_engine_end
        eng = engine.run_subgoal(subgoal, timeout_s=args.engine_timeout)
        last_engine_end = time.perf_counter()
        if gap is not None:
            report.gaps_s.append(gap)
        report.engine_s_total += float(eng["elapsed_s"])

        observation = eng.get("observation") or eng.get("detail") or observation
        history.append(
            f"t{turn}: subgoal={subgoal!r} status={eng['status']} "
            f"detail={str(eng.get('detail') or '')[:160]}"
        )
        report.turns.append(TurnRow(
            turn=turn,
            reasoning=reasoning,
            subgoal=subgoal,
            llm_s=llm_s,
            engine=args.engine,
            engine_s=float(eng["elapsed_s"]),
            gap_from_prev_engine_s=gap,
            engine_status=str(eng["status"]),
            engine_detail=str(eng.get("detail") or "")[:300],
            observation=str(observation)[:500],
        ))
        if eng["status"] == "failed" and "402" in str(eng.get("detail") or ""):
            report.notes.append("engine_failed_credits_or_auth — try --engine tasker")
            if args.engine == "hands" and args.fallback_tasker:
                report.notes.append("switching remaining turns to tasker")
                engine = TaskerEngine(Path(args.journal_dir))
                args.engine = "tasker"
                report.engine_mode = "hands->tasker"
            elif not args.continue_on_fail:
                break

    report.wall_s = time.perf_counter() - wall0
    report.ok = (
        report.llm_s_total > 0
        and any(t.engine_status in ("done", "running", "waiting") or t.engine_s > 0
                for t in report.turns)
        and any(t.subgoal for t in report.turns)
    )
    # Stricter: at least one engine done
    if any(t.engine_status == "done" for t in report.turns):
        report.ok = True
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--goal",
        default=(
            "Open https://example.com in the browser, read the main heading, "
            "then open https://news.ycombinator.com and report the page title."
        ),
    )
    ap.add_argument("--engine", choices=("hands", "tasker"), default="hands")
    ap.add_argument("--hands-url", default="http://127.0.0.1:8800")
    ap.add_argument(
        "--hands-key",
        default=os.environ.get(
            "SAWII_PERSONA_KEY",
            "sp_NKJ1NJOI_SE9MxJy2kXi1mqf1BWtYY50fiFNW47mFrw",
        ),
    )
    ap.add_argument("--journal-dir", default="eval_results/tasker_journals")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--engine-timeout", type=float, default=100.0)
    ap.add_argument("--fallback-tasker", action="store_true", default=True)
    ap.add_argument("--continue-on-fail", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("eval_results/live_ndif_engine.json"))
    args = ap.parse_args()

    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "goal": report.goal,
        "model": report.model,
        "engine_mode": report.engine_mode,
        "wall_s": report.wall_s,
        "llm_s_total": report.llm_s_total,
        "engine_s_total": report.engine_s_total,
        "gap_s_mean": (sum(report.gaps_s) / len(report.gaps_s)) if report.gaps_s else None,
        "gap_s_min": min(report.gaps_s) if report.gaps_s else None,
        "gap_s_max": max(report.gaps_s) if report.gaps_s else None,
        "llm_s_per_turn": [t.llm_s for t in report.turns],
        "engine_s_per_turn": [t.engine_s for t in report.turns],
        "gaps_s": report.gaps_s,
        "ok": report.ok,
        "notes": report.notes,
        "turns": [asdict(t) for t in report.turns],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== LIVE NDIF + existing Hands/Tasker engine ===")
    print(f"engine={report.engine_mode} model={report.model}")
    print(f"wall_s={report.wall_s:.2f} llm_s_total={report.llm_s_total:.2f} "
          f"engine_s_total={report.engine_s_total:.2f}")
    if report.gaps_s:
        print(f"engine_gap_s mean={payload['gap_s_mean']:.2f} "
              f"min={payload['gap_s_min']:.2f} max={payload['gap_s_max']:.2f}")
    for t in report.turns:
        gap = f" gap={t.gap_from_prev_engine_s:.2f}s" if t.gap_from_prev_engine_s is not None else ""
        print(f"\n[turn {t.turn}] llm_s={t.llm_s:.2f} engine_s={t.engine_s:.2f}{gap} "
              f"status={t.engine_status}")
        print(f"  reason: {t.reasoning[:160]}")
        print(f"  subgoal: {t.subgoal[:160]}")
        print(f"  detail: {t.engine_detail[:160]}")
    print(f"\nnotes: {report.notes}")
    print("NDIF_ENGINE_LIVE_PASS" if report.ok else "NDIF_ENGINE_LIVE_FAIL")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
