"""Live NDIF + real browser multi-turn: LLM reasoning → Playwright actions.

Measures:
  - LLM turn latency (NDIF generate_sync)
  - Per-action browser elapsed
  - Gap between consecutive actions (action-to-action)
  - End-to-end wall for the full multi-turn task
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ndif import LobePrompt, NDIFChannels
from host_bridge import push_playwright_to_v, register_playwright_page
from action_runtime import ActionProposal, execute_action
from live_screen import LIVE_SCREEN


@dataclass
class ActionTiming:
    turn: int
    index: int
    action_type: str
    target: str
    success: bool
    llm_s: float | None
    action_s: float
    gap_from_prev_action_s: float | None
    error: str = ""
    output: str = ""


@dataclass
class TurnTiming:
    turn: int
    llm_s: float
    reasoning_excerpt: str
    actions_planned: int
    actions_ran: int
    observation: str
    reply: str


@dataclass
class Report:
    goal: str
    model: str
    wall_s: float
    turns: list[TurnTiming] = field(default_factory=list)
    actions: list[ActionTiming] = field(default_factory=list)
    llm_s_total: float = 0.0
    action_s_total: float = 0.0
    gaps_s: list[float] = field(default_factory=list)
    ok: bool = False
    final_url: str = ""
    final_title: str = ""
    notes: list[str] = field(default_factory=list)


def _load_key() -> None:
    if os.environ.get("NDIF_API_KEY"):
        return
    for path in (Path.home() / ".nnsight" / "key", Path.home() / ".env", Path(".env")):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.name == "key":
            os.environ["NDIF_API_KEY"] = text.splitlines()[0].strip()
            return
        for line in text.splitlines():
            if line.startswith("NDIF_API_KEY="):
                os.environ["NDIF_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
                return


def _parse_actions(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract reasoning + JSON action list from model text."""
    reasoning = text.strip()
    actions: list[dict[str, Any]] = []
    # Prefer ```json ... ``` or a bare JSON array/object with "actions"
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    blob = fence.group(1).strip() if fence else text
    # Array of actions
    for match in re.finditer(r"\[[\s\S]*?\]", blob):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                actions = obj
                break
        except json.JSONDecodeError:
            continue
    if not actions:
        # Object with actions key
        for match in re.finditer(r"\{[\s\S]*\}", blob):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict) and isinstance(obj.get("actions"), list):
                    actions = obj["actions"]
                    reasoning = str(obj.get("reasoning") or obj.get("think") or reasoning)
                    break
                if isinstance(obj, dict) and (obj.get("action") or obj.get("type")):
                    actions = [obj]
                    break
            except json.JSONDecodeError:
                continue
    # Normalize keys — models sometimes emit the enum list as the action string
    allowed = {"navigate", "click", "fill", "scroll", "observe", "key", "goto", "open"}
    norm: list[dict[str, Any]] = []
    for a in actions[:6]:
        at = (a.get("action") or a.get("type") or a.get("action_type") or "").lower().strip()
        if "|" in at or at not in allowed:
            # pick first allowed verb mentioned
            hit = next((v for v in ("click", "navigate", "fill", "scroll", "observe", "key") if v in at), "")
            at = hit
        target = a.get("target") or a.get("selector") or a.get("url") or a.get("query") or ""
        if not at or not target:
            continue
        if at in ("goto", "open"):
            at = "navigate"
        norm.append({
            "action": at,
            "target": str(target),
            "value": a.get("value") or a.get("payload") or a.get("text"),
            "priority": float(a.get("priority", 1.0)),
        })
    return reasoning[:800], norm


async def _page_observation(page) -> str:
    title = await page.title()
    url = page.url
    # Compact interactive snapshot
    links = await page.eval_on_selector_all(
        "a[href]",
        "els => els.slice(0,12).map(e => ({text:(e.innerText||'').trim().slice(0,60), href:e.href}))",
    )
    inputs = await page.eval_on_selector_all(
        "input,textarea,button,select",
        "els => els.slice(0,12).map(e => ({tag:e.tagName.toLowerCase(), type:e.type||'', "
        "name:e.name||'', id:e.id||'', text:(e.innerText||e.value||'').trim().slice(0,40)}))",
    )
    return json.dumps({
        "url": url,
        "title": title,
        "links": links,
        "controls": inputs,
    }, ensure_ascii=False)[:3500]


def _planner_prompt(goal: str, history: list[str], observation: str, turn: int, max_turns: int) -> str:
    return (
        f"You are Sync's browser operator (turn {turn}/{max_turns}).\n"
        f"GOAL: {goal}\n"
        f"CURRENT PAGE OBSERVATION (JSON):\n{observation}\n"
        f"HISTORY:\n" + ("\n".join(history[-6:]) or "(none)") + "\n\n"
        "Reason briefly about the next step, then emit ONLY a JSON object:\n"
        '{"reasoning":"...","actions":[{"action":"navigate|click|fill|scroll|observe|key","target":"...","value":"..."}]}\n'
        "Rules: 1-3 actions max this turn. Prefer real CSS selectors or absolute URLs from the observation. "
        "Hacker News URL is exactly https://news.ycombinator.com/ — never invent domains. "
        "V already holds live page state shared into worker lanes — do NOT emit observe/screenshot "
        "unless the observation above is empty. Act on the next real action immediately. "
        "When the goal is satisfied, emit actions:[] and say DONE in reasoning."
    )


async def run(args: argparse.Namespace) -> Report:
    _load_key()
    if not os.environ.get("NDIF_API_KEY"):
        raise SystemExit("NDIF_API_KEY missing")

    from playwright.async_api import async_playwright

    model = os.environ.get("NDIF_MODEL", args.model)
    report = Report(goal=args.goal, model=model, wall_s=0.0)
    wall0 = time.perf_counter()

    channels = NDIFChannels(model_name=model)
    history: list[str] = []
    last_action_end: float | None = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.headed)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()
        register_playwright_page(page)

        # Seed on a real public page; V owns live state shared into worker lanes
        await page.goto(args.start_url, wait_until="domcontentloaded", timeout=30000)
        await push_playwright_to_v(page)

        for turn in range(1, args.turns + 1):
            # Prefer V's live bus — no extra screenshot round-trip for workers
            if not LIVE_SCREEN.has_live():
                await push_playwright_to_v(page)
            obs = LIVE_SCREEN.worker_context()
            # Keep a compact JSON observation for the planner as well
            page_json = await _page_observation(page)
            # Refresh V from structured page JSON so R lanes stay current
            try:
                data = json.loads(page_json)
                LIVE_SCREEN.push(
                    source="browser",
                    url=data.get("url", ""),
                    title=data.get("title", ""),
                    summary=f"{data.get('title', '')} @ {data.get('url', '')}",
                    links=list(data.get("links") or []),
                    controls=list(data.get("controls") or []),
                )
            except json.JSONDecodeError:
                await push_playwright_to_v(page)
            obs = LIVE_SCREEN.worker_context()
            prompt = _planner_prompt(args.goal, history, page_json, turn, args.turns)

            t_llm0 = time.perf_counter()
            gen = await asyncio.to_thread(
                channels.generate_sync,
                [
                    LobePrompt("L", prompt),
                    LobePrompt(
                        "R",
                        f"browser task: {args.goal}\n"
                        f"{LIVE_SCREEN.worker_context()}\n"
                        f"skip_observe: V live already shared — act next",
                    ),
                    LobePrompt("B", "memory: track progress toward the goal"),
                    LobePrompt("V", f"live screen (owner):\n{LIVE_SCREEN.worker_context()}"),
                    LobePrompt("C", "constraints: only safe public pages; no login; no purchases"),
                ],
                args.max_tokens,
            )
            llm_s = time.perf_counter() - t_llm0
            report.llm_s_total += llm_s
            reply = (gen.text or "").strip()
            reasoning, planned = _parse_actions(reply)

            force_all = bool(getattr(args, "force_turns", False))

            if "DONE" in reasoning.upper() and not planned:
                report.turns.append(TurnTiming(
                    turn=turn, llm_s=llm_s, reasoning_excerpt=reasoning[:400],
                    actions_planned=0, actions_ran=0, observation=obs[:500], reply=reply[:600],
                ))
                report.notes.append(f"model signaled DONE at turn {turn}")
                if not force_all:
                    break
                # Keep going for latency measurement — nudge next turn
                history.append(f"t{turn}: model said DONE early; continue forced turns")
                continue

            # V live → never force observe/screenshot; leave empty so next turn acts
            if not planned:
                report.notes.append(
                    f"turn {turn}: no JSON actions — V live shared, skipping observe"
                )
                LIVE_SCREEN.note_observe_skipped()
                report.turns.append(TurnTiming(
                    turn=turn, llm_s=llm_s, reasoning_excerpt=reasoning[:400],
                    actions_planned=0, actions_ran=0, observation=obs[:500], reply=reply[:600],
                ))
                history.append(f"t{turn}: (no actions; V live seq={LIVE_SCREEN.seq})")
                continue

            # Drop observe when V already shared — workers act immediately
            filtered = []
            for a in planned:
                if a["action"] in ("observe", "screenshot", "snapshot") and LIVE_SCREEN.should_skip_observe():
                    LIVE_SCREEN.note_observe_skipped()
                    report.notes.append(
                        f"turn {turn}: skipped {a['action']} — V live seq={LIVE_SCREEN.seq}"
                    )
                    continue
                filtered.append(a)
            planned = filtered
            if not planned:
                report.turns.append(TurnTiming(
                    turn=turn, llm_s=llm_s, reasoning_excerpt=reasoning[:400],
                    actions_planned=0, actions_ran=0, observation=obs[:500], reply=reply[:600],
                ))
                continue

            ran = 0
            for i, a in enumerate(planned):
                prop = ActionProposal(
                    lobe="R",
                    action_type=a["action"],
                    target=a["target"],
                    payload=a.get("value"),
                    priority=float(a.get("priority", 1.0)),
                )
                # Map navigate aliases
                if prop.action_type in ("goto", "open"):
                    prop.action_type = "navigate"

                now = time.perf_counter()
                gap = None if last_action_end is None else now - last_action_end
                result = await execute_action(prop)
                action_end = time.perf_counter()
                if gap is not None:
                    report.gaps_s.append(gap)
                report.action_s_total += result.elapsed_s
                report.actions.append(ActionTiming(
                    turn=turn,
                    index=i,
                    action_type=prop.action_type,
                    target=prop.target[:120],
                    success=result.success,
                    llm_s=llm_s if i == 0 else None,
                    action_s=result.elapsed_s,
                    gap_from_prev_action_s=gap,
                    error=result.error or "",
                    output=str(result.output or "")[:200],
                ))
                last_action_end = action_end
                ran += 1
                if not result.success:
                    report.notes.append(
                        f"turn {turn} action {i} failed: {prop.action_type} {prop.target}: {result.error}"
                    )

            new_obs = await _page_observation(page)
            history.append(
                f"t{turn}: reasoning={reasoning[:200]} actions={planned} "
                f"url={page.url} title={(await page.title())[:80]}"
            )
            report.turns.append(TurnTiming(
                turn=turn, llm_s=llm_s, reasoning_excerpt=reasoning[:400],
                actions_planned=len(planned), actions_ran=ran,
                observation=new_obs[:500], reply=reply[:600],
            ))

            if args.goal_check == "hn_top" and not getattr(args, "force_turns", False):
                # Require leaving the HN index (clicked into a story)
                if "news.ycombinator.com" in page.url and "/item?" in page.url:
                    report.notes.append("goal_check hn_top story opened")
                    break
                if page.url and "news.ycombinator.com" not in page.url and page.url.startswith("http"):
                    report.notes.append(f"goal_check left HN to {page.url}")
                    break

        report.final_url = page.url
        report.final_title = await page.title()
        await browser.close()

    report.wall_s = time.perf_counter() - wall0
    left_hn_index = bool(
        report.final_url
        and (
            ("news.ycombinator.com" in report.final_url and "/item?" in report.final_url)
            or (
                report.final_url.startswith("http")
                and "news.ycombinator.com" not in report.final_url
            )
        )
    )
    if args.goal_check == "hn_top":
        report.ok = (
            report.llm_s_total > 0
            and any(a.success for a in report.actions)
            and left_hn_index
        )
        if not left_hn_index:
            report.notes.append("goal_check fail: still on HN index (story not opened)")
    else:
        report.ok = (
            len(report.actions) >= 1
            and report.llm_s_total > 0
            and any(a.success for a in report.actions)
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--goal",
        default=(
            "Open Hacker News, identify the top story title from the page, "
            "then open that story link."
        ),
    )
    ap.add_argument("--start-url", default="https://example.com/")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--goal-check", default="hn_top")
    ap.add_argument(
        "--force-turns",
        action="store_true",
        help="Run all --turns even if goal is met / model says DONE (for gap timing)",
    )
    ap.add_argument("--out", type=Path, default=Path("eval_results/live_ndif_browser.json"))
    args = ap.parse_args()

    report = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "goal": report.goal,
        "model": report.model,
        "wall_s": report.wall_s,
        "llm_s_total": report.llm_s_total,
        "action_s_total": report.action_s_total,
        "n_turns": len(report.turns),
        "n_actions": len(report.actions),
        "gap_s_mean": (sum(report.gaps_s) / len(report.gaps_s)) if report.gaps_s else None,
        "gap_s_min": min(report.gaps_s) if report.gaps_s else None,
        "gap_s_max": max(report.gaps_s) if report.gaps_s else None,
        "llm_s_per_turn": [t.llm_s for t in report.turns],
        "action_s": [a.action_s for a in report.actions],
        "gaps_s": report.gaps_s,
        "final_url": report.final_url,
        "final_title": report.final_title,
        "ok": report.ok,
        "notes": report.notes,
        "turns": [asdict(t) for t in report.turns],
        "actions": [asdict(a) for a in report.actions],
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== LIVE NDIF + browser multi-turn ===")
    print(f"goal: {report.goal}")
    print(f"model: {report.model}")
    print(f"wall_s={report.wall_s:.2f}  llm_s_total={report.llm_s_total:.2f}  "
          f"action_s_total={report.action_s_total:.3f}")
    print(f"turns={len(report.turns)} actions={len(report.actions)}")
    if report.gaps_s:
        print(f"action_gap_s mean={payload['gap_s_mean']:.3f} "
              f"min={payload['gap_s_min']:.3f} max={payload['gap_s_max']:.3f}")
    for t in report.turns:
        print(f"\n[turn {t.turn}] llm_s={t.llm_s:.2f} planned={t.actions_planned} ran={t.actions_ran}")
        print(f"  reason: {t.reasoning_excerpt[:180]}")
    for a in report.actions:
        gap = f" gap={a.gap_from_prev_action_s:.3f}s" if a.gap_from_prev_action_s is not None else ""
        print(
            f"  act t{a.turn}.{a.index} {a.action_type}({a.target[:50]}) "
            f"ok={a.success} action_s={a.action_s:.3f}{gap}"
        )
    print(f"\nfinal: {report.final_title!r} {report.final_url}")
    print(f"notes: {report.notes}")
    print("NDIF_BROWSER_LIVE_PASS" if report.ok else "NDIF_BROWSER_LIVE_FAIL")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
