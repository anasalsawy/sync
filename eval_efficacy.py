"""Efficacy + latency harness: solo L vs connected group.

Usage:
  py -3.12 eval_efficacy.py
  py -3.12 eval_efficacy.py --ids mem_room_preference,env_status_saved
  py -3.12 eval_efficacy.py --aspects memory_grounding --repeats 3
  py -3.12 eval_efficacy.py --max-tokens 48 --out eval_results/retest.json

Solo = L alone (1-row generate). Connected = L+R+B+V+C mind-read generate.
`--repeats N` runs each item N times and reports latency mean/p50/p95.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from eval_bank import ASPECTS, BANK, EvalItem, items_by_aspect, items_by_id
from eval_scoring import connected_helps, score_item
from ndif import LobePrompt, NDIFChannels
from server import _clean_reply


def _load_railway_key() -> None:
    if os.environ.get("NDIF_API_KEY"):
        return
    for path in (
        Path.home() / ".nnsight" / "key",
        Path.home() / ".env",
        Path(__file__).with_name(".env"),
    ):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if path.name == "key":
            os.environ["NDIF_API_KEY"] = text.splitlines()[0].strip()
            return
        for line in text.splitlines():
            if line.startswith("NDIF_API_KEY="):
                os.environ["NDIF_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
                return
    try:
        out = subprocess.check_output(
            "railway variables --project sawii-sync --environment production --json",
            shell=True, text=True, stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        for k in ("NDIF_API_KEY", "NDIF_MODEL"):
            if k in data and k not in os.environ:
                os.environ[k] = str(data[k])
    except Exception:
        pass


@dataclass
class ConditionResult:
    mode: str
    text: str
    cleaned: str
    elapsed_s: float
    n_tokens: int
    n_exchanges: int
    score: dict
    latencies_s: list[float]


@dataclass
class ItemResult:
    id: str
    aspect: str
    user: str
    solo: ConditionResult
    connected: ConditionResult
    helps: bool | None


def _prompts_connected(item: EvalItem) -> list[LobePrompt]:
    return [
        LobePrompt("L", f"user: {item.user}\nassistant:"),
        LobePrompt("R", item.peers.get("R", "environment: idle")),
        LobePrompt("B", item.peers.get("B", "memory: none")),
        LobePrompt("V", item.peers.get("V", "vision: none")),
        LobePrompt("C", item.peers.get("C", "constraints: none")),
    ]


def _prompts_solo(item: EvalItem) -> list[LobePrompt]:
    # Ablation: L alone — no peer rows, so no mind-read channels.
    return [LobePrompt("L", f"user: {item.user}\nassistant:")]


def _latency_stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(xs)
    n = len(ordered)
    p95_i = min(n - 1, max(0, int(round(0.95 * (n - 1)))))
    return {
        "n": n,
        "mean": round(statistics.fmean(ordered), 3),
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[p95_i], 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def _pack(mode: str, gen, elapsed: float, item: EvalItem,
          latencies: list[float]) -> ConditionResult:
    cleaned = _clean_reply(gen.text)
    sc = score_item(cleaned, item)
    return ConditionResult(
        mode=mode,
        text=gen.text,
        cleaned=cleaned,
        elapsed_s=round(elapsed, 2),
        n_tokens=len(gen.token_ids),
        n_exchanges=len(getattr(gen.decision, "exchanges", ()) or ()),
        score={
            "pass": sc.pass_,
            "recall": round(sc.recall, 3),
            "clean": round(sc.clean, 3),
            "detail": sc.detail,
        },
        latencies_s=[round(x, 3) for x in latencies],
    )


def run_item(channels: NDIFChannels, item: EvalItem,
             max_tokens: int, repeats: int = 1) -> ItemResult:
    solo_lats: list[float] = []
    solo_gen = None
    for _ in range(max(1, repeats)):
        t0 = time.time()
        solo_gen = channels.generate_sync(
            _prompts_solo(item), max_new_tokens=max_tokens
        )
        solo_lats.append(time.time() - t0)
        gc.collect()
    solo = _pack("solo", solo_gen, solo_lats[-1], item, solo_lats)

    conn_lats: list[float] = []
    conn_gen = None
    for _ in range(max(1, repeats)):
        t1 = time.time()
        conn_gen = channels.generate_sync(
            _prompts_connected(item), max_new_tokens=max_tokens
        )
        conn_lats.append(time.time() - t1)
        gc.collect()
    connected = _pack("connected", conn_gen, conn_lats[-1], item, conn_lats)

    solo_sc = score_item(solo.cleaned, item)
    conn_sc = score_item(connected.cleaned, item)
    return ItemResult(
        id=item.id,
        aspect=item.aspect,
        user=item.user,
        solo=solo,
        connected=connected,
        helps=connected_helps(solo_sc, conn_sc, item),
    )


def summarize(results: list[ItemResult]) -> dict:
    n = len(results)
    solo_pass = sum(1 for r in results if r.solo.score["pass"])
    conn_pass = sum(1 for r in results if r.connected.score["pass"])
    needed = [r for r in results if next(
        (i.rubric.needs_peers for i in BANK if i.id == r.id), True
    )]
    helps = sum(1 for r in needed if r.helps)
    by_aspect: dict[str, dict] = {}
    for r in results:
        bucket = by_aspect.setdefault(
            r.aspect, {"n": 0, "solo": 0, "connected": 0, "helps": 0}
        )
        bucket["n"] += 1
        bucket["solo"] += int(r.solo.score["pass"])
        bucket["connected"] += int(r.connected.score["pass"])
        bucket["helps"] += int(bool(r.helps))

    solo_all = [t for r in results for t in r.solo.latencies_s]
    conn_all = [t for r in results for t in r.connected.latencies_s]
    return {
        "n": n,
        "solo_pass": solo_pass,
        "connected_pass": conn_pass,
        "peer_needed": len(needed),
        "connected_helps": helps,
        "by_aspect": by_aspect,
        "latency": {
            "solo": _latency_stats(solo_all),
            "connected": _latency_stats(conn_all),
            "overhead_mean_s": (
                round(
                    _latency_stats(conn_all)["mean"] - _latency_stats(solo_all)["mean"],
                    3,
                )
                if solo_all and conn_all else None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync solo vs connected efficacy")
    parser.add_argument("--ids", default="", help="Comma-separated eval ids")
    parser.add_argument("--aspects", default="", help="Comma-separated aspects")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeats per condition for latency stats")
    parser.add_argument("--out", default="eval_results/retest.json")
    parser.add_argument("--list", action="store_true", help="List bank and exit")
    parser.add_argument("--model", default="",
                        help="Override model (default NDIF_MODEL or Qwen3-8B)")
    args = parser.parse_args()

    if args.list:
        print("aspects:", ", ".join(ASPECTS))
        for item in BANK:
            print(f"  {item.id:28} [{item.aspect}] {item.user[:60]}")
        return 0

    _load_railway_key()
    if not os.environ.get("NDIF_API_KEY"):
        print("NDIF_API_KEY missing")
        return 2

    if args.ids:
        items = items_by_id([x.strip() for x in args.ids.split(",") if x.strip()])
    elif args.aspects:
        items = items_by_aspect([x.strip() for x in args.aspects.split(",") if x.strip()])
    else:
        items = list(BANK)

    model = args.model or os.environ.get("NDIF_MODEL") or "Qwen/Qwen3-8B"
    channels = NDIFChannels(model_name=model)
    print(
        f"model={model} items={len(items)} max_tokens={args.max_tokens} "
        f"repeats={args.repeats}",
        flush=True,
    )

    results: list[ItemResult] = []
    for item in items:
        print(f"\n=== {item.id} [{item.aspect}] ===", flush=True)
        print(f"Q: {item.user}", flush=True)
        try:
            result = run_item(channels, item, args.max_tokens, args.repeats)
        except Exception as exc:
            print(f"FAIL {type(exc).__name__}: {exc}", flush=True)
            continue
        results.append(result)
        print(
            f"SOLO ({result.solo.elapsed_s}s x{len(result.solo.latencies_s)}, "
            f"pass={result.solo.score['pass']}): {result.solo.cleaned[:160]!r}",
            flush=True,
        )
        print(
            f"CONN ({result.connected.elapsed_s}s x{len(result.connected.latencies_s)}, "
            f"pass={result.connected.score['pass']}, "
            f"exchanges={result.connected.n_exchanges}): "
            f"{result.connected.cleaned[:160]!r}",
            flush=True,
        )
        print(f"helps={result.helps}", flush=True)

    summary = summarize(results)
    payload = {
        "summary": summary,
        "model": model,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "results": [
            {
                "id": r.id,
                "aspect": r.aspect,
                "user": r.user,
                "helps": r.helps,
                "solo": asdict(r.solo),
                "connected": asdict(r.connected),
                "latency": {
                    "solo": _latency_stats(r.solo.latencies_s),
                    "connected": _latency_stats(r.connected.latencies_s),
                },
            }
            for r in results
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
