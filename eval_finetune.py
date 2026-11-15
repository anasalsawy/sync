"""Select eval items → NDIF remote LoRA fine-tune → retest efficacy.

NDIF supports remote training via ``model.session(remote=True)`` with a LoRA
adapter living on the server (see nnsight remote-training pattern).

Usage:
  py -3.12 eval_finetune.py --ids mem_room_preference,env_status_saved --steps 20
  py -3.12 eval_finetune.py --ids plain_chat_hello --steps 10 --skip-retest
  py -3.12 eval_finetune.py --load adapters/speech_lora.pt --ids plain_chat_hello --skip-train
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from eval_bank import items_by_id
from eval_efficacy import _load_railway_key, run_item, summarize
from ndif import NDIFChannels, _speech_prompt


def _build_pairs(ids: list[str], tok) -> list[tuple[str, str]]:
    """(prompt_text, completion_text) supervised pairs from the bank gold."""
    pairs = []
    for item in items_by_id(ids):
        if not item.gold.strip():
            continue
        peer_notes = [f"{k}: {v[:500]}" for k, v in item.peers.items()]
        prompt = _speech_prompt(tok, f"user: {item.user}\nassistant:", peer_notes)
        pairs.append((prompt, item.gold.strip()))
    return pairs


class LoRA(torch.nn.Module):
    """Minimal LoRA matching nnsight remote-training docs."""

    def __init__(self, module, dim, rank, WA=None, WB=None):
        # Named super: shipped class is recompiled outside any class body.
        super(LoRA, self).__init__()
        self.module = module
        # Docs: use module.device. On NDIF meta Envoys that can be "meta",
        # which silently discards Parameter data — fall back to a real weight.
        device = module.device
        if str(device) == "meta":
            device = next(module.parameters()).device
        WA = torch.randn(dim, rank) if WA is None else WA
        WB = torch.zeros(rank, dim) if WB is None else WB
        self.WA = torch.nn.Parameter(WA.to(device), requires_grad=True)
        self.WB = torch.nn.Parameter(WB.to(device), requires_grad=True)

    def __call__(self):
        hidden = self.module.input
        delta = torch.matmul(torch.matmul(hidden.to(self.WA.dtype), self.WA), self.WB)
        self.module.output = delta.to(hidden.dtype) + self.module.output

    def parameters(self):
        return [self.WA, self.WB]


def train_speech_lora(channels: NDIFChannels, ids: list[str], *,
                      steps: int, rank: int, lr: float,
                      out_path: Path) -> dict:
    """Remote LoRA on speech host last MLP — nnsight remote-training shape."""
    speech = channels._speech_model()
    tok = speech.tokenizer
    pairs = _build_pairs(ids, tok)
    if not pairs:
        raise ValueError("No gold pairs for the selected ids")

    train_rows = []
    for prompt, gold in pairs:
        gold_ids = tok.encode(gold, add_special_tokens=False)
        if not gold_ids:
            continue
        train_rows.append((prompt, int(gold_ids[0])))

    module = speech.model.layers[-1].mlp
    with speech.scan(" "):
        dim = module.output.shape[-1].save()
    dim = int(dim)

    losses: list[float] = []
    wb_norms: list[float] = []
    with speech.session(remote=True) as session:
        # Adapter + optimizer MUST be constructed inside the session (server-side).
        adapter = LoRA(module, dim, rank)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
        for step in range(steps):
            prompt, target_id = train_rows[step % len(train_rows)]
            with speech.trace(prompt):
                adapter()
                logits = speech.lm_head.output
                target = torch.tensor([target_id], device=logits.device)
                loss = torch.nn.functional.cross_entropy(logits[:, -1], target)
                # nnsight 0.8+: interleaved backward (bare .backward() is vanilla).
                with loss.backward():
                    pass
            optimizer.step()
            optimizer.zero_grad()
            print(f"step {step} loss {loss.item():.4f} |WB| {adapter.WB.norm().item():.4f}")
            losses.append(loss.item())
            wb_norms.append(adapter.WB.norm().item())
        trained_WA = adapter.WA.detach().save()
        trained_WB = adapter.WB.detach().save()

    def _f(x):
        try:
            return float(x.item() if hasattr(x, "item") else x)
        except Exception:
            return None

    losses_f = [_f(x) for x in losses]
    wb_f = [_f(x) for x in wb_norms]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "WA": trained_WA,
            "WB": trained_WB,
            "rank": rank,
            "dim": dim,
            "target": "layers[-1].mlp",
            "speech_model": channels.speech_name,
            "ids": ids,
            "losses": losses_f,
            "wb_norms": wb_f,
        },
        out_path,
    )
    print(f"saved adapter {out_path}", flush=True)
    known = [x for x in wb_f if x is not None]
    moved = (max(known) - min(known)) > 1e-6 if known else False
    if not moved:
        print(
            "WARN: |WB| never moved — NDIF remote optimizer looks like a no-op "
            "(bit-identical loss). Adapter saved anyway; retest will not show learning.",
            flush=True,
        )
    return {
        "steps": steps,
        "n_pairs": len(train_rows),
        "losses": losses_f,
        "wb_norms": wb_f,
        "weights_moved": moved,
        "path": str(out_path),
    }


def retest(channels: NDIFChannels, ids: list[str], max_tokens: int) -> dict:
    results = []
    for item in items_by_id(ids):
        print(f"retest {item.id}", flush=True)
        results.append(run_item(channels, item, max_tokens))
    return summarize(results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune then retest Sync speech host")
    parser.add_argument("--ids", required=True, help="Comma-separated eval ids with gold")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.0)
    parser.add_argument("--out", default="adapters/speech_lora.pt")
    parser.add_argument("--load", default="", help="Existing adapter path")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-retest", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()

    _load_railway_key()
    if not os.environ.get("NDIF_API_KEY"):
        print("NDIF_API_KEY missing")
        return 2

    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    os.environ.setdefault("NDIF_CHANNEL_MODEL", "Qwen/Qwen3-8B")
    os.environ.setdefault("NDIF_SPEECH_MODEL", "Qwen/Qwen3-8B")
    channels = NDIFChannels()

    report: dict = {"ids": ids, "speech_model": channels.speech_name}
    adapter_path = Path(args.load) if args.load else Path(args.out)

    if not args.skip_train:
        print(
            f"fine-tune speech={channels.speech_name} steps={args.steps} "
            f"rank={args.rank} ids={ids}",
            flush=True,
        )
        try:
            report["train"] = train_speech_lora(
                channels, ids, steps=args.steps, rank=args.rank, lr=args.lr,
                out_path=Path(args.out),
            )
            adapter_path = Path(args.out)
        except Exception as exc:
            print(f"TRAIN_FAIL {type(exc).__name__}: {exc}", flush=True)
            report["train_error"] = f"{type(exc).__name__}: {exc}"

    if adapter_path.exists():
        try:
            channels.load_speech_lora(str(adapter_path))
            report["loaded"] = str(adapter_path)
            print(f"speech LoRA loaded: {adapter_path}", flush=True)
        except Exception as exc:
            print(f"LOAD_FAIL {type(exc).__name__}: {exc}", flush=True)
            report["load_error"] = f"{type(exc).__name__}: {exc}"
    elif args.load or args.skip_train:
        print(f"adapter missing: {adapter_path}", flush=True)
        report["load_error"] = f"missing {adapter_path}"

    if not args.skip_retest:
        print("retest after train/load...", flush=True)
        try:
            report["retest"] = retest(channels, ids, args.max_tokens)
            print(json.dumps(report["retest"], indent=2), flush=True)
        except Exception as exc:
            print(f"RETEST_FAIL {type(exc).__name__}: {exc}", flush=True)
            report["retest_error"] = f"{type(exc).__name__}: {exc}"

    Path("eval_finetune_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print("wrote eval_finetune_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
