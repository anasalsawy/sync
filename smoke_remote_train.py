"""Minimal NDIF remote-training smoke: can |bias| move?"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import torch


def _load_key() -> None:
    if os.environ.get("NDIF_API_KEY"):
        return
    try:
        out = subprocess.check_output(
            "railway variables --project sawii-sync --environment production --json",
            shell=True, text=True, stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
        if "NDIF_API_KEY" in data:
            os.environ["NDIF_API_KEY"] = str(data["NDIF_API_KEY"])
    except Exception:
        pass


def main() -> int:
    _load_key()
    if not os.environ.get("NDIF_API_KEY"):
        print("NDIF_API_KEY missing")
        return 2

    import nnsight
    from nnsight import TransformersModel
    from transformers import AutoTokenizer

    nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
    name = os.environ.get("NDIF_SPEECH_MODEL", "Qwen/Qwen3-8B")
    tok = AutoTokenizer.from_pretrained(name)
    model = TransformersModel(name, tokenizer=tok, device_map="meta", task="text-generation")
    prompt = "The capital of France is"
    target_id = tok.encode(" Paris", add_special_tokens=False)[0]
    print(f"model={name} target_id={target_id} nnsight={nnsight.__version__}", flush=True)

    with model.session(remote=True):
        with model.trace(prompt):
            device = model.lm_head.output.device
            vocab = model.lm_head.output.shape[-1]
        bias = torch.nn.Parameter(torch.zeros(vocab, device=device), requires_grad=True)
        opt = torch.optim.SGD([bias], lr=50.0)
        for step in range(5):
            with model.trace(prompt):
                model.lm_head.output = model.lm_head.output + bias
                logits = model.lm_head.output[:, -1]
                loss = torch.nn.functional.cross_entropy(
                    logits, torch.tensor([target_id], device=logits.device)
                )
                # 0.8.x: bare loss.backward() is vanilla no-op; use interleaved form.
                with loss.backward():
                    pass
            opt.step()
            opt.zero_grad()
            print(f"step {step} loss {loss.item():.4f} |bias| {bias.norm().item():.4f}")
        trained = bias.detach().save()

    print(f"final |bias|={float(trained.norm()):.4f} moved={float(trained.norm()) > 0}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
