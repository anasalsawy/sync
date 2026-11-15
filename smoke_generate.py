"""Smoke test: multi-lobe generate with live mind-read; only L speaks."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_key() -> None:
    if os.environ.get("NDIF_API_KEY"):
        return
    for path in (
        Path.home() / ".env",
        Path.home() / ".nnsight" / "key",
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


def main() -> int:
    _load_key()
    if not os.environ.get("NDIF_API_KEY"):
        print("NDIF_API_KEY missing — set it to run the live smoke test", flush=True)
        return 2

    from ndif import NDIFChannels, LobePrompt, _dialogue_to_messages, _format_l_for_qwen

    model = os.environ.get("NDIF_MODEL", "Qwen/Qwen3.8-27B")
    print(f"model={model}", flush=True)

    # Offline check of the empty-reply bugfix first.
    msgs = _dialogue_to_messages("user: Name one ocean.\nassistant:")
    assert msgs == [{"role": "user", "content": "Name one ocean."}], msgs
    print("prompt parse ok:", msgs, flush=True)

    adapter = NDIFChannels(model_name=model)
    preview = _format_l_for_qwen(adapter.model.tokenizer, "user: Name one ocean.\nassistant:")
    print("L template preview (first 240 chars):", repr(preview[:240]), flush=True)

    prompts = [
        LobePrompt("L", "user: Reply with exactly: hello sync\nassistant:"),
        LobePrompt("R", "environment: idle"),
        LobePrompt("B", "memory: none"),
        LobePrompt("V", "vision: none"),
        LobePrompt("C", "constraints: none"),
    ]
    print("generating...", flush=True)
    gen = adapter.generate_sync(prompts, max_new_tokens=32)
    print("exchanges:", len(gen.decision.exchanges), flush=True)
    print("token_ids:", len(gen.token_ids), gen.token_ids[:12], flush=True)
    print("text:", repr(gen.text), flush=True)
    if not gen.text.strip():
        print("FAIL: empty reply", flush=True)
        return 1
    print("OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
