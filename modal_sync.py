"""Run Sync (HF + Crossing, V lobe) on Modal GPU — not a tokens-only API.

Owns the forward pass so activation / attention sharing works.
Base model: Qwen/Qwen3.8-27B (override with SYNC_MODEL).
"""
from __future__ import annotations

import modal

APP_NAME = "sync-crossing"
MODEL_DEFAULT = "Qwen/Qwen3.8-27B"

# A100 80GB: 27B bf16 ≈ 54GB weights + KV headroom for short turns
GPU = "A100-80GB"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        # Qwen3.8 hybrid arch needs a recent transformers
        "transformers>=4.57.0",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "safetensors",
        "huggingface_hub",
        "einops",
    )
    .add_local_dir(
        ".",
        remote_path="/root/sync",
        ignore=[
            ".git",
            "**/__pycache__",
            "eval_results/**",
            "*.db",
            "hands_live_root/**",
            ".venv/**",
            "agent-transcripts/**",
        ],
    )
)

app = modal.App(APP_NAME, image=image)
model_vol = modal.Volume.from_name("sync-hf-cache", create_if_missing=True)


@app.function(
    gpu=GPU,
    timeout=60 * 60,
    memory=65536,
    volumes={"/root/.cache/huggingface": model_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def sync_turn(
    message: str = "Say hello in one short sentence.",
    *,
    model_name: str = MODEL_DEFAULT,
    max_new_tokens: int = 32,
    activation_injection: bool = True,
) -> dict:
    """One SyncSession turn on Modal GPU with live Crossing."""
    import os
    import sys
    import asyncio

    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface")
    os.environ["SYNC_LOCAL_BACKEND"] = "hf"
    os.environ["NDIF_MODEL"] = model_name
    # Prefer Modal secret; fall back to empty (public weights)
    if not os.environ.get("HF_TOKEN") and os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    sys.path.insert(0, "/root/sync")

    from sync_session import SyncSession

    async def _run() -> dict:
        session = SyncSession(
            model_name=model_name,
            backend="hf",
            max_new_tokens=max_new_tokens,
            crossing_enabled=True,
            activation_injection=activation_injection,
            isolate_peers=False,  # single GPU process — share Crossing in-proc
        )
        result = await session.turn(message)
        l_prof = result.l_inject_profile()
        return {
            "ok": True,
            "model": model_name,
            "roster": list(session.ROSTER),
            "has_v": "V" in session.ROSTER,
            "reply": result.reply,
            "config": result.config,
            "activation_injection": result.activation_injection,
            "ttft_s": result.ttft_s,
            "total_s": result.total_s,
            "l_inject_count": l_prof.get("inject_count", 0),
            "l_inject_ops": l_prof.get("inject_ops", 0),
            "crossing": result.crossing_telemetry,
            "lobe_texts": {k: (v.text or "")[:200] for k, v in result.results.items()},
        }

    out = asyncio.run(_run())
    model_vol.commit()
    return out


@app.local_entrypoint()
def main(
    message: str = "Ping Sync on Modal — one short reply.",
    model: str = MODEL_DEFAULT,
    max_tokens: int = 32,
):
    print(f"Modal Sync turn model={model} gpu={GPU}")
    result = sync_turn.remote(
        message,
        model_name=model,
        max_new_tokens=max_tokens,
        activation_injection=True,
    )
    print("=== SYNC MODAL RESULT ===")
    for k, v in result.items():
        if k == "crossing":
            print(f"  crossing_keys={list(v.keys()) if isinstance(v, dict) else type(v)}")
            continue
        print(f"  {k}={v}")
    if not result.get("has_v"):
        raise SystemExit("V lobe missing from roster")
    if result.get("activation_injection") and result.get("l_inject_count", 0) < 0:
        raise SystemExit("activation injection profile missing")
    print("SYNC_MODAL_OK")
