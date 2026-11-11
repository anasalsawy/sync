"""sawii/sync HTTP server — the model endpoint.

POST /v1/chat
  {"message": "..."}
  → {"reply": "..."}

This is the model. One call in, one reply out.
Internally: one batched NDIF forward pass across all lobes.
That's invisible to the caller — same as any model API.
"""
from __future__ import annotations

import os
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from ndif import NDIFChannels, LobePrompt
from memory_store import WorkingMemory, EpisodicMemory

MODEL = os.environ.get("NDIF_MODEL", "Qwen/Qwen3-0.6B")
PORT = int(os.environ.get("PORT", 8080))

# Module-level singletons — one model, one memory, shared across requests
_channels: NDIFChannels | None = None
_working_mem = WorkingMemory(maxlen=20)
_episodic_mem = EpisodicMemory(os.environ.get("EPISODIC_DB", "episodic.db"))
_history: list[str] = []  # rolling dialogue for L's context


def get_channels() -> NDIFChannels:
    global _channels
    if _channels is None:
        _channels = NDIFChannels(model_name=MODEL)
    return _channels


def _build_lobe_prompts(message: str) -> list[LobePrompt]:
    """Construct the internal lobe inputs from a user message.

    L gets the conversation. Everything else gets its own data stream.
    The user never sees this split — to them it's one model.
    """
    # L: raw conversation only
    dialogue = "\n".join(_history[-6:] + [f"user: {message}"])
    l_prompt = LobePrompt(lobe="L", text=dialogue)

    # R: environment placeholder (expand with real DOM/tool state later)
    r_prompt = LobePrompt(lobe="R", text="environment: idle")

    # B: episodic memory retrieval placeholder (expand with real retrieval later)
    b_prompt = LobePrompt(lobe="B", text="memory: none")

    return [l_prompt, r_prompt, b_prompt]


def _decode_reply(logits: dict, channels: NDIFChannels) -> str:
    """Greedy decode one token from L's logits, then return as text."""
    import torch
    l_logits = logits.get("L")
    if l_logits is None:
        return ""
    token_id = int(torch.argmax(l_logits.float()).item())
    return channels.model.tokenizer.decode([token_id], skip_special_tokens=True)


async def _think(message: str) -> str:
    channels = get_channels()
    prompts = _build_lobe_prompts(message)

    decision, logits, slot_act, tier1, auto_demands = await channels.think(prompts)

    reply = _decode_reply(logits, channels)

    # Update rolling history
    _history.append(f"user: {message}")
    _history.append(f"assistant: {reply}")
    if len(_history) > 40:
        _history[:] = _history[-40:]

    # Save to working memory
    _working_mem.save(tier1, summary=message[:80])

    return reply


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence default access log

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": MODEL})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat":
            self._json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except Exception:
            self._json(400, {"error": "invalid json"})
            return

        message = (data.get("message") or "").strip()
        if not message:
            self._json(400, {"error": "message required"})
            return

        try:
            reply = asyncio.get_event_loop().run_until_complete(_think(message))
            self._json(200, {"reply": reply})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"sawii/sync starting on port {PORT}, model={MODEL}", flush=True)
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
