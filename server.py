"""sawii/sync HTTP server — the model endpoint.

POST /v1/chat
  {"message": "...", "max_tokens": 64}      # max_tokens optional
  → {"reply": "..."}

This is the model. One call in, one reply out.
Internally: ONE remote NDIF generate() job. The multi-lobe channel pass runs
once on the prefill step and the reply is decoded server-side with the KV
cache — no per-token network round trips. That's invisible to the caller —
same as any model API.
"""
from __future__ import annotations

import json
import os
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ndif import NDIFChannels, LobePrompt
from memory_store import WorkingMemory, EpisodicMemory

MODEL = os.environ.get("NDIF_MODEL", "Qwen/Qwen3.8-27B")
PORT = int(os.environ.get("PORT", 8080))

# Reply length. Decode time dominates latency, so keep the default modest;
# callers can ask for more per request (up to the hard cap).
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("NDIF_MAX_NEW_TOKENS", 64))
HARD_MAX_NEW_TOKENS = int(os.environ.get("NDIF_HARD_MAX_NEW_TOKENS", 256))

# Module-level singletons — one model, one memory, shared across requests
_channels: NDIFChannels | None = None
_working_mem = WorkingMemory(maxlen=20)
_episodic_mem = EpisodicMemory(os.environ.get("EPISODIC_DB", "episodic.db"))
_history: list[str] = []  # rolling dialogue for L's context

# The history, memory and model are shared, so one turn runs at a time.
# (Health checks never take this lock, so they stay responsive mid-generation.)
_turn_lock = threading.Lock()


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
    # L: raw conversation only. The trailing "assistant:" is the cue that
    # makes the model answer; without it the next token continues the
    # user's own line.
    dialogue = "\n".join(_history[-6:] + [f"user: {message}", "assistant:"])
    l_prompt = LobePrompt(lobe="L", text=dialogue)

    # R: environment placeholder (expand with real DOM/tool state later)
    r_prompt = LobePrompt(lobe="R", text="environment: idle")

    # B: episodic memory retrieval placeholder (expand with real retrieval later)
    b_prompt = LobePrompt(lobe="B", text="memory: none")

    return [l_prompt, r_prompt, b_prompt]


_STOP_MARKERS = ("\nuser:", "\nassistant:", "\nUser:", "\nAssistant:")


def _clean_reply(text: str) -> str:
    """Trim a raw completion down to the assistant's single turn."""
    # Drop reasoning blocks if the model emits them.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:  # unclosed block: nothing usable after it
        text = text.split("<think>", 1)[0]
    # The model is completing a transcript; cut where it starts the next turn.
    cut = len(text)
    for marker in _STOP_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def _think(message: str, max_new_tokens: int) -> str:
    with _turn_lock:
        channels = get_channels()
        prompts = _build_lobe_prompts(message)

        gen = channels.generate_sync(prompts, max_new_tokens=max_new_tokens)
        reply = _clean_reply(gen.text)

        # Update rolling history
        _history.append(f"user: {message}")
        _history.append(f"assistant: {reply}")
        if len(_history) > 40:
            _history[:] = _history[-40:]

        # Save to working memory
        _working_mem.save(gen.tier1, summary=message[:80])

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
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
        except Exception:
            self._json(400, {"error": "invalid json"})
            return

        message = (data.get("message") or "").strip()
        if not message:
            self._json(400, {"error": "message required"})
            return

        try:
            max_new = int(data.get("max_tokens", DEFAULT_MAX_NEW_TOKENS))
        except (TypeError, ValueError):
            self._json(400, {"error": "max_tokens must be an integer"})
            return
        max_new = max(1, min(max_new, HARD_MAX_NEW_TOKENS))

        try:
            reply = _think(message, max_new)
            self._json(200, {"reply": reply})
        except Exception as e:
            traceback.print_exc()  # access log is silenced; keep errors visible
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
    # Load the model handle at boot so the first request doesn't pay for it.
    try:
        get_channels()
    except Exception:
        traceback.print_exc()
        print("model warm-up failed; will retry on first request", flush=True)
    # Threaded: /health keeps answering while a generation is in flight.
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
