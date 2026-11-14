"""sawii/sync HTTP server — concurrent multi-lobe architecture.

Each turn fires L, R, and B as three independent concurrent NDIF calls
via asyncio.gather. They run simultaneously on the remote cluster.
Between turns they share context through a rolling history dict.

POST /v1/chat
  {"message": "...", "max_tokens": 64}
  → {"reply": "..."}
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ndif import NDIFChannels, LobePrompt, Generation
from memory_store import WorkingMemory, EpisodicMemory

MODEL = os.environ.get("NDIF_MODEL", "Qwen/Qwen3.8-27B")
PORT = int(os.environ.get("PORT", 8080))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("NDIF_MAX_NEW_TOKENS", 64))
HARD_MAX_NEW_TOKENS = int(os.environ.get("NDIF_HARD_MAX_NEW_TOKENS", 256))

_working_mem = WorkingMemory(maxlen=20)
_episodic_mem = EpisodicMemory(os.environ.get("EPISODIC_DB", "episodic.db"))

# Shared context between turns — each lobe reads the others' last output
_ctx: dict[str, str] = {"L": "", "R": "", "B": ""}
_history: list[tuple[str, str]] = []  # [(role, text), ...]

# One NDIFChannels instance is stateless per call — safe to share across lobes
_channels: NDIFChannels | None = None
_channels_lock = asyncio.Lock()

# asyncio event loop running in background thread for the HTTP server
_loop: asyncio.AbstractEventLoop | None = None


def get_channels() -> NDIFChannels:
    global _channels
    if _channels is None:
        _channels = NDIFChannels(model_name=MODEL)
    return _channels


_STOP_MARKERS = ("\nuser:", "\nassistant:", "\nUser:", "\nAssistant:")


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    cut = len(text)
    for m in _STOP_MARKERS:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def _l_prompt(message: str) -> str:
    """L: conversation lobe — replies to the user."""
    history = "\n".join(f"{r}: {t}" for r, t in _history[-8:])
    r_ctx = f"\n[R is executing: {_ctx['R']}]" if _ctx["R"] else ""
    b_ctx = f"\n[Memory: {_ctx['B']}]" if _ctx["B"] else ""
    return (
        f"{history}\n{r_ctx}{b_ctx}\nuser: {message}\nassistant:"
    ).strip()


def _r_prompt(message: str) -> str:
    """R: execution lobe — decides what action to take."""
    history = "\n".join(f"{r}: {t}" for r, t in _history[-4:])
    return (
        f"{history}\nuser request: {message}\n"
        f"[Determine what action or task to execute, if any. "
        f"Respond with the action or 'none'.]\nexecution:"
    ).strip()


def _b_prompt(message: str) -> str:
    """B: memory lobe — surfaces relevant context."""
    history = "\n".join(f"{r}: {t}" for r, t in _history[-4:])
    return (
        f"{history}\nuser: {message}\n"
        f"[Recall any relevant memory or context for this request.]\nmemory:"
    ).strip()


async def _run_lobe(lobe: str, prompt: str, max_tokens: int) -> Generation:
    channels = get_channels()
    return await channels.generate(
        [LobePrompt(lobe=lobe, text=prompt)],
        max_new_tokens=max_tokens,
        speaker=lobe,
    )


async def _turn(message: str, max_tokens: int) -> str:
    """Fire L, R, B concurrently. Return L's reply."""
    l_gen, r_gen, b_gen = await asyncio.gather(
        _run_lobe("L", _l_prompt(message), max_tokens),
        _run_lobe("R", _r_prompt(message), max_tokens // 2),
        _run_lobe("B", _b_prompt(message), max_tokens // 2),
        return_exceptions=True,
    )

    # Extract text, tolerating lobe failures
    l_text = _clean(l_gen.text) if isinstance(l_gen, Generation) else ""
    r_text = _clean(r_gen.text) if isinstance(r_gen, Generation) else ""
    b_text = _clean(b_gen.text) if isinstance(b_gen, Generation) else ""

    if not l_text:
        raise RuntimeError(f"L lobe failed: {l_gen}")

    # Update shared context for next turn
    _ctx["L"] = l_text
    _ctx["R"] = r_text
    _ctx["B"] = b_text
    _history.append(("user", message))
    _history.append(("assistant", l_text))
    if len(_history) > 40:
        _history[:] = _history[-40:]

    # Persist memory snapshot
    if isinstance(l_gen, Generation) and l_gen.tier1:
        _working_mem.save(l_gen.tier1, summary=message[:80])

    return l_text


def _run_turn_sync(message: str, max_tokens: int) -> str:
    """Run the async turn on the background event loop, block until done."""
    fut = asyncio.run_coroutine_threadsafe(_turn(message, max_tokens), _loop)
    return fut.result(timeout=300)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

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
            max_new = int(data.get("max_tokens", DEFAULT_MAX_NEW_TOKENS))
        except (TypeError, ValueError):
            self._json(400, {"error": "max_tokens must be an integer"})
            return
        max_new = max(1, min(max_new, HARD_MAX_NEW_TOKENS))

        try:
            reply = _run_turn_sync(message, max_new)
            self._json(200, {"reply": reply})
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop.run_forever()


if __name__ == "__main__":
    import threading
    print(f"sawii/sync starting on port {PORT}, model={MODEL}", flush=True)

    loop_thread = threading.Thread(target=_start_loop, daemon=True)
    loop_thread.start()

    # Wait for loop to be ready
    import time
    while _loop is None:
        time.sleep(0.01)

    try:
        get_channels()
        print("model channels ready", flush=True)
    except Exception:
        traceback.print_exc()
        print("channels init failed; will retry on first request", flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
