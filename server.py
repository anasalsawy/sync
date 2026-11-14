"""sawii/sync HTTP server — non-blocking L/R architecture.

L (conversation) answers every user message immediately.
R (execution) runs as a background offspring — never blocks L.
B (memory) spawns per-turn, feeds context back to L.

POST /v1/chat  {"message": "...", "max_tokens": 64}  → {"reply": "..."}
GET  /health   → {"status": "ok"}
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ndif import NDIFChannels, NDIFOffspring, LobePrompt, SharedState, Generation
from memory_store import WorkingMemory, EpisodicMemory

MODEL = os.environ.get("NDIF_MODEL", "Qwen/Qwen3.8-27B")
PORT = int(os.environ.get("PORT", 8080))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("NDIF_MAX_NEW_TOKENS", 64))
HARD_MAX_NEW_TOKENS = int(os.environ.get("NDIF_HARD_MAX_NEW_TOKENS", 256))

_working_mem = WorkingMemory(maxlen=20)
_history: list[tuple[str, str]] = []

# Shared state hub — R writes here, L reads it
_shared: SharedState | None = None
_channels: NDIFChannels | None = None      # L's inference engine
_offspring: NDIFOffspring | None = None    # R/B's inference engine
_r_handle = None                           # current R task handle (may be running)

_loop: asyncio.AbstractEventLoop | None = None
_init_lock = asyncio.Lock()


async def _ensure_init():
    global _shared, _channels, _offspring
    async with _init_lock:
        if _channels is None:
            _channels = NDIFChannels(model_name=MODEL)
        if _offspring is None:
            _offspring = NDIFOffspring(model_name=MODEL)
        if _shared is None:
            _shared = SharedState()


_STOP = ("\nuser:", "\nassistant:", "\nUser:", "\nAssistant:")


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    cut = len(text)
    for m in _STOP:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def _l_prompt(message: str) -> str:
    history = "\n".join(f"{r}: {t}" for r, t in _history[-8:])
    # Read R's latest status from shared state (non-blocking)
    r_state = _shared.get_offspring("R") if _shared else None
    r_ctx = f"\n[R is working: {r_state.text_so_far[:120]}]" if (r_state and r_state.text_so_far) else ""
    return f"{history}{r_ctx}\nuser: {message}\nassistant:".strip()


def _r_prompt(message: str) -> str:
    history = "\n".join(f"{r}: {t}" for r, t in _history[-4:])
    return (
        f"{history}\nuser request: {message}\n"
        f"[Execute any needed action for this request. Be thorough.]\nexecution:"
    ).strip()


async def _handle_turn(message: str, max_tokens: int) -> str:
    global _r_handle

    await _ensure_init()

    # Spawn R as a background task — does NOT block L
    # Cancel previous R if still running (new message supersedes old task)
    if _r_handle is not None and not _r_handle.task.done():
        _r_handle.task.cancel()

    _r_handle = _offspring.spawn(
        LobePrompt(lobe="R", text=_r_prompt(message)),
        _shared,
        max_new_tokens=max_tokens,
    )
    # R is now running in the background — L does not wait for it

    # L answers immediately
    l_gen: Generation = await _channels.generate(
        [LobePrompt(lobe="L", text=_l_prompt(message))],
        max_new_tokens=max_tokens,
        speaker="L",
        shared=_shared,
    )

    l_text = _clean(l_gen.text)
    if not l_text:
        raise RuntimeError("L lobe returned empty reply")

    _history.append(("user", message))
    _history.append(("assistant", l_text))
    if len(_history) > 40:
        _history[:] = _history[-40:]

    if l_gen.tier1:
        _working_mem.save(l_gen.tier1, summary=message[:80])

    return l_text


def _run_sync(message: str, max_tokens: int) -> str:
    fut = asyncio.run_coroutine_threadsafe(
        _handle_turn(message, max_tokens), _loop
    )
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
        try:
            data = json.loads(self.rfile.read(length))
        except Exception:
            self._json(400, {"error": "invalid json"})
            return

        message = (data.get("message") or "").strip()
        if not message:
            self._json(400, {"error": "message required"})
            return

        try:
            max_new = max(1, min(int(data.get("max_tokens", DEFAULT_MAX_NEW_TOKENS)),
                                 HARD_MAX_NEW_TOKENS))
        except (TypeError, ValueError):
            self._json(400, {"error": "max_tokens must be an integer"})
            return

        try:
            self._json(200, {"reply": _run_sync(message, max_new)})
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
    import threading, time
    print(f"sawii/sync starting — port={PORT} model={MODEL}", flush=True)

    threading.Thread(target=_start_loop, daemon=True).start()
    while _loop is None:
        time.sleep(0.01)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
