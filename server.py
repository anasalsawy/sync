"""sawii/sync HTTP server — dual-mode L/R architecture.

COUPLED mode: L and R think together in one batched NDIF call.
  - Full live activation sharing during prefill (cross-slot wiring)
  - Used when both lobes are working on the same thing
  - L and R start and finish together

DECOUPLED mode: L converses while R executes independently.
  - R spawns as a background offspring, never blocks L
  - L answers immediately, reads R's latest state from SharedState
  - Used when execution is separate from conversation

The system starts coupled. It switches to decoupled when R has an
ongoing task that should keep running across turns. It snaps back to
coupled when a new task needs joint reasoning.

POST /v1/chat  {"message": "...", "max_tokens": 64}  → {"reply": "..."}
GET  /health   → {"status": "ok"}
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ndif import NDIFChannels, NDIFOffspring, LobePrompt, SharedState, Generation
from memory_store import WorkingMemory, EpisodicMemory

MODEL_L = os.environ.get("MODEL_L", os.environ.get("NDIF_MODEL", "Qwen/Qwen3.8-27B"))
MODEL_R = os.environ.get("MODEL_R", os.environ.get("NDIF_MODEL", "Qwen/Qwen3.8-27B"))
MODEL_B = os.environ.get("MODEL_B", os.environ.get("NDIF_MODEL", "Qwen/Qwen3.8-27B"))

PORT = int(os.environ.get("PORT", 8080))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("NDIF_MAX_NEW_TOKENS", 64))
HARD_MAX_NEW_TOKENS = int(os.environ.get("NDIF_HARD_MAX_NEW_TOKENS", 256))

_working_mem = WorkingMemory(maxlen=20)
_history: list[tuple[str, str]] = []
_shared: SharedState = SharedState()

# Separate channel instances — different models if configured, else same model
# with separate queues on NDIF
_channels_coupled: NDIFChannels | None = None   # for batched L+R+B passes
_offspring_engine: NDIFOffspring | None = None  # for decoupled R background tasks
_r_handles: dict[str, any] = {}                 # active R lanes: {lane_id: handle}
_r_counter = 0                                  # increments per spawned lane
_mode = "coupled"                               # "coupled" | "decoupled"

_loop: asyncio.AbstractEventLoop | None = None
_init_lock: asyncio.Lock | None = None

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


async def _ensure_init():
    global _channels_coupled, _offspring_engine, _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    async with _init_lock:
        if _channels_coupled is None:
            _channels_coupled = NDIFChannels(model_name=MODEL_L)
        if _offspring_engine is None:
            _offspring_engine = NDIFOffspring(model_name=MODEL_R)


def _has_active_r() -> bool:
    return any(not h.task.done() for h in _r_handles.values())


def _reap_done_lanes():
    """Remove completed lanes from the pool."""
    done = [lid for lid, h in _r_handles.items() if h.task.done()]
    for lid in done:
        del _r_handles[lid]


def _r_status() -> str:
    """Summarise all active R lanes from SharedState — non-blocking."""
    states = _shared.offspring_states()
    parts = []
    for lid in list(_r_handles.keys()):
        state = states.get(lid)
        if state and state.text_so_far:
            parts.append(f"{lid}: {state.text_so_far[:100]}")
    return "; ".join(parts) if parts else ""


def _build_history_str(n: int = 8) -> str:
    return "\n".join(f"{r}: {t}" for r, t in _history[-n:])


def _l_prompt(message: str) -> str:
    h = _build_history_str()
    r_ctx = f"\n[R executing: {_r_status()}]" if _r_status() else ""
    return f"{h}{r_ctx}\nuser: {message}\nassistant:".strip()


def _r_prompt(message: str) -> str:
    h = _build_history_str(4)
    return (
        f"{h}\nuser request: {message}\n"
        f"[Execute any needed action. Be thorough.]\nexecution:"
    ).strip()


def _b_prompt(message: str) -> str:
    h = _build_history_str(4)
    return (
        f"{h}\nuser: {message}\n"
        f"[Recall relevant memory or context.]\nmemory:"
    ).strip()


def _needs_execution(message: str) -> bool:
    """Heuristic: does this message need R to do independent background work?"""
    exec_keywords = (
        "open", "search", "find", "click", "browse", "go to", "navigate",
        "run", "execute", "do", "perform", "check", "fetch", "get",
        "write", "create", "make", "build", "install", "download",
    )
    low = message.lower()
    return any(kw in low for kw in exec_keywords)


async def _coupled_turn(message: str, max_tokens: int) -> str:
    """L + R + B in one batched call — full live activation sharing."""
    gen: Generation = await _channels_coupled.generate(
        [
            LobePrompt(lobe="L", text=_l_prompt(message)),
            LobePrompt(lobe="R", text=_r_prompt(message)),
            LobePrompt(lobe="B", text=_b_prompt(message)),
        ],
        max_new_tokens=max_tokens,
        speaker="L",
        shared=_shared,
    )
    return _clean(gen.text)


async def _decoupled_turn(message: str, max_tokens: int) -> str:
    """L answers immediately. R spawns as a new concurrent lane."""
    global _r_counter, _mode

    # Reap finished lanes before spawning
    _reap_done_lanes()

    # Spawn a new R lane — does NOT cancel existing lanes
    _r_counter += 1
    lane_id = f"R{_r_counter}"
    handle = _offspring_engine.spawn(
        LobePrompt(lobe=lane_id, text=_r_prompt(message)),
        _shared,
        max_new_tokens=max_tokens,
    )
    _r_handles[lane_id] = handle

    # L answers without waiting for R
    gen: Generation = await _channels_coupled.generate(
        [LobePrompt(lobe="L", text=_l_prompt(message))],
        max_new_tokens=max_tokens,
        speaker="L",
        shared=_shared,
    )
    return _clean(gen.text)


async def _handle_turn(message: str, max_tokens: int) -> str:
    global _mode

    await _ensure_init()

    # Mode decision:
    # - If R is actively running a background task → stay decoupled (don't interrupt)
    # - If message needs independent execution → go decoupled
    # - Otherwise → coupled (joint reasoning, full activation sharing)
    if _has_active_r():
        _mode = "decoupled"
    elif _needs_execution(message):
        _mode = "decoupled"
    else:
        _mode = "coupled"

    if _mode == "coupled":
        reply = await _coupled_turn(message, max_tokens)
    else:
        reply = await _decoupled_turn(message, max_tokens)

    _history.append(("user", message))
    _history.append(("assistant", reply))
    if len(_history) > 40:
        _history[:] = _history[-40:]

    return reply


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
            self._json(200, {"status": "ok", "model": MODEL_L, "mode": _mode})
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
    import threading
    print(f"sawii/sync starting — port={PORT} L={MODEL_L} R={MODEL_R}", flush=True)

    threading.Thread(target=_start_loop, daemon=True).start()
    while _loop is None:
        time.sleep(0.01)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
