"""sawii/sync HTTP server — thin HTTP wrapper around SyncRuntime.

SyncRuntime handles everything:
  - 5 lobes (L, R, B, V, C) in one batched pass with live activation sharing
  - Concurrent R lanes spawned emergently from R↔C divergence
  - Offspring running independently while L handles new messages
  - SharedState connecting all lobes between turns

POST /v1/chat  {"message": "...", "max_tokens": 64}  → {"reply": "..."}
GET  /health   → {"status": "ok"}
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from runtime_sync import SyncRuntime

MODEL = os.environ.get("NDIF_MODEL", "Qwen/Qwen3-8B")
PORT = int(os.environ.get("PORT", 8080))
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("NDIF_MAX_NEW_TOKENS", 64))
HARD_MAX_NEW_TOKENS = int(os.environ.get("NDIF_HARD_MAX_NEW_TOKENS", 256))

_runtime: SyncRuntime | None = None
_loop: asyncio.AbstractEventLoop | None = None
_init_lock: asyncio.Lock | None = None


async def _ensure_init():
    global _runtime, _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    async with _init_lock:
        if _runtime is None:
            _runtime = SyncRuntime(
                model_name=MODEL,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            )


async def _handle(message: str, max_tokens: int) -> str:
    await _ensure_init()
    _runtime.max_new_tokens = max_tokens
    result = await _runtime.turn(message)
    return result.text


def _run_sync(message: str, max_tokens: int) -> str:
    fut = asyncio.run_coroutine_threadsafe(
        _handle(message, max_tokens), _loop
    )
    return fut.result(timeout=300)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[http] {fmt % args}", flush=True)

    def do_GET(self):
        if self.path == "/health":
            print(f"[health] ok", flush=True)
            self._json(200, {"status": "ok", "model": MODEL})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat", "/v1/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length))
        except Exception:
            self._json(400, {"error": "invalid json"})
            return

        # OpenAI-compatible: extract last user message from messages array
        if "messages" in data:
            msgs = data["messages"]
            user_msgs = [m.get("content", "") for m in msgs if m.get("role") == "user"]
            message = (user_msgs[-1] if user_msgs else "").strip()
        else:
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
            reply = _run_sync(message, max_new)
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": str(e)})
            return

        # OpenAI-compatible response for /v1/chat/completions
        if self.path == "/v1/chat/completions":
            self._json(200, {
                "id": f"chatcmpl-sync-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        else:
            self._json(200, {"reply": reply})

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
    print(f"sawii/sync starting — port={PORT} model={MODEL}", flush=True)

    threading.Thread(target=_start_loop, daemon=True).start()
    while _loop is None:
        time.sleep(0.01)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"sawii/sync HTTP server bound on 0.0.0.0:{PORT}", flush=True)
    try:
        server.serve_forever()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
