"""sawii/sync HTTP server — SyncRuntime + companion architecture.

Modes (SYNC_RUNTIME):
  - sync_runtime (default): thin wrapper around SyncRuntime (NDIF lobes + offspring)
  - local_crossing: CompanionSession (Crossing, boards, voice, Multi-R task groups)
  - ndif_batch: legacy remote NDIF batched L/R/B/V/C generate

POST /v1/chat              message → reply (all modes)
POST /v1/chat/completions  OpenAI-compatible (sync_runtime)
POST /v1/barge_in          stop speech only (companion)
POST /v1/task              L authorize → main R task group (companion)
GET  /v1/session           boards + lanes (companion)
GET  /health
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MODEL = os.environ.get("NDIF_MODEL", "Qwen/Qwen3-8B")
PORT = int(os.environ.get("PORT", 8080))
SYNC_RUNTIME = os.environ.get("SYNC_RUNTIME", "sync_runtime")
COMPANION_DB = os.environ.get("COMPANION_DB", "companion.db")

DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("NDIF_MAX_NEW_TOKENS", 64))
HARD_MAX_NEW_TOKENS = int(os.environ.get("NDIF_HARD_MAX_NEW_TOKENS", 256))

_runtime = None
_loop: asyncio.AbstractEventLoop | None = None
_init_lock: asyncio.Lock | None = None
_channels = None
_history: list[str] = []
_local_runtime = None
_companion = None
_turn_lock = threading.Lock()
_working_mem = None
_episodic_mem = None


def _memory():
    global _working_mem, _episodic_mem
    if _working_mem is None:
        from memory_store import WorkingMemory, EpisodicMemory
        _working_mem = WorkingMemory(maxlen=20)
        _episodic_mem = EpisodicMemory(os.environ.get("EPISODIC_DB", "episodic.db"))
    return _working_mem, _episodic_mem


async def _ensure_sync_runtime():
    global _runtime, _init_lock
    from runtime_sync import SyncRuntime
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    async with _init_lock:
        if _runtime is None:
            _runtime = SyncRuntime(
                model_name=MODEL,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            )


async def _handle_sync_runtime(message: str, max_tokens: int) -> str:
    await _ensure_sync_runtime()
    _runtime.max_new_tokens = max_tokens
    result = await _runtime.turn(message)
    return result.text


def _run_sync_runtime(message: str, max_tokens: int) -> str:
    fut = asyncio.run_coroutine_threadsafe(
        _handle_sync_runtime(message, max_tokens), _loop,
    )
    return fut.result(timeout=300)


def get_channels():
    global _channels
    if _channels is None:
        from ndif import NDIFChannels
        _channels = NDIFChannels(model_name=MODEL)
    return _channels


def get_companion(session_id: str | None = None, *, reset: bool = False):
    global _companion
    if reset or _companion is None or (
        session_id and _companion.session_id != session_id
    ):
        from companion_session import CompanionSession
        _companion = CompanionSession(
            session_id=session_id,
            db_path=COMPANION_DB,
            crossing_enabled=True,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        )
    return _companion


_STOP_MARKERS = ("\nuser:", "\nassistant:", "\nUser:", "\nAssistant:")


def _clean_reply(text: str) -> str:
    text = text or ""
    closed = re.search(r"</think>\s*(.*)$", text, flags=re.S)
    if closed:
        text = closed.group(1)
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        if "<think>" in text:
            text = text.split("<think>", 1)[0]
    cut = len(text)
    for marker in _STOP_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


def _think_ndif_batch(message: str, max_new_tokens: int) -> str:
    from ndif import LobePrompt
    working, _ = _memory()
    channels = get_channels()
    dialogue = "\n".join(_history[-6:] + [f"user: {message}", "assistant:"])
    prompts = [
        LobePrompt(lobe="L", text=dialogue),
        LobePrompt(lobe="R", text="environment: idle"),
        LobePrompt(lobe="B", text="memory: none"),
        LobePrompt(lobe="V", text="vision: none"),
        LobePrompt(lobe="C", text="constraints: none"),
    ]
    gen = channels.generate_sync(prompts, max_new_tokens=max_new_tokens)
    reply = _clean_reply(gen.text)
    _history.append(f"user: {message}")
    _history.append(f"assistant: {reply}")
    if len(_history) > 40:
        _history[:] = _history[-40:]
    working.save(gen.tier1, summary=message[:80])
    return reply


def _think_companion(
    message: str,
    max_new_tokens: int,
    token_queue: queue.Queue | None = None,
) -> dict:
    companion = get_companion()
    companion.mind.max_new_tokens = max_new_tokens

    def on_token(_lobe: str, piece: str) -> None:
        if token_queue is not None:
            token_queue.put(("token", piece))

    def on_event(ev: dict) -> None:
        if token_queue is not None and ev.get("type") in ("audio", "audio_end", "barge_in"):
            token_queue.put(("event", ev))

    companion.on_event = on_event
    result = asyncio.run(companion.chat(message, speak=True, on_token=on_token))
    reply = _clean_reply(result.reply) if result.reply else (result.reply or "")
    _history[:] = companion.mind.history
    payload = {
        "reply": reply,
        "runtime": SYNC_RUNTIME,
        "session_id": result.session_id,
        "lanes_active": result.lanes_active,
        "ttfa_s": result.ttfa_s,
    }
    if token_queue is not None:
        token_queue.put(("done", payload))
    return payload


def _think(message: str, max_new_tokens: int) -> dict | str:
    with _turn_lock:
        if SYNC_RUNTIME == "ndif_batch":
            return _think_ndif_batch(message, max_new_tokens)
        if SYNC_RUNTIME == "local_crossing":
            return _think_companion(message, max_new_tokens)
        return _run_sync_runtime(message, max_new_tokens)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[http] {fmt % args}", flush=True)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/health":
            print("[health] ok", flush=True)
            roster = {
                "ndif_batch": ["L", "R", "B", "V", "C"],
                "local_crossing": ["L", "R", "B", "V", "C", "Embedder"],
                "sync_runtime": ["L", "R", "B", "V", "C"],
            }.get(SYNC_RUNTIME, ["L", "R", "B", "V", "C"])
            self._json(200, {
                "status": "ok",
                "model": MODEL,
                "runtime": SYNC_RUNTIME,
                "roster": roster,
            })
            return

        if path == "/v1/session" and SYNC_RUNTIME == "local_crossing":
            sid = (qs.get("id") or [None])[0]
            try:
                self._json(200, get_companion(sid).snapshot())
            except Exception as e:
                traceback.print_exc()
                self._json(500, {"error": str(e)})
            return

        self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw or b"{}")
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
        except Exception:
            self._json(400, {"error": "invalid json"})
            return

        if path == "/v1/barge_in" and SYNC_RUNTIME == "local_crossing":
            try:
                get_companion().barge_in()
                self._json(200, {"ok": True, "speech": "stopped", "lanes": "continue"})
            except Exception as e:
                traceback.print_exc()
                self._json(500, {"error": str(e)})
            return

        if path == "/v1/task" and SYNC_RUNTIME == "local_crossing":
            goal = (data.get("goal") or data.get("message") or "").strip()
            if not goal:
                self._json(400, {"error": "goal required"})
                return
            require_c = bool(data.get("require_c", True))
            try:
                with _turn_lock:
                    companion = get_companion()
                    lane_id = asyncio.run(
                        companion.submit_task(goal, require_c=require_c),
                    )
                self._json(200, {
                    "lane_id": lane_id,
                    "started": bool(lane_id),
                    "decision_wait_s": companion.decision_wait_s,
                    "session": companion.snapshot(),
                })
            except Exception as e:
                traceback.print_exc()
                self._json(500, {"error": str(e)})
            return

        if path == "/v1/task/redirect" and SYNC_RUNTIME == "local_crossing":
            lane_id = (data.get("lane_id") or "").strip()
            instruction = (data.get("instruction") or data.get("goal") or "").strip()
            if not lane_id or not instruction:
                self._json(400, {"error": "lane_id and instruction required"})
                return
            try:
                get_companion().redirect_lane(lane_id, instruction)
                self._json(200, {"ok": True, "lane_id": lane_id})
            except Exception as e:
                traceback.print_exc()
                self._json(500, {"error": str(e)})
            return

        if path not in ("/v1/chat", "/v1/chat/completions"):
            self._json(404, {"error": "not found"})
            return

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
            max_new = max(
                1,
                min(int(data.get("max_tokens", DEFAULT_MAX_NEW_TOKENS)), HARD_MAX_NEW_TOKENS),
            )
        except (TypeError, ValueError):
            self._json(400, {"error": "max_tokens must be an integer"})
            return

        stream = bool(data.get("stream"))
        if stream and SYNC_RUNTIME == "local_crossing":
            self._stream_companion(message, max_new)
            return

        try:
            result = _think(message, max_new)
            if path == "/v1/chat/completions":
                reply = result if isinstance(result, str) else result.get("reply", "")
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
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                })
            elif isinstance(result, str):
                self._json(200, {"reply": result, "runtime": SYNC_RUNTIME})
            else:
                self._json(200, result)
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": str(e)})

    def _stream_companion(self, message: str, max_new: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        q: queue.Queue = queue.Queue()

        def runner() -> None:
            try:
                with _turn_lock:
                    _think_companion(message, max_new, token_queue=q)
            except BaseException as e:
                q.put(("error", str(e)))

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        try:
            while True:
                kind, payload = q.get()
                if kind == "token":
                    frame = json.dumps({"token": payload})
                    self.wfile.write(f"event: token\ndata: {frame}\n\n".encode())
                    self.wfile.flush()
                elif kind == "event":
                    frame = json.dumps(payload)
                    self.wfile.write(
                        f"event: {payload.get('type', 'event')}\ndata: {frame}\n\n".encode()
                    )
                    self.wfile.flush()
                elif kind == "done":
                    frame = json.dumps(
                        payload if isinstance(payload, dict) else {"reply": payload}
                    )
                    self.wfile.write(f"event: reply\ndata: {frame}\n\n".encode())
                    self.wfile.write(b"event: done\ndata: {}\n\n")
                    self.wfile.flush()
                    break
                elif kind == "error":
                    frame = json.dumps({"error": payload})
                    self.wfile.write(f"event: error\ndata: {frame}\n\n".encode())
                    self.wfile.flush()
                    break
        except Exception:
            traceback.print_exc()
        t.join(timeout=1)

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
    print(
        f"sawii/sync starting — port={PORT} model={MODEL} runtime={SYNC_RUNTIME}",
        flush=True,
    )
    if SYNC_RUNTIME == "sync_runtime":
        threading.Thread(target=_start_loop, daemon=True).start()
        while _loop is None:
            time.sleep(0.01)
    else:
        try:
            if SYNC_RUNTIME == "ndif_batch":
                get_channels()
            else:
                get_companion()
        except Exception:
            traceback.print_exc()
            print("warm-up failed; will retry on first request", flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"sawii/sync HTTP server bound on 0.0.0.0:{PORT}", flush=True)
    try:
        server.serve_forever()
    except Exception:
        traceback.print_exc()
        raise
