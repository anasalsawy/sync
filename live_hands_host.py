"""Real Multi-R host: one Hands engine per task.

Design: three R workers claim/prepare on one shared task; dispatch goes through
a single Hands instance (one hand). R workers are not 1:1 with Hands engines.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from host_subagents import PreparedActionMessage

HANDS_REPO = Path(r"C:\Users\anasa\hands")
if str(HANDS_REPO) not in sys.path:
    sys.path.insert(0, str(HANDS_REPO))

from sawii_hands import Hands  # noqa: E402


def _tool_for(message: PreparedActionMessage, root: Path) -> tuple[str, dict[str, Any]]:
    """Map a Multi-R prepared action onto a Hands tool call."""
    effect = (message.effect or "").lower()
    resource = message.resource or message.target or "artifact.txt"
    rel = resource.replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        rel = "safe/" + Path(rel).name
    path = str(root / rel)

    if effect in ("edit", "write"):
        content = str(
            message.payload.get("content")
            or message.payload.get("file_content")
            or message.args.get("text")
            or message.args.get("patch")
            or f"patched by {message.worker_id}:{message.action_id}\n"
        )
        prior = message.payload.get("prior_content")
        if effect == "edit" and prior and Path(path).is_file():
            return "edit_file", {
                "path": path,
                "old": str(prior),
                "new": content if content != str(prior) else str(prior) + "\n# ok\n",
            }
        return "write_file", {"path": path, "content": content, "mode": "overwrite"}

    if effect in ("read", "investigate"):
        if Path(path).is_file():
            return "read_file", {"path": path, "offset": 0, "limit": 200}
        return "list_files", {"path": str(root), "depth": 2}

    if effect == "test":
        failing = bool(message.payload.get("failing", False))
        cmd = "exit 1" if failing else "Write-Output 'TESTS_PASS'"
        return "shell", {"command": cmd, "cwd": str(root), "timeout_s": 30}

    if effect in ("process", "send"):
        stamp = f"{effect}:{message.action_id}:{message.worker_id}\n"
        out = root / "processed" / f"{Path(rel).name}.done"
        return "write_file", {
            "path": str(out),
            "content": stamp,
            "mode": "overwrite",
        }

    return "write_file", {
        "path": str(root / "actions" / f"{message.action_id}.txt"),
        "content": f"{effect}@{resource}\n",
        "mode": "overwrite",
    }


def _seed_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "src").mkdir(exist_ok=True)
    (root / "processed").mkdir(exist_ok=True)
    (root / "actions").mkdir(exist_ok=True)
    app = root / "src" / "app.py"
    if not app.exists():
        app.write_text("# live hands fixture\ndef main():\n    return 0\n", encoding="utf-8")
    inbox = root / "inbox"
    if not inbox.exists():
        inbox.write_text("hello from inbox\n", encoding="utf-8")


@dataclass
class HandsHost:
    """One Hands engine for the whole task. Three R workers prepare ahead.

    ``supports_subagents=False`` → companion keeps all three R workers active
    and feeds this single execution path (prep-ahead / claim / dispatch).
    """

    root: Path
    supports_subagents: bool = False
    supports_concurrent_dispatch: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    in_flight: int = 0
    idle_s: float = 0.0
    busy_s: float = 0.0
    hands_engine_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    launched: list[str] = field(default_factory=list)
    _last_free: float | None = None
    _hands: Hands | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        _seed_root(self.root)
        self._hands = Hands(roots=[str(self.root)], approve="auto")

    async def launch_subagents(self, worker_ids: list[str]) -> dict[str, Any]:
        raise RuntimeError(
            "HandsHost is one hand per task — do not map R workers 1:1 to Hands. "
            "Keep three R workers preparing on the shared task; dispatch through "
            "execute_single."
        )

    async def execute_single(
        self, message: PreparedActionMessage,
    ) -> dict[str, Any]:
        assert self._hands is not None
        async with self._lock:
            now = time.perf_counter()
            if self._last_free is not None and self.in_flight == 0:
                self.idle_s += now - self._last_free
            self.in_flight = 1
            tool, args = _tool_for(message, self.root)
            t0 = time.perf_counter()
            raw = await asyncio.to_thread(self._hands.run, tool, args)
            elapsed = time.perf_counter() - t0
            self.busy_s += elapsed

            try:
                parsed: Any = (
                    json.loads(raw)
                    if isinstance(raw, str) and raw[:1] in "{["
                    else raw
                )
            except json.JSONDecodeError:
                parsed = raw

            ok = True
            if isinstance(parsed, dict) and parsed.get("error"):
                ok = False
            if message.effect == "test" and isinstance(parsed, dict):
                if parsed.get("exit_code", 0) != 0:
                    ok = False

            if message.effect == "edit":
                result = f"EDITED:{message.resource}"
                file_content = (
                    message.payload.get("content")
                    or message.args.get("patch")
                    or f"patched:{message.action_id}"
                )
            elif message.effect == "test":
                result = "TESTS_PASS" if ok else "TESTS_FAIL"
                file_content = None
            else:
                result = f"DONE:{message.effect}@{message.resource}"
                file_content = None

            obs: dict[str, Any] = {
                "ok": ok,
                "result": result,
                "action_id": message.action_id,
                "worker_id": message.worker_id,
                "hands_engine_id": self.hands_engine_id,
                "target": message.target,
                "via": "hands_one_per_task",
                "hands_tool": tool,
                "hands_args": {k: v for k, v in args.items() if k != "content"},
                "hands_raw": (
                    parsed if not isinstance(parsed, str) or len(parsed) < 500
                    else parsed[:500]
                ),
                "elapsed_s": elapsed,
            }
            if file_content is not None:
                obs["file_content"] = file_content
            self.history.append(obs)
            self.in_flight = 0
            self._last_free = time.perf_counter()
            return obs


# Back-compat alias used by earlier eval drafts
HandsSinglePathHost = HandsHost
