"""Two-tier memory for sawii/sync.

Tier 1 — in-memory ring buffer of raw peer_slots from recent turns.
         Same session only. Used for replay, not cross-turn injection.

Tier 2 — SQLite on disk. Text facts keyed by embedding similarity.
         On recall: retrieved text goes into Embedder's prompt → fresh slots.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


# ── Tier 1 ──────────────────────────────────────────────────────────────────

@dataclass
class TurnSnapshot:
    turn_id: int
    timestamp: float
    # {lobe_name: {layer_idx: tensor(slots, hidden)}}
    slots: dict[str, dict[int, torch.Tensor]]
    summary: str = ""


class WorkingMemory:
    """Ring buffer of the last N turns' peer_slots.

    Never injected into future passes — only used for same-session replay
    and inspection. Cross-turn injection would clash with positional context.
    """

    CAPTURE_LAYERS = (9, 18, 27)

    def __init__(self, maxlen: int = 20) -> None:
        self._buf: deque[TurnSnapshot] = deque(maxlen=maxlen)
        self._turn = 0

    def save(self, slots_by_lobe: dict[str, dict[int, torch.Tensor]],
             summary: str = "") -> int:
        tid = self._turn
        self._buf.append(TurnSnapshot(
            turn_id=tid,
            timestamp=time.time(),
            slots={lobe: {li: t.detach().cpu().clone()
                          for li, t in layers.items()
                          if li in self.CAPTURE_LAYERS}
                   for lobe, layers in slots_by_lobe.items()},
            summary=summary,
        ))
        self._turn += 1
        return tid

    def recent(self, n: int = 5) -> list[TurnSnapshot]:
        items = list(self._buf)
        return items[-n:]

    def get(self, turn_id: int) -> TurnSnapshot | None:
        for snap in self._buf:
            if snap.turn_id == turn_id:
                return snap
        return None

    def __len__(self) -> int:
        return len(self._buf)


# ── Tier 2 ──────────────────────────────────────────────────────────────────

class EpisodicMemory:
    """Durable text facts in SQLite, retrieved by cosine similarity.

    Storage:
      - facts table: id, timestamp, lobe_roles (JSON), text, embedding (BLOB)
      - Each row = one cross-lobe turn summary: what L said, R did, Embedder knew

    Retrieval:
      - Embed the query text, cosine-search, return top-k text facts
      - Caller feeds returned text into Embedder lobe prompt → fresh slots

    No raw tensors stored here — text only, model-agnostic.
    """

    def __init__(self, path: str | Path = "episodic.db") -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                lobe_roles TEXT,
                text      TEXT,
                embedding BLOB
            )
        """)
        self._db.commit()

    # ── write ────────────────────────────────────────────────────────────────

    def store(self, text: str, lobe_roles: dict[str, str],
              embedding: list[float]) -> int:
        """Store a cross-lobe turn summary with its search embedding."""
        cur = self._db.execute(
            "INSERT INTO facts (timestamp, lobe_roles, text, embedding) VALUES (?,?,?,?)",
            (time.time(), json.dumps(lobe_roles), text,
             _vec_to_blob(embedding)),
        )
        self._db.commit()
        return cur.lastrowid

    # ── retrieve ─────────────────────────────────────────────────────────────

    def query(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return top-k facts by cosine similarity to the query embedding."""
        q = torch.tensor(embedding, dtype=torch.float32)
        rows = self._db.execute(
            "SELECT id, timestamp, lobe_roles, text, embedding FROM facts"
        ).fetchall()
        scored = []
        for row_id, ts, roles_json, text, blob in rows:
            vec = _blob_to_vec(blob)
            sim = _cosine(q, vec)
            scored.append((sim, row_id, ts, roles_json, text))
        scored.sort(reverse=True)
        return [
            {"id": rid, "timestamp": ts,
             "lobe_roles": json.loads(roles), "text": txt, "score": float(sim)}
            for sim, rid, ts, roles, txt in scored[:top_k]
        ]

    def all_texts(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT text FROM facts").fetchall()]

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def close(self) -> None:
        self._db.close()


# ── helpers ──────────────────────────────────────────────────────────────────

def _vec_to_blob(v: list[float]) -> bytes:
    return torch.tensor(v, dtype=torch.float32).numpy().tobytes()


def _blob_to_vec(b: bytes) -> torch.Tensor:
    import numpy as np
    return torch.from_numpy(np.frombuffer(b, dtype=np.float32).copy())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ))


# ── convenience: build Embedder prompt from retrieved facts ──────────────────

def embedder_prompt_from_recall(facts: list[dict], max_facts: int = 5) -> str:
    """Format retrieved text facts as an Embedder lobe prompt.

    The Embedder lobe receives this text and generates fresh, RMS-matched
    activations for the current context — no raw tensor injection.
    """
    lines = ["Memory:"]
    for f in facts[:max_facts]:
        lines.append(f"  [{f['score']:.2f}] {f['text']}")
    return "\n".join(lines)
