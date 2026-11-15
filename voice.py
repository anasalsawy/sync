"""L's streaming speech path with utterance IDs and immediate barge-in.

Stub TTS emits synthetic frames. Barge-in cancels the active producer,
discards queued frames for that utterance, and emits an interruption event
so clients can flush buffered playback. Measures interruption-to-last-frame
and time-to-first-frame. Real TTS latency remains unverified.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Protocol


AudioCallback = Callable[[dict], None]


class VoiceOut(Protocol):
    def speak(self, text: str) -> str: ...
    def barge_in(self) -> dict: ...
    @property
    def speaking(self) -> bool: ...
    @property
    def first_audio_at(self) -> float | None: ...
    @property
    def active_utterance_id(self) -> str | None: ...


@dataclass
class StubVoiceOut:
    on_event: AudioCallback | None = None
    frame_delay_s: float = 0.01
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _speaking: bool = False
    first_audio_at: float | None = None
    last_frame_at: float | None = None
    frames_emitted: int = 0
    frames_discarded: int = 0
    interruption_to_last_frame_s: float | None = None
    time_to_first_frame_s: float | None = None
    active_utterance_id: str | None = None
    _queued: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _speak_started: float | None = None

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._speaking

    def speak(self, text: str) -> str:
        """Start a new utterance. Returns utterance_id."""
        self.barge_in()
        utterance_id = uuid.uuid4().hex
        chunks = [c for c in text.split() if c] or [text or ""]
        self._stop.clear()
        with self._lock:
            self.active_utterance_id = utterance_id
            self._queued = list(chunks)
            self.first_audio_at = None
            self.last_frame_at = None
            self.frames_emitted = 0
            self.frames_discarded = 0
            self.interruption_to_last_frame_s = None
            self.time_to_first_frame_s = None
            self._speaking = True
            self._speak_started = time.perf_counter()

        def _run(uid: str) -> None:
            try:
                seq = 0
                while True:
                    if self._stop.is_set():
                        break
                    with self._lock:
                        if uid != self.active_utterance_id:
                            break
                        if not self._queued:
                            break
                        chunk = self._queued.pop(0)
                    now = time.perf_counter()
                    with self._lock:
                        if self.first_audio_at is None:
                            self.first_audio_at = now
                            if self._speak_started is not None:
                                self.time_to_first_frame_s = now - self._speak_started
                        self.last_frame_at = now
                        self.frames_emitted += 1
                    frame = {
                        "type": "audio",
                        "utterance_id": uid,
                        "seq": seq,
                        "text": chunk,
                        "pcm_b64": "",
                        "samples": max(80, len(chunk) * 40),
                        "at": now,
                    }
                    seq += 1
                    if self.on_event:
                        self.on_event(frame)
                    time.sleep(self.frame_delay_s)
            finally:
                discarded = 0
                with self._lock:
                    if uid == self.active_utterance_id:
                        discarded = len(self._queued)
                        self._queued.clear()
                        self.frames_discarded += discarded
                        self._speaking = False
                if self.on_event:
                    self.on_event({
                        "type": "audio_end",
                        "utterance_id": uid,
                        "discarded": discarded,
                        "at": time.perf_counter(),
                    })

        self._thread = threading.Thread(
            target=_run, args=(utterance_id,), daemon=True, name=f"voice-{utterance_id[:8]}",
        )
        self._thread.start()
        return utterance_id

    def barge_in(self) -> dict:
        """Cancel active producer, discard queued frames, emit interruption."""
        interrupt_at = time.perf_counter()
        uid = None
        discarded = 0
        last_frame = None
        with self._lock:
            uid = self.active_utterance_id
            discarded = len(self._queued)
            self._queued.clear()
            self.frames_discarded += discarded
            last_frame = self.last_frame_at
            self._speaking = False
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=0.5)
        metrics = {
            "type": "interrupted",
            "utterance_id": uid,
            "discarded_frames": discarded,
            "at": interrupt_at,
            "interruption_to_last_frame_s": (
                (interrupt_at - last_frame) if last_frame is not None else None
            ),
            "time_to_first_frame_s": self.time_to_first_frame_s,
            "client_flush": True,
        }
        with self._lock:
            self.interruption_to_last_frame_s = metrics["interruption_to_last_frame_s"]
            self.active_utterance_id = None
        if self.on_event:
            self.on_event(metrics)
            self.on_event({**metrics, "type": "barge_in"})
        return metrics

    def wait(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
