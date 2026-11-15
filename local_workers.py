"""Local concurrent lobe workers with activation Crossing hooks.

Each autoregressive worker (L/R/B/V/C) owns independent decode state (KV /
position / sampling). Weights may be shared. V owns live screen/browser state
shared into worker lanes. The Embedder runs encode passes with mid-pass receive
hooks. Capture → transfer → inject never blocks peers and never adds forward
passes or restarts.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from crossing import Crossing, ReadyHandle

# V owns live browser/desktop state and shares it into R worker lanes.
MAIN_WORKERS = ("L", "R", "B", "V", "C", "Embedder")
AR_WORKERS = ("L", "R", "B", "V", "C")

PUBLISH_LAYERS = (2, 4, 6)
INJECT_LAYERS = (3, 5, 7)
HIDDEN = int(os.environ.get("SYNC_LOCAL_HIDDEN", "64"))
INJECT_SCALE = float(os.environ.get("SYNC_INJECT_SCALE", "1.0"))


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class WorkerTiming:
    lobe: str
    task_started_at: float = 0.0
    compute_started_at: float = 0.0
    first_token_at: float | None = None
    finished_at: float | None = None
    forward_count: int = 0
    decode_steps: int = 0
    step_durations: list[float] = field(default_factory=list)
    inject_count: int = 0
    miss_count: int = 0
    # Injection profiling — must stay off L's critical-path budget
    inject_s: list[float] = field(default_factory=list)
    inject_ops: int = 0
    peer_wait_count: int = 0  # must remain 0: never block on peers
    skipped_transfer: int = 0  # activation not already on L's device
    extra_forward_count: int = 0  # must remain 0


@dataclass
class WorkerResult:
    lobe: str
    text: str
    token_ids: list[int]
    timing: WorkerTiming
    embedding: torch.Tensor | None = None
    decisions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulation backend
# ---------------------------------------------------------------------------

class SimModel:
    """Tiny local model: residual state + peer inject at layer boundaries."""

    def __init__(self, lobe: str, vocab: dict[str, int], hidden: int = HIDDEN) -> None:
        self.lobe = lobe
        self.vocab = vocab
        self.inv = {i: t for t, i in vocab.items()}
        self.hidden = hidden
        self.device = _device()
        g = torch.Generator(device="cpu")
        g.manual_seed(hash(lobe) & 0xFFFFFFFF)
        self.W = torch.randn(hidden, len(vocab), generator=g)
        self.state = torch.zeros(hidden, device=self.device)
        self._fact_bias: dict[int, float] = {}
        self.plan_token: int | None = None
        self.objective_token: int | None = None

    def reset(self, prompt_ids: list[int]) -> None:
        self.state = torch.zeros(self.hidden, device=self.device)
        for tid in prompt_ids:
            self.state = 0.9 * self.state
            self.state[tid % self.hidden] += 1.0

    def inject_residual(self, peer_act: torch.Tensor, scale: float = INJECT_SCALE) -> None:
        """Norm-gated write into the current residual (active decode/encode)."""
        peer = peer_act.detach().to(self.device).float().flatten()
        if peer.numel() < self.hidden:
            pad = torch.zeros(self.hidden, device=self.device)
            pad[: peer.numel()] = peer
            peer = pad
        else:
            peer = peer[: self.hidden]
        send = peer.norm() + 1e-8
        recv = self.state.norm() + 1e-8
        gate = float((recv / send).clamp(0.0, 5.0).item())
        self.state = self.state + scale * gate * peer

    def set_fact_token(self, token_id: int, strength: float = 8.0) -> None:
        self._fact_bias[token_id] = strength

    def layer_view(self, layer: int) -> torch.Tensor:
        return torch.roll(self.state, shifts=layer, dims=0).detach().clone()

    def forward_step(self) -> tuple[int, torch.Tensor, dict[int, torch.Tensor]]:
        logits = self.state @ self.W.to(self.device)
        for tid, strength in self._fact_bias.items():
            if 0 <= tid < logits.numel():
                logits[tid] = logits[tid] + strength
        if self.plan_token is not None and 0 <= self.plan_token < logits.numel():
            logits[self.plan_token] = logits[self.plan_token] + 6.0
        if self.objective_token is not None and 0 <= self.objective_token < logits.numel():
            logits[self.objective_token] = logits[self.objective_token] + 6.0
        probs = F.softmax(logits, dim=-1)
        tid = int(torch.argmax(probs).item())
        snaps = {layer: self.layer_view(layer) for layer in set(PUBLISH_LAYERS) | set(INJECT_LAYERS)}
        self.state = 0.85 * self.state
        self.state[tid % self.hidden] += 1.0
        return tid, self.state.detach().clone(), snaps


# ---------------------------------------------------------------------------
# HF shared-weights bundle
# ---------------------------------------------------------------------------

def _hf_layer_modules(model) -> list:
    """Resolve transformer block list across Llama/Qwen3.x layouts."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "layers"):
        return list(model.layers)
    return []


class HFBundle:
    """One weight copy; each worker keeps independent decode state externally.

    Parallel Sync lobes share this model under ``forward_lock``. We avoid
    ``device_map=auto`` lazy wrappers — single-GPU ``.to(device)`` so mid-pass
    hooks + Crossing inject stay valid.
    """

    def __init__(self, model_name: str, embedder_model: str | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = _device()
        self.forward_lock = threading.RLock()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # bf16 on modern NVIDIA; fp16 fallback; fp32 on CPU
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        # No device_map=auto: concurrent lobe forwards + hooks need a real module.
        self.shared = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.shared.eval()
        self.embedder = None
        if embedder_model and embedder_model != model_name:
            self.embedder = AutoModelForCausalLM.from_pretrained(
                embedder_model,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.embedder.eval()

    def model_for(self, lobe: str):
        if lobe == "Embedder" and self.embedder is not None:
            return self.embedder
        return self.shared


DEFAULT_VOCAB = {
    "<pad>": 0, "<eos>": 1, "hello": 2, "world": 3, "plan": 4, "act": 5,
    "memory": 6, "fact": 7, "ok": 8, "no": 9, "yes": 10, "secret": 11,
    "alpha": 12, "beta": 13, "gamma": 14, "verify": 15, "pass": 16,
    "fail": 17, "embed": 18, "index": 19, "user": 20, "assistant": 21,
    "idle": 22, "done": 23, "next": 24, "wait": 25, "go": 26, "stop": 27,
    "the": 28, "a": 29, "to": 30, "is": 31, "and": 32, "of": 33,
    "revise": 34, "objective": 35, "verdict": 36, "reject": 37, "accept": 38,
}


def _tokenize_sim(text: str, vocab: dict[str, int]) -> list[int]:
    toks = []
    for w in text.lower().replace("\n", " ").split():
        w = "".join(ch for ch in w if ch.isalnum() or ch in ("_", "-"))
        if not w:
            continue
        if w not in vocab:
            vocab[w] = len(vocab)
        toks.append(vocab[w])
    return toks or [vocab.get("hello", 2)]


def _make_ready(payload: torch.Tensor) -> ReadyHandle:
    """Record CUDA event after payload is on device; CPU/sim = immediate ready."""
    handle = ReadyHandle()
    if payload.is_cuda:
        event = torch.cuda.Event()
        event.record()
        handle.event = event
        handle.ready = False
    else:
        handle.ready = True
    return handle


class LocalLobeWorker:
    """One independent inference worker bound to a Crossing."""

    def __init__(
        self,
        lobe: str,
        crossing: Crossing,
        *,
        backend: str = "sim",
        hf: HFBundle | None = None,
        vocab: dict[str, int] | None = None,
        max_new_tokens: int = 32,
        speak: bool = False,
        peers: tuple[str, ...] = MAIN_WORKERS,
        on_token: Callable[[str, str], None] | None = None,
        inject_scale: float = INJECT_SCALE,
        sim_model: SimModel | None = None,
        mid_pass_hook: Callable[[str, int, "LocalLobeWorker"], None] | None = None,
    ) -> None:
        if lobe not in MAIN_WORKERS:
            raise ValueError(f"unknown lobe {lobe}")
        self.lobe = lobe
        self.crossing = crossing
        self.backend = backend
        self.hf = hf
        self.vocab = vocab if vocab is not None else dict(DEFAULT_VOCAB)
        self.max_new_tokens = max_new_tokens
        self.speak = speak and lobe == "L"
        self.peers = tuple(p for p in peers if p != lobe)
        self.on_token = on_token
        self.inject_scale = inject_scale
        self.timing = WorkerTiming(lobe=lobe)
        self._stop = threading.Event()
        self._pass_id = 0
        self.text = ""
        self.token_ids: list[int] = []
        self.embedding: torch.Tensor | None = None
        self.decisions: list[str] = []
        self.mid_pass_hook = mid_pass_hook
        self.sim = sim_model or (SimModel(lobe, self.vocab) if backend == "sim" else None)

    def request_stop(self) -> None:
        self._stop.set()

    def _consume_peers(self, step: int, layer: int) -> int:
        """Inject newest *ready* peer activation — never blocks L's token path.

        Only consumes a signal already ready on this worker's device. Misses
        use the last ready value or skip. No device sync, transfers L must
        wait for, locks that peer work holds, extra forwards, or restarts.
        """
        if not self.crossing.enabled:
            return 0
        injected = 0
        for peer in self.peers:
            if peer == self.lobe:
                continue
            # Non-blocking ready check only — never wait for peer produce.
            sig = self.crossing.latest_ready(peer)
            if sig is None:
                raw = self.crossing.latest_one(peer)
                reason = "in_flight" if raw is not None else "absent"
                self.crossing.record_consume(
                    self.lobe, peer, step=step, layer=layer, signal=None, reason=reason,
                )
                self.timing.miss_count += 1
                continue
            self.crossing.record_consume(
                self.lobe, peer, step=step, layer=layer, signal=sig,
            )
            payload = sig.payload
            if isinstance(payload, dict) and "vector" in payload:
                payload = payload["vector"]
            kind = getattr(sig, "kind", "") or ""
            if self.sim is not None and isinstance(payload, torch.Tensor):
                if not self.crossing.allows_activation_inject(kind or "hidden"):
                    continue
                # Skip if tensor is not already on our device (would force transfer)
                if payload.device != self.sim.device and str(payload.device) != str(self.sim.device):
                    try:
                        if payload.device.type != self.sim.device.type:
                            self.timing.skipped_transfer += 1
                            self.crossing.record_consume(
                                self.lobe, peer, step=step, layer=layer,
                                signal=None, reason="wrong_device",
                            )
                            continue
                    except Exception:
                        pass
                t_inj = time.perf_counter()
                self.sim.inject_residual(payload, scale=self.inject_scale)
                self.timing.inject_s.append(time.perf_counter() - t_inj)
                self.timing.inject_ops += 1
                injected += 1
                self.timing.inject_count += 1
            elif isinstance(payload, dict) and "decision" in payload:
                # Structured coordination — metadata only, no activation inject
                self.decisions.append(str(payload["decision"]))
                if self.sim is not None and "token" in payload:
                    tid = int(payload["token"])
                    if self.lobe == "R":
                        self.sim.objective_token = tid
                    if self.lobe == "L":
                        self.sim.set_fact_token(tid, strength=10.0)
                injected += 1
                self.timing.inject_count += 1
        return injected

    def _publish_snaps(
        self,
        step: int,
        snaps: dict[int, torch.Tensor],
        kind: str,
        *,
        immediate: bool = False,
    ) -> None:
        for layer, tensor in snaps.items():
            if kind != "embed" and layer not in PUBLISH_LAYERS:
                continue
            # HF publishes after the locked forward finishes — already GPU-complete.
            ready = ReadyHandle(ready=True) if immediate else _make_ready(tensor)
            sig = self.crossing.publish(
                self.lobe,
                pass_id=self._pass_id,
                step=step,
                layer=int(layer),
                kind=kind,
                payload=tensor.detach(),
                readiness=ready,
            )
            if sig is not None and (immediate or ready.event is None):
                self.crossing.mark_ready(self.lobe, seq=sig.seq)

    def run_sync(self, prompt: str, *, pass_id: int = 0) -> WorkerResult:
        self._pass_id = pass_id
        self._stop.clear()
        self.timing = WorkerTiming(
            lobe=self.lobe, task_started_at=time.perf_counter(),
        )
        self.text = ""
        self.token_ids = []
        self.embedding = None
        self.decisions = []

        if self.backend == "hf" and self.hf is not None:
            return self._run_hf(prompt)
        return self._run_sim(prompt)

    def _run_sim(self, prompt: str) -> WorkerResult:
        assert self.sim is not None
        ids = _tokenize_sim(prompt, self.vocab)
        self.sim.reset(ids)
        self.timing.compute_started_at = time.perf_counter()

        if self.lobe == "Embedder":
            return self._run_sim_embedder(prompt)

        pieces: list[str] = []
        for step in range(self.max_new_tokens):
            if self._stop.is_set():
                break
            t0 = time.perf_counter()
            # Mid-pass hook for causal tests (mutate crossing / peer state
            # after this worker's compute has already begun).
            if self.mid_pass_hook is not None:
                self.mid_pass_hook(self.lobe, step, self)
            for layer in INJECT_LAYERS:
                self._consume_peers(step=step, layer=layer)
            tid, _hidden, snaps = self.sim.forward_step()
            self.timing.forward_count += 1
            self.timing.decode_steps += 1
            self._publish_snaps(step, snaps, kind="hidden")
            # Structured decision signals from L / C for peer plan updates.
            if self.lobe in ("L", "C") and step == max(1, self.max_new_tokens // 3):
                decision = "accept" if self.lobe == "C" else "objective"
                tok_name = "accept" if self.lobe == "C" else "revise"
                token_id = self.vocab.get(tok_name, tid)
                ready = ReadyHandle(ready=True)
                self.crossing.publish(
                    self.lobe,
                    pass_id=self._pass_id,
                    step=step,
                    layer=0,
                    kind="decision",
                    payload={"decision": decision, "token": token_id},
                    readiness=ready,
                )
                self.crossing.mark_ready(self.lobe)
            tok = self.sim.inv.get(tid, str(tid))
            pieces.append(tok)
            self.token_ids.append(tid)
            if self.timing.first_token_at is None:
                self.timing.first_token_at = time.perf_counter()
            if self.speak and self.on_token:
                self.on_token(self.lobe, tok if not pieces[:-1] else " " + tok)
            self.timing.step_durations.append(time.perf_counter() - t0)
            if tid == self.vocab.get("<eos>", 1):
                break
            time.sleep(0)

        self.text = " ".join(pieces).strip()
        self.timing.finished_at = time.perf_counter()
        return WorkerResult(
            lobe=self.lobe, text=self.text, token_ids=self.token_ids,
            timing=self.timing, embedding=None, decisions=list(self.decisions),
        )

    def _run_sim_embedder(self, prompt: str) -> WorkerResult:
        """Encode with mid-pass peer receive — proves live Embedder sharing."""
        assert self.sim is not None
        pieces = []
        # Multi-layer encode: consume peers BETWEEN layer steps of one pass.
        n_layers = max(INJECT_LAYERS) + 1
        for layer in range(n_layers):
            if self.mid_pass_hook is not None:
                self.mid_pass_hook(self.lobe, layer, self)
            if layer in INJECT_LAYERS:
                self._consume_peers(step=0, layer=layer)
            # One micro-forward per encode layer (still one encode pass).
            tid, hidden, snaps = self.sim.forward_step()
            self.timing.forward_count += 1
            if layer in PUBLISH_LAYERS:
                self._publish_snaps(0, {layer: snaps[layer]}, kind="embed")
            pieces.append(self.sim.inv.get(tid, "embed"))
        self.embedding = self.sim.state.detach().cpu()
        ready = _make_ready(self.embedding)
        self.crossing.publish(
            self.lobe,
            pass_id=self._pass_id,
            step=0,
            layer=0,
            kind="index",
            payload={"vector": self.embedding, "text": prompt[:200]},
            readiness=ready,
        )
        self.crossing.mark_ready(self.lobe)
        self.text = " ".join(pieces).strip()
        self.token_ids = []
        self.timing.decode_steps = 1
        self.timing.first_token_at = time.perf_counter()
        self.timing.finished_at = time.perf_counter()
        return WorkerResult(
            lobe=self.lobe, text=self.text, token_ids=self.token_ids,
            timing=self.timing, embedding=self.embedding, decisions=list(self.decisions),
        )

    def _hf_inject_from_peers(self, h: torch.Tensor, *, step: int, layer: int) -> None:
        """Write ready peer activations into residual ``h`` (same-process HF)."""
        if not self.crossing.enabled:
            return
        for peer in self.peers:
            if peer == self.lobe:
                continue
            sig = self.crossing.latest_ready(peer)
            if sig is None:
                self.crossing.record_consume(
                    self.lobe, peer, step=step, layer=layer,
                    signal=None, reason="absent",
                )
                continue
            self.crossing.record_consume(
                self.lobe, peer, step=step, layer=layer, signal=sig,
            )
            vec = sig.payload
            if isinstance(vec, dict):
                vec = vec.get("vector")
            kind = getattr(sig, "kind", "") or "hidden"
            if not isinstance(vec, torch.Tensor):
                continue
            if not self.crossing.allows_activation_inject(kind):
                continue
            # Same-process HF: move onto activation device if needed.
            if vec.device != h.device:
                try:
                    vec = vec.to(h.device, non_blocking=True)
                except Exception:
                    self.timing.skipped_transfer += 1
                    continue
            t_inj = time.perf_counter()
            v = vec.detach().flatten().float()
            width = h.shape[-1]
            if v.numel() < width:
                v = F.pad(v, (0, width - v.numel()))
            h[:, -1, :] = h[:, -1, :] + (
                INJECT_SCALE * 0.05 * v[:width].to(dtype=h.dtype)
            )
            self.timing.inject_s.append(time.perf_counter() - t_inj)
            self.timing.inject_ops += 1
            self.timing.inject_count += 1

    def _run_hf(self, prompt: str) -> WorkerResult:
        """HF decode with per-step lock: hooks + forward + publish for Crossing."""
        assert self.hf is not None
        model = self.hf.model_for(self.lobe)
        tok = self.hf.tokenizer
        device = self.hf.device
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.timing.compute_started_at = time.perf_counter()
        layer_modules = _hf_layer_modules(model)
        hook_layers = sorted(
            i for i in set(PUBLISH_LAYERS + INJECT_LAYERS) if i < len(layer_modules)
        )

        if self.lobe == "Embedder":
            captured: dict[int, torch.Tensor] = {}

            def make_hook(idx: int):
                def _hook(_module, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if idx in INJECT_LAYERS:
                        self._hf_inject_from_peers(h, step=0, layer=idx)
                    captured[idx] = h[0, -1, :].detach()
                return _hook

            with self.hf.forward_lock:
                handles = [
                    layer_modules[idx].register_forward_hook(make_hook(idx))
                    for idx in hook_layers
                ]
                try:
                    with torch.no_grad():
                        out = model(**inputs, output_hidden_states=True)
                    self.timing.forward_count += 1
                finally:
                    for h in handles:
                        h.remove()
            hidden = out.hidden_states[-1][0].mean(dim=0).detach()
            # Keep on GPU so peer injects don't skip device mismatch.
            self.embedding = hidden
            ready = _make_ready(hidden)
            self.crossing.publish(
                self.lobe, pass_id=self._pass_id, step=0, layer=0, kind="index",
                payload={"vector": hidden, "text": prompt[:200]},
                readiness=ready,
            )
            if ready.event is None:
                self.crossing.mark_ready(self.lobe)
            self._publish_snaps(0, dict(captured), kind="embed", immediate=True)
            self.text = "[embedded]"
            self.timing.first_token_at = time.perf_counter()
            self.timing.finished_at = time.perf_counter()
            self.timing.decode_steps = 1
            return WorkerResult(
                lobe=self.lobe, text=self.text, token_ids=[],
                timing=self.timing, embedding=self.embedding.detach().cpu(),
            )

        generated = inputs["input_ids"]
        pieces: list[str] = []
        for step in range(self.max_new_tokens):
            if self._stop.is_set():
                break
            t0 = time.perf_counter()
            if self.mid_pass_hook is not None:
                self.mid_pass_hook(self.lobe, step, self)
            captured: dict[int, torch.Tensor] = {}

            def make_hook(idx: int, step_i: int = step):
                def _hook(_module, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if idx in INJECT_LAYERS:
                        self._hf_inject_from_peers(h, step=step_i, layer=idx)
                    captured[idx] = h[0, -1, :].detach()
                return _hook

            # Serialize register → forward → remove so peers interleave by step.
            with self.hf.forward_lock:
                handles = [
                    layer_modules[idx].register_forward_hook(make_hook(idx))
                    for idx in hook_layers
                ]
                try:
                    with torch.no_grad():
                        out = model(input_ids=generated)
                        next_id = int(torch.argmax(out.logits[:, -1, :], dim=-1).item())
                    self.timing.forward_count += 1
                    self.timing.decode_steps += 1
                    snaps = {k: v.detach() for k, v in captured.items()}
                finally:
                    for h in handles:
                        h.remove()

            self._publish_snaps(step, snaps, kind="hidden", immediate=True)
            generated = torch.cat(
                [generated, torch.tensor([[next_id]], device=device)], dim=1,
            )
            piece = tok.decode([next_id], skip_special_tokens=True)
            pieces.append(piece)
            self.token_ids.append(next_id)
            if self.timing.first_token_at is None:
                self.timing.first_token_at = time.perf_counter()
            if self.speak and self.on_token and piece:
                self.on_token(self.lobe, piece)
            self.timing.step_durations.append(time.perf_counter() - t0)
            if next_id == tok.eos_token_id:
                break
            # Yield so other lobes can take the lock between tokens.
            time.sleep(0)

        self.text = "".join(pieces).strip()
        self.timing.finished_at = time.perf_counter()
        return WorkerResult(
            lobe=self.lobe, text=self.text, token_ids=self.token_ids,
            timing=self.timing, embedding=None, decisions=list(self.decisions),
        )

    async def run(self, prompt: str, *, pass_id: int = 0) -> WorkerResult:
        return await asyncio.to_thread(self.run_sync, prompt, pass_id=pass_id)


def role_prompt(lobe: str, conversation: str, role_stream: str) -> str:
    if lobe == "L":
        return conversation.rstrip() + "\nassistant:"
    if lobe == "R":
        return f"{conversation}\n\nexecution stream:\n{role_stream}\nplan:"
    if lobe == "B":
        return f"{conversation}\n\nmemory stream:\n{role_stream}\nrecall:"
    if lobe == "V":
        return (
            f"{conversation}\n\nlive screen stream (keep freshest browser/"
            f"desktop state for worker lanes):\n{role_stream}\nvision:"
        )
    if lobe == "C":
        return f"{conversation}\n\nconstraints stream:\n{role_stream}\nverify:"
    if lobe == "Embedder":
        return f"index session:\n{conversation}\n{role_stream}"
    return conversation
