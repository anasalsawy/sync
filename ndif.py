"""NDIF adapter: continuous peer channels, norm-gated, demand-widened.

Every lobe runs in one batched forward pass. Every lobe carries a slot band
for every peer, and each peer's live activations are written into that band
at every layer.

No scheduling, no averaging, no truncation.

Routing is entirely emergent:

  1. At every layer, before writing a peer's activations, we measure how
     "loud" the sender is relative to the receiver's current state.  Strong
     sender → full write.  Weak sender → channel narrows.  The model's own
     activation energies decide; nothing coded plans the route.

  2. A lobe that stated attention=[...] in its last Demand gets extra slot
     width for those channels in the next pass.  More slots = more surface
     area for attention to land on = stronger potential signal.  The lobe
     declared intent; attention carries it out.

  3. Entropy-based auto-demand: after each pass, L's (and optionally other
     lobes') output logits are measured for token entropy. When entropy
     exceeds a threshold, the system automatically generates Demand.attention
     for the peers most likely to resolve the uncertainty. No prompt
     engineering — the model's own probability distribution declares need.

  4. Asymmetric layer wiring: different lobe-pairs inject across different
     layer ranges. Perception lobes (V, B) inject early (layers 2-18) so
     downstream lobes build grounded representations. Verification (C) ↔
     Execution (R) connect mid-late (14-27). Speech (L) reads all peers
     late (18-27). No persona, no roleplay — just topology.

Input convention: each lobe receives ONLY its own raw data stream.
  L: raw user utterance + dialogue turns only.
  R: raw environment telemetry (DOM, tool schemas, current state) only.
  B: raw retrieved Tier-2 records only.
  V: raw visual patches/coordinates only.
  C: raw constraint targets + current R/L state only.
No "You are X", no job descriptions. The input tokens ARE the identity.

Generation
----------
think() is ONE forward pass and yields next-token logits only. To produce a
whole reply, use generate()/generate_sync(): the channel writes happen once,
during the prefill pass, inside a single remote model.generate() call. They
land in the KV cache, so the remaining tokens are decoded server-side with the
cache and no extra round trips. One remote job per reply, not one per token.

Offspring (second-tier concurrent lobes)
-----------------------------------------
NDIFOffspring runs alongside the main lobe batch. Offspring lobes (e.g. R1,
R2, R3) receive internal activations from main lobes (read-only channel) but:
  - Do NOT write back to main lobes (no feedback loop, true concurrency)
  - Do NOT share cross-attention with each other (fully parallel)
  - Run as separate concurrent NDIF jobs via asyncio.gather

This achieves concurrent specialised execution: R1/R2/R3 explore different
branches simultaneously while L/R/B/V/C converge on the main answer.

Latency improvements (this version):
  - Slot loop vectorized: instead of `for slot in range(width)` inside the
    trace graph (O(slots x pairs x layers) graph nodes), each pair is now a
    single tensor assign — [width, hidden] in one op.
  - .save() calls reduced: was 5 per pass (snap9, snap18, snap27, crossing,
    slot_act). Now 4: crossing + slot_act at mid, snaps at _SNAP_LAYERS.
    The diagnostic crossing check no longer adds an extra save outside the loop.
  - Tokenizer length batch: replaced N individual tok(text) calls with one
    batched tok(texts) call before the trace.
  - C→L veto strengthened: C_VETO_BOOST=3.0 multiplier + wiring extended to
    layer 18 (was 22) so C shapes L earlier.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import math
import torch
import nnsight
from nnsight import TransformersModel
from transformers import AutoTokenizer

from runtime import Decision, Demand, Exchange
from vision_buffer import VisionBuffer

# Default channel width per peer, in token slots.
SLOTS_PER_PEER = 4
SLOT_TOKEN = " ?"
# When a lobe requests more from a peer via Demand.attention, its slot band
# for that peer is multiplied by this factor.
DEMAND_MULTIPLIER = 2

# Entropy threshold above which a lobe auto-generates Demand.attention.
ENTROPY_DEMAND_THRESHOLD = 4.0

# Default asymmetric layer ranges for the 5-lobe topology (L, R, B, V, C).
ASYMMETRIC_WIRING_DEFAULT: dict[tuple[str, str], tuple[int, int]] = {
    # V → all: broadcast visual truth early
    ("V", "B"): (2, 18),
    ("V", "R"): (2, 18),
    ("V", "C"): (2, 27),
    ("V", "L"): (18, 62),
    # B → all: broadcast factual memory early
    ("B", "R"): (2, 18),
    ("B", "C"): (2, 27),
    ("B", "L"): (18, 62),
    ("B", "V"): (2, 18),
    # R ↔ C: mid-late sync
    ("R", "C"): (14, 62),
    ("C", "R"): (14, 62),
    ("R", "L"): (18, 62),
    ("R", "B"): (2, 18),
    ("R", "V"): (2, 18),
    # C → L: verification signal — extended to layer 18 (was 22) + veto boost
    ("C", "L"): (18, 62),
    ("C", "B"): (2, 18),
    ("C", "V"): (2, 18),
    # L → all: speech intent broadcast
    ("L", "R"): (14, 62),
    ("L", "C"): (14, 62),
    ("L", "B"): (2, 18),
    ("L", "V"): (2, 18),
}

_FIVE_LOBE_NAMES = frozenset({"L", "R", "B", "V", "C"})

# Layers whose slot band is snapshotted for Tier-1 memory.
_SNAP_LAYERS = (9, 18, 27)

# C→L veto boost: when C is active its channel into L is amplified.
# Prevents L's generation momentum from overriding verified facts.
C_VETO_BOOST = 3.0


@dataclass(frozen=True)
class LobePrompt:
    lobe: str
    text: str


@dataclass
class Generation:
    """Result of generate(): everything think() returns, plus L's reply."""
    decision: Decision
    logits: dict[str, Any]
    slot_act: Any
    tier1: dict[str, dict[int, Any]]
    auto_demands: list[Demand]
    text: str
    token_ids: list[int] = field(default_factory=list)


@dataclass
class OffspringGeneration:
    """Result from one offspring lobe pass."""
    lobe: str
    text: str
    token_ids: list[int] = field(default_factory=list)
    slot_act: Any = None


@dataclass
class OffspringState:
    """Live state shared from an offspring lobe back to the main lobes.

    Written by the offspring after each pass. Main lobes read this at the top
    of their next generate() call and inject it into the slot band for the
    matching lobe name so all main lobes can see what every offspring is doing
    in real time — no polling, no waiting for the offspring to finish.
    """
    lobe: str                        # e.g. "R1", "R2"
    text_so_far: str = ""            # partial decoded text this pass
    tier1: dict[int, Any] = field(default_factory=dict)   # latest activations
    done: bool = False               # True when offspring has finished
    seq: int = 0                     # monotonic update counter


class SharedState:
    """Central shared-state hub: main↔offspring bidirectional live feed.

    Main lobes write their tier1 here after each pass.
    Offspring write their OffspringState here after each step.
    Everyone reads everyone else — the one-shared-mind rule holds.

    Thread-safe for asyncio: all mutations are single assignments.
    """

    def __init__(self) -> None:
        # main lobe tier1: {lobe_name: {layer_idx: tensor}}
        self._main: dict[str, dict[int, Any]] = {}
        # offspring state: {lobe_name: OffspringState}
        self._offspring: dict[str, OffspringState] = {}
        # vision buffer: live V frames for fast screen-state sharing
        self.vision: VisionBuffer = VisionBuffer()

    # ---- main lobe writes ------------------------------------------------

    def update_main(self, tier1: dict[str, dict[int, Any]]) -> None:
        """Called by NDIFChannels after each generate(). Pushes V to vision buffer."""
        self._main.update(tier1)
        if "V" in tier1:
            self.vision.push(tier1["V"])

    # ---- offspring writes ------------------------------------------------

    def update_offspring(self, state: OffspringState) -> None:
        self._offspring[state.lobe] = state

    def mark_done(self, lobe: str) -> None:
        s = self._offspring.get(lobe)
        if s is not None:
            self._offspring[lobe] = OffspringState(
                lobe=s.lobe, text_so_far=s.text_so_far,
                tier1=s.tier1, done=True, seq=s.seq + 1,
            )

    # ---- reads -----------------------------------------------------------

    def main_tier1(self, names: list[str] | None = None) -> dict[str, dict[int, Any]]:
        """Current main lobe activations, optionally filtered to `names`."""
        if names is None:
            return dict(self._main)
        return {n: self._main[n] for n in names if n in self._main}

    def offspring_states(self) -> dict[str, OffspringState]:
        return dict(self._offspring)

    def active_offspring(self) -> list[str]:
        return [n for n, s in self._offspring.items() if not s.done]

    def latest_vision(self) -> dict[int, Any] | None:
        return self.vision.latest()


@dataclass
class OffspringHandle:
    """Live handle to a spawned offspring task.

    Returned by NDIFOffspring.spawn(). The main loop uses this to:
    - check if the offspring is still running (.task.done())
    - read its latest partial output (.state_ref in SharedState)
    - cancel it if a veto fires or the task is superseded
    """
    lobe: str
    task: asyncio.Task


def entropy_demands(
    logits: dict[str, Any],
    peer_names: list[str],
    threshold: float = ENTROPY_DEMAND_THRESHOLD,
    top_k: int = 200,
) -> list[Demand]:
    """Auto-generate Demand.attention from output logit entropy."""
    demands = []
    for lobe_name, lobe_logits in logits.items():
        try:
            topk = torch.topk(lobe_logits.float(), min(top_k, lobe_logits.shape[-1]))
            probs = torch.softmax(topk.values, dim=-1)
            entropy = -(probs * probs.log()).sum().item()
            if entropy > threshold:
                peers = tuple(n for n in peer_names if n != lobe_name)
                demands.append(Demand(lobe=lobe_name, attention=peers))
        except Exception:
            pass
    return demands


def channel_prompt(text: str, peer_names: list[str],
                   slot_widths: dict[str, int],
                   default: int = SLOTS_PER_PEER) -> str:
    band = "".join(SLOT_TOKEN * slot_widths.get(name, default)
                   for name in peer_names)
    return band + " " + text


class NDIFChannels:
    """An Inference where every lobe hears every peer, continuously.

    Channel bandwidth is content-driven: a norm gate modulates each write,
    and demand-based slot widening lets lobes request more from specific peers.

    Key optimizations vs original:
    - Slot writes are vectorized: one tensor assign per pair per layer
      instead of a Python loop over individual slots inside the trace graph.
    - .save() calls: 4 per pass (down from 5) — crossing + slot_act at mid,
      snaps at _SNAP_LAYERS. No diagnostic-only saves in the hot path.
    - Tokenizer length: one batched call before the trace instead of N calls.
    - C→L veto: C_VETO_BOOST=3.0 and wiring from layer 18 (was 22).
    """

    def __init__(self, *, model_name: str = "Qwen/Qwen3.8-27B",
                 slots: int = SLOTS_PER_PEER,
                 gate_strength: float = 2.0) -> None:
        nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TransformersModel(model_name, tokenizer=tokenizer,
                                       device_map="meta",
                                       task="text-generation")
        self.layers = list(self.model.model.layers)
        self.slots = slots
        self.gate_strength = gate_strength

        tok = self.model.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    def _build_slot_map(self, prompts: list[LobePrompt],
                        attention_demands: dict[str, tuple[str, ...]]) -> dict[tuple[int, int], int]:
        result = {}
        for ri, rp in enumerate(prompts):
            wanted = set(attention_demands.get(rp.lobe, ()))
            for si, sp in enumerate(prompts):
                if si == ri:
                    continue
                result[(ri, si)] = (
                    self.slots * DEMAND_MULTIPLIER if sp.lobe in wanted
                    else self.slots
                )
        return result

    def _prepare(self, prompts: list[LobePrompt],
                 attention_demands: dict[str, tuple[str, ...]] | None) -> dict[str, Any]:
        """Everything that is plain Python and must exist before the trace."""
        model = self.model
        layers = self.layers
        count = len(prompts)
        names = [p.lobe for p in prompts]
        first, last = 2, len(layers) - 2
        demands = attention_demands or {}
        default_w = self.slots
        gs = self.gate_strength

        slots_map = self._build_slot_map(prompts, demands)

        slots_offset: dict[tuple[int, int], int] = {}
        for ri in range(count):
            pos = 0
            for si in range(count):
                if si == ri:
                    continue
                slots_offset[(ri, si)] = pos
                pos += slots_map[(ri, si)]

        peer_names_per_lobe = [[names[si] for si in range(count) if si != ri]
                               for ri in range(count)]
        slot_widths_per_lobe = [{names[si]: slots_map[(ri, si)]
                                  for si in range(count) if si != ri}
                                 for ri in range(count)]
        tok = model.tokenizer

        texts = []
        for ri in range(count):
            text = prompts[ri].text
            # All lobes need chat template so the model knows when to generate.
            # This is structural formatting only — not a system prompt, no role labels.
            text = text.removesuffix("\\nassistant:")
            text = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(channel_prompt(
                text,
                peer_names_per_lobe[ri],
                slot_widths_per_lobe[ri],
                default_w,
            ))

        # One batched tokenize call to get pad lengths — avoids N individual calls
        encoded = tok(texts, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"]
        longest = input_ids.shape[1]
        pad_id = tok.pad_token_id
        lens = [(row != pad_id).sum().item() for row in input_ids]
        pad_of = [longest - n for n in lens]

        is_five_lobe = frozenset(names) <= _FIVE_LOBE_NAMES and len(names) >= 4
        order = []
        for ri in range(count):
            for si in range(count):
                if si == ri:
                    continue
                pair = (names[si], names[ri])
                if is_five_lobe and pair in ASYMMETRIC_WIRING_DEFAULT:
                    wire_first, wire_last = ASYMMETRIC_WIRING_DEFAULT[pair]
                    wire_first = max(wire_first, first)
                    wire_last = min(wire_last, last)
                else:
                    wire_first, wire_last = first, last
                order.append((ri, si, pad_of[ri] + slots_offset[(ri, si)],
                              slots_map[(ri, si)], wire_first, wire_last))

        max_width = max(slots_map.values())

        row_idx = [[r] for r in range(count)]
        slot_idx = [[pad_of[r] + k for k in range(max_width)] for r in range(count)]

        hdim_sqrt = math.sqrt(model.config.hidden_size)

        demand_boost_grid = [
            [
                gs if (si != ri and names[si] in demands.get(names[ri], ()))
                else 1.0
                for si in range(count)
            ]
            for ri in range(count)
        ]

        return dict(
            model=model, layers=layers, count=count, names=names,
            first=first, last=last, default_w=default_w, texts=texts,
            longest=longest, pad_of=pad_of, order=order, max_width=max_width,
            row_idx=row_idx, slot_idx=slot_idx, hdim_sqrt=hdim_sqrt,
            demand_boost_grid=demand_boost_grid, is_five_lobe=is_five_lobe,
        )

    def _finish(self, prompts: list[LobePrompt], P: dict[str, Any], shared,
                crossing, slot_act, snaps: dict[int, Any]):
        count = P["count"]
        names = P["names"]
        first, last = P["first"], P["last"]

        site = f"layers.{first}-{last - 1}"
        exchanges = []
        for ri in range(count):
            for si in range(count):
                if si == ri:
                    continue
                exchanges.append(Exchange(
                    source=prompts[si].lobe, target=prompts[ri].lobe,
                    site=site, magnitude=float(crossing[si]),
                ))
        for item in exchanges:
            if item.magnitude <= 0:
                raise RuntimeError(f"Nothing crossed the channel: {item}")

        logits = {prompt.lobe: shared[index]
                  for index, prompt in enumerate(prompts)}
        raw_snaps = {li: s for li, s in snaps.items() if s is not None}
        tier1: dict[str, dict[int, Any]] = {
            prompt.lobe: {li: raw_snaps[li][idx, :, :] for li in raw_snaps}
            for idx, prompt in enumerate(prompts)
        }

        auto_demands = entropy_demands(logits, names)

        return Decision(exchanges=tuple(exchanges)), logits, slot_act, tier1, auto_demands

    # ------------------------------------------------------------------
    # One forward pass: next-token logits only.
    # ------------------------------------------------------------------
    def _think_sync(self, prompts: list[LobePrompt],
                    attention_demands: dict[str, tuple[str, ...]] | None = None):
        P = self._prepare(prompts, attention_demands)
        model, layers = P["model"], P["layers"]
        first, last = P["first"], P["last"]
        default_w, texts = P["default_w"], P["texts"]
        order, max_width = P["order"], P["max_width"]
        row_idx, slot_idx = P["row_idx"], P["slot_idx"]
        hdim_sqrt = P["hdim_sqrt"]
        demand_boost_grid = P["demand_boost_grid"]
        names = P["names"]
        mid = len(layers) // 2

        snap_set = set(_SNAP_LAYERS)
        snaps: dict[int, Any] = {}
        slot_act = crossing = shared = None

        with model.trace(texts, remote=True):
            for index in range(first, last):
                layer = layers[index]
                recv_rms = layer.output[:, -1, :].norm(dim=-1) / hdim_sqrt

                if index == mid:
                    crossing = layer.output[:, -default_w:, :].abs().sum(dim=(1, 2)).save()

                # Vectorized writes: one tensor op per pair (no inner slot loop)
                for ri, si, offset, width, wire_first, wire_last in order:
                    if index < wire_first or index >= wire_last:
                        continue

                    peer_slots = layer.output[si, -default_w:, :]   # [default_w, H]
                    send_rms = peer_slots.norm(dim=-1) / hdim_sqrt   # [default_w]
                    scale = (recv_rms[ri].to(send_rms.device) / (send_rms + 1e-8)).clamp(0.0, 5.0)  # [default_w]

                    boost = demand_boost_grid[ri][si]
                    if names[si] == "C" and names[ri] == "L":
                        boost = boost * C_VETO_BOOST

                    scaled = peer_slots * scale.unsqueeze(-1) * boost  # [default_w, H]

                    if width == default_w:
                        layer.output[ri, offset:offset + width, :] = scaled
                    else:
                        # Demand-widened: tile to fill wider band
                        reps = (width + default_w - 1) // default_w
                        tiled = scaled.repeat(reps, 1)[:width]
                        layer.output[ri, offset:offset + width, :] = tiled

                if index == mid:
                    slot_act = layer.output[row_idx, slot_idx, :].abs().mean(dim=-1).save()
                if index in snap_set:
                    snaps[index] = layer.output[row_idx, slot_idx, :].save()

            shared = model.lm_head.output[:, -1, :].save()

        return self._finish(prompts, P, shared, crossing, slot_act, snaps)

    # ------------------------------------------------------------------
    # Full reply: ONE remote generate() call.
    # ------------------------------------------------------------------
    def generate_sync(self, prompts: list[LobePrompt], max_new_tokens: int = 64,
                      attention_demands: dict[str, tuple[str, ...]] | None = None,
                      speaker: str = "L") -> Generation:
        P = self._prepare(prompts, attention_demands)
        model, layers = P["model"], P["layers"]
        names, longest = P["names"], P["longest"]
        first, last = P["first"], P["last"]
        default_w, texts = P["default_w"], P["texts"]
        order, max_width = P["order"], P["max_width"]
        row_idx, slot_idx = P["row_idx"], P["slot_idx"]
        hdim_sqrt = P["hdim_sqrt"]
        demand_boost_grid = P["demand_boost_grid"]
        mid = len(layers) // 2
        tok = model.tokenizer
        pad_id = tok.pad_token_id

        snap_set = set(_SNAP_LAYERS)
        snaps: dict[int, Any] = {}
        slot_act = crossing = shared = out = None

        with model.generate(texts, max_new_tokens=int(max_new_tokens),
                            do_sample=False, pad_token_id=pad_id,
                            remote=True) as tracer:
            for index in range(first, last):
                layer = layers[index]
                recv_rms = layer.output[:, -1, :].norm(dim=-1) / hdim_sqrt

                if index == mid:
                    crossing = layer.output[:, -default_w:, :].abs().sum(dim=(1, 2)).save()

                for ri, si, offset, width, wire_first, wire_last in order:
                    if index < wire_first or index >= wire_last:
                        continue

                    peer_slots = layer.output[si, -default_w:, :]
                    send_rms = peer_slots.norm(dim=-1) / hdim_sqrt
                    scale = (recv_rms[ri].to(send_rms.device) / (send_rms + 1e-8)).clamp(0.0, 5.0)

                    boost = demand_boost_grid[ri][si]
                    if names[si] == "C" and names[ri] == "L":
                        boost = boost * C_VETO_BOOST

                    scaled = peer_slots * scale.unsqueeze(-1) * boost

                    if width == default_w:
                        layer.output[ri, offset:offset + width, :] = scaled
                    else:
                        reps = (width + default_w - 1) // default_w
                        tiled = scaled.repeat(reps, 1)[:width]
                        layer.output[ri, offset:offset + width, :] = tiled

                if index == mid:
                    slot_act = layer.output[row_idx, slot_idx, :].abs().mean(dim=-1).save()
                if index in snap_set:
                    snaps[index] = layer.output[row_idx, slot_idx, :].save()

            shared = model.lm_head.output[:, -1, :].save()
            out = (tracer.result if hasattr(tracer, "result")
                   else model.generator.output).save()

        decision, logits, slot_act, tier1, auto_demands = self._finish(
            prompts, P, shared, crossing, slot_act, snaps,
        )

        row = names.index(speaker) if speaker in names else 0
        sequence = out[row].tolist()
        new_ids = sequence[longest:] if len(sequence) > longest else sequence
        stop_ids = {tok.eos_token_id}
        for special in ("<|im_end|>", "<|endoftext|>"):
            tid = tok.convert_tokens_to_ids(special)
            if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                stop_ids.add(tid)
        trimmed: list[int] = []
        for tid in new_ids:
            if tid in stop_ids:
                break
            trimmed.append(tid)
        text = tok.decode(trimmed, skip_special_tokens=True)

        return Generation(decision=decision, logits=logits, slot_act=slot_act,
                          tier1=tier1, auto_demands=auto_demands,
                          text=text, token_ids=trimmed)

    async def think(self, prompts: list[LobePrompt],
                    attention_demands: dict[str, tuple[str, ...]] | None = None):
        """Returns (Decision, logits, slot_act, tier1_snapshot, auto_demands)."""
        return await asyncio.to_thread(self._think_sync, prompts, attention_demands)

    async def generate(self, prompts: list[LobePrompt], max_new_tokens: int = 64,
                       attention_demands: dict[str, tuple[str, ...]] | None = None,
                       speaker: str = "L",
                       shared: SharedState | None = None) -> Generation:
        """Async wrapper around generate_sync (one remote job per reply).

        If `shared` is provided, tier1 snapshots are pushed to SharedState
        after the pass so offspring and V's vision buffer stay current.
        """
        result = await asyncio.to_thread(self.generate_sync, prompts, max_new_tokens,
                                         attention_demands, speaker)
        if shared is not None:
            shared.update_main(result.tier1)
        return result

    async def reliance(self, prompts: list[LobePrompt], bands: int = 3,
                       attention_demands: dict[str, tuple[str, ...]] | None = None):
        """Emergent routing by depth band: how much does silencing one channel move output?"""
        return await asyncio.to_thread(self._reliance_sync, prompts, bands, attention_demands)

    def _reliance_sync(self, prompts, bands, attention_demands=None):
        P = self._prepare(prompts, attention_demands)
        model, layers = P["model"], P["layers"]
        count = P["count"]
        names = P["names"]
        first, last = P["first"], P["last"]
        default_w, texts = P["default_w"], P["texts"]
        order = P["order"]
        hdim_sqrt = P["hdim_sqrt"]
        demand_boost_grid = P["demand_boost_grid"]

        def pass_with(mute_ri, mute_si, low, high):
            with model.trace(texts, remote=True):
                for index in range(first, last):
                    layer = layers[index]
                    recv_rms = layer.output[:, -1, :].norm(dim=-1) / hdim_sqrt
                    for ri, si, offset, width, wire_first, wire_last in order:
                        if index < wire_first or index >= wire_last:
                            continue
                        if ri == mute_ri and si == mute_si and low <= index < high:
                            continue
                        peer_slots = layer.output[si, -default_w:, :]
                        send_rms = peer_slots.norm(dim=-1) / hdim_sqrt
                        scale = (recv_rms[ri].to(send_rms.device) / (send_rms + 1e-8)).clamp(0.0, 5.0)
                        boost = demand_boost_grid[ri][si]
                        if names[si] == "C" and names[ri] == "L":
                            boost = boost * C_VETO_BOOST
                        scaled = peer_slots * scale.unsqueeze(-1) * boost
                        if width == default_w:
                            layer.output[ri, offset:offset + width, :] = scaled
                        else:
                            reps = (width + default_w - 1) // default_w
                            tiled = scaled.repeat(reps, 1)[:width]
                            layer.output[ri, offset:offset + width, :] = tiled
                out = model.lm_head.output[:, -1, :].save()
            return out

        full = pass_with(-1, -1, 0, 0)
        edges = [first + (last - first) * step // bands for step in range(bands + 1)]
        spans = list(zip(edges, edges[1:]))
        table = {}
        for ri in range(count):
            for si in range(count):
                if si == ri:
                    continue
                moved = []
                for low, high in spans:
                    out = pass_with(ri, si, low, high)
                    moved.append((out[ri] - full[ri]).abs().sum().item())
                table[(prompts[ri].lobe, prompts[si].lobe)] = moved
        return spans, table

    async def close(self, mind) -> None:
        pass


# ===========================================================================
# NDIFOffspring — second-tier concurrent lobes
# ===========================================================================

class NDIFOffspring:
    """Second-tier concurrent execution lobes with live bidirectional sharing.

    Offspring lobes (R1, R2, R3, ...) are spawned on-demand by the main lobes
    via spawn(). The number of lanes is not fixed — zero, one, or ten depending
    on what the task needs. Lanes are killed when done or when C vetoes them.

    Information flow (all live, no polling):
      Main → offspring:  tier1 snapshots injected into offspring slot bands
                         at every layer within INJECT_FIRST..INJECT_LAST.
      V → offspring:     VisionBuffer.latest() checked at each inject layer —
                         if V finished a new screen capture, offspring picks it
                         up on the very next layer without waiting.
      Offspring → main:  OffspringState written to SharedState after each pass;
                         main lobes read it at the top of their next generate().

    Spawn-on-demand:
      spawn(prompt, shared) → OffspringHandle
      The handle holds the asyncio.Task so the caller can cancel it, and the
      lobe name so SharedState can be queried for its partial output.

    Design rule: offspring never modify main lobe activations during the main
    forward pass. The feedback channel is SharedState (between passes), not
    inside the trace graph. This keeps each pass deterministic and the main
    mind fully in control.
    """

    INJECT_FIRST = 2
    INJECT_LAST = 32

    def __init__(self, *, model_name: str = "Qwen/Qwen3.8-27B",
                 slots: int = SLOTS_PER_PEER) -> None:
        nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TransformersModel(model_name, tokenizer=tokenizer,
                                       device_map="meta",
                                       task="text-generation")
        self.layers = list(self.model.model.layers)
        self.slots = slots

        tok = self.model.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    # ------------------------------------------------------------------
    # Core: one offspring pass with live state injection
    # ------------------------------------------------------------------

    def _generate_offspring_sync(
        self,
        prompt: LobePrompt,
        shared: "SharedState",
        main_lobe_names: list[str],
        max_new_tokens: int = 64,
    ) -> OffspringGeneration:
        """Single offspring lobe: inject current SharedState, generate, write back.

        At each inject layer we pull the freshest available snapshot:
          - For V: VisionBuffer.latest() — updated whenever V completes a pass.
            This is what makes browser/computer use fast: a new screen frame
            lands in the buffer mid-offspring-pass and is picked up immediately.
          - For other main lobes: their most recent tier1 snapshot from SharedState.

        After the pass, OffspringState is written back so main lobes see this
        offspring's partial output on their next pass.
        """
        model = self.model
        layers = self.layers
        tok = model.tokenizer
        pad_id = tok.pad_token_id
        default_w = self.slots
        num_main = len(main_lobe_names)

        band = SLOT_TOKEN * (default_w * num_main)
        raw = tok.apply_chat_template(
            [{"role": "user", "content": prompt.text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        text = band + " " + raw
        texts = [text]

        slot_offsets = {name: i * default_w for i, name in enumerate(main_lobe_names)}
        inject_first = self.INJECT_FIRST
        inject_last = min(self.INJECT_LAST, len(layers) - 2)

        # Snapshot all main tier1 once before entering the trace — avoids
        # reading SharedState inside the graph (not safe with nnsight proxies).
        # V is the exception: we re-read the vision buffer at each layer step
        # to pick up any new frame that arrived while we were running.
        main_snap = shared.main_tier1(main_lobe_names)

        slot_act = out = None

        with model.generate(texts, max_new_tokens=int(max_new_tokens),
                            do_sample=False, pad_token_id=pad_id,
                            remote=True) as tracer:
            for index in range(inject_first, len(layers) - 2):
                layer = layers[index]

                if index <= inject_last:
                    for lobe_name in main_lobe_names:
                        # V: always use the absolute latest vision frame
                        if lobe_name == "V":
                            snap_dict = shared.latest_vision()
                        else:
                            snap_dict = main_snap.get(lobe_name)
                        if not snap_dict:
                            continue

                        snap_at = max(
                            (sl for sl in snap_dict if sl <= index),
                            default=min(snap_dict.keys()),
                        )
                        snap = snap_dict[snap_at]
                        offset = slot_offsets[lobe_name]
                        layer.output[0, offset:offset + default_w, :] = snap[:default_w, :]

            mid = len(layers) // 2
            slot_act = layers[mid].output[0, :default_w * num_main, :].abs().mean(dim=-1).save()
            out = (tracer.result if hasattr(tracer, "result")
                   else model.generator.output).save()

        sequence = out[0].tolist()
        prompt_len = len(tok(text)["input_ids"])
        new_ids = sequence[prompt_len:] if len(sequence) > prompt_len else sequence
        stop_ids = {tok.eos_token_id}
        for special in ("<|im_end|>", "<|endoftext|>"):
            tid = tok.convert_tokens_to_ids(special)
            if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
                stop_ids.add(tid)
        trimmed = []
        for tid in new_ids:
            if tid in stop_ids:
                break
            trimmed.append(tid)
        text_out = tok.decode(trimmed, skip_special_tokens=True)

        return OffspringGeneration(lobe=prompt.lobe, text=text_out,
                                   token_ids=trimmed, slot_act=slot_act)

    # ------------------------------------------------------------------
    # Spawn-on-demand: returns a live handle, not a future result
    # ------------------------------------------------------------------

    async def _run_and_report(
        self,
        prompt: LobePrompt,
        shared: "SharedState",
        main_lobe_names: list[str],
        max_new_tokens: int,
    ) -> OffspringGeneration:
        """Run the offspring pass and push its result into SharedState."""
        try:
            gen = await asyncio.to_thread(
                self._generate_offspring_sync,
                prompt, shared, main_lobe_names, max_new_tokens,
            )
            # Write final state back so main lobes see the completed output
            prev_seq = 0
            existing = shared.offspring_states().get(prompt.lobe)
            if existing is not None:
                prev_seq = existing.seq
            shared.update_offspring(OffspringState(
                lobe=prompt.lobe,
                text_so_far=gen.text,
                tier1={},
                done=True,
                seq=prev_seq + 1,
            ))
            return gen
        except asyncio.CancelledError:
            shared.mark_done(prompt.lobe)
            raise

    def spawn(
        self,
        prompt: LobePrompt,
        shared: "SharedState",
        main_lobe_names: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> OffspringHandle:
        """Spawn one offspring lane on-demand. Returns immediately with a handle.

        The handle holds the asyncio.Task so the caller can:
          - await handle.task  to get the OffspringGeneration when done
          - handle.task.cancel()  to kill it (C veto, task superseded, etc.)
          - check handle.task.done()  to see if it finished

        SharedState is updated both ways:
          - offspring reads latest main/V state at every inject layer
          - offspring writes its final text back when done

        Args:
            prompt: the offspring's identity and goal (lobe="R1", text="...")
            shared: the SharedState hub connecting all lobes
            main_lobe_names: which main lobes to inject (default: all in shared)
            max_new_tokens: generation budget

        Returns:
            OffspringHandle with .lobe and .task
        """
        if main_lobe_names is None:
            main_lobe_names = list(shared.main_tier1().keys())

        # Register as active immediately so main lobes know it exists
        shared.update_offspring(OffspringState(lobe=prompt.lobe, done=False))

        task = asyncio.create_task(
            self._run_and_report(prompt, shared, main_lobe_names, max_new_tokens),
            name=f"offspring-{prompt.lobe}",
        )
        return OffspringHandle(lobe=prompt.lobe, task=task)

    # ------------------------------------------------------------------
    # Convenience: spawn N lanes at once and wait for all
    # ------------------------------------------------------------------

    async def run_concurrent(
        self,
        offspring_prompts: list[LobePrompt],
        shared: "SharedState",
        main_lobe_names: list[str] | None = None,
        max_new_tokens: int = 64,
    ) -> list[OffspringGeneration]:
        """Spawn all offspring concurrently and wait for all to finish.

        Use spawn() instead when you want non-blocking handles (e.g. to cancel
        individual lanes mid-flight or read partial results).
        """
        handles = [
            self.spawn(p, shared, main_lobe_names, max_new_tokens)
            for p in offspring_prompts
        ]
        return list(await asyncio.gather(*(h.task for h in handles)))
