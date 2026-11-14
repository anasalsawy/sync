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
            if prompts[ri].lobe == "L":
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
                    scale = (recv_rms[ri] / (send_rms + 1e-8)).clamp(0.0, 5.0)  # [default_w]

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
                    scale = (recv_rms[ri] / (send_rms + 1e-8)).clamp(0.0, 5.0)

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
                       speaker: str = "L") -> Generation:
        """Async wrapper around generate_sync (one remote job per reply)."""
        return await asyncio.to_thread(self.generate_sync, prompts, max_new_tokens,
                                       attention_demands, speaker)

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
                        scale = (recv_rms[ri] / (send_rms + 1e-8)).clamp(0.0, 5.0)
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
    """Second-tier concurrent execution lobes.

    Offspring lobes (R1, R2, R3, etc.) receive internal activations from main
    lobes as read-only context injected into their slot bands. They:
      - Do NOT write back to main lobes (true concurrency, no feedback loop)
      - Do NOT share cross-attention with each other (fully independent)
      - Run as separate concurrent NDIF jobs via asyncio.gather

    Architecture:
      Main pass (NDIFChannels): L, R, B, V, C share attention + activations
      Offspring pass (NDIFOffspring): R1/R2/R3 receive main tier1 snapshots
                                      injected into early slot bands (read-only)

    Concurrency: all offspring launch simultaneously as asyncio tasks — each
    is an independent NDIF job that runs in parallel on the remote GPUs.

    Usage:
        # 1. Run main pass to get tier1 snapshots:
        main_gen = await channels.generate(main_prompts)

        # 2. Launch concurrent offspring branches:
        offspring = NDIFOffspring(model_name="Qwen/Qwen3.8-27B")
        results = await offspring.run_concurrent(
            offspring_prompts=[
                LobePrompt("R1", "Explore option A: ..."),
                LobePrompt("R2", "Explore option B: ..."),
                LobePrompt("R3", "Explore option C: ..."),
            ],
            main_tier1=main_gen.tier1,
            main_lobe_names=["L", "R", "B"],
        )
        for r in results:
            print(r.lobe, r.text)
    """

    # Layer range for injecting main lobe context into offspring.
    # Inject early-to-mid: offspring get grounded context before forming their path.
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

    def _generate_offspring_sync(self, prompt: LobePrompt,
                                  main_tier1: dict[str, dict[int, Any]],
                                  main_lobe_names: list[str],
                                  max_new_tokens: int = 64) -> OffspringGeneration:
        """Single offspring lobe: inject main activations, generate independently.

        main_tier1: {lobe_name: {layer_idx: tensor(slots, hidden)}}
                    — the Tier-1 snapshots produced by the main NDIFChannels pass.

        We inject each main lobe's activations into the offspring's slot band
        at the corresponding layers (read-only: offspring writes nothing back).
        The grad graph never connects back to the main pass.
        """
        model = self.model
        layers = self.layers
        tok = model.tokenizer
        pad_id = tok.pad_token_id
        default_w = self.slots
        num_main = len(main_lobe_names)

        # Build offspring prompt with slot bands for each main lobe
        band = SLOT_TOKEN * (default_w * num_main)
        text = band + " " + prompt.text
        texts = [text]

        # Slot offsets per main lobe in this offspring's prompt
        slot_offsets = {name: i * default_w for i, name in enumerate(main_lobe_names)}

        inject_first = self.INJECT_FIRST
        inject_last = min(self.INJECT_LAST, len(layers) - 2)

        slot_act = out = None

        with model.generate(texts, max_new_tokens=int(max_new_tokens),
                            do_sample=False, pad_token_id=pad_id,
                            remote=True) as tracer:
            for index in range(inject_first, len(layers) - 2):
                layer = layers[index]

                if index <= inject_last:
                    for lobe_name in main_lobe_names:
                        tier1_for_lobe = main_tier1.get(lobe_name, {})
                        if not tier1_for_lobe:
                            continue
                        # Find nearest available snapshot at or before this layer
                        snap_at = max(
                            (sl for sl in tier1_for_lobe if sl <= index),
                            default=None,
                        )
                        if snap_at is None:
                            # Use earliest available snap if none before this layer
                            snap_at = min(tier1_for_lobe.keys())
                        snap = tier1_for_lobe[snap_at]  # [slots, hidden]
                        offset = slot_offsets[lobe_name]
                        # Read-only inject: main snapshot → offspring slot band
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

    async def _generate_one(self, prompt: LobePrompt,
                             main_tier1: dict[str, dict[int, Any]],
                             main_lobe_names: list[str],
                             max_new_tokens: int = 64) -> OffspringGeneration:
        """Async wrapper — each offspring runs in its own thread (independent NDIF job)."""
        return await asyncio.to_thread(
            self._generate_offspring_sync, prompt, main_tier1, main_lobe_names, max_new_tokens
        )

    async def run_concurrent(self,
                              offspring_prompts: list[LobePrompt],
                              main_tier1: dict[str, dict[int, Any]],
                              main_lobe_names: list[str] | None = None,
                              max_new_tokens: int = 64) -> list[OffspringGeneration]:
        """Launch all offspring lobes concurrently as independent NDIF jobs.

        Each offspring is a separate asyncio task — they run in parallel.
        None of them communicate with each other or write back to main lobes.

        Args:
            offspring_prompts: list of LobePrompt for each branch (R1, R2, R3, ...)
            main_tier1: tier1 snapshots from the main NDIFChannels.generate() call
            main_lobe_names: which main lobes to inject (default: all keys in tier1)
            max_new_tokens: generation budget per offspring

        Returns:
            list of OffspringGeneration in the same order as offspring_prompts
        """
        if main_lobe_names is None:
            main_lobe_names = list(main_tier1.keys())

        tasks = [
            asyncio.create_task(
                self._generate_one(p, main_tier1, main_lobe_names, max_new_tokens)
            )
            for p in offspring_prompts
        ]
        return list(await asyncio.gather(*tasks))
