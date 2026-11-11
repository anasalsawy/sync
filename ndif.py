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

from runtime import Decision, Demand, Exchange

# Default channel width per peer, in token slots.
# Measured on Qwen3-0.6B: 1 slot can't carry a fact (rank 855);
# 4 carries it (rank 1); 8 destabilises.
SLOTS_PER_PEER = 4
SLOT_TOKEN = " ?"
# When a lobe requests more from a peer via Demand.attention, its slot band
# for that peer is multiplied by this factor.
DEMAND_MULTIPLIER = 2

# Entropy threshold above which a lobe auto-generates Demand.attention.
# Shannon entropy of a uniform 32000-token vocab ≈ 10.4 nats.
# A focused prediction has entropy < 2. 4.0 = model is genuinely uncertain.
ENTROPY_DEMAND_THRESHOLD = 4.0

# Default asymmetric layer ranges for the 5-lobe topology (L, R, B, V, C).
# Each entry: (sender_role, receiver_role, first_layer, last_layer_exclusive)
# Perception (V, B) inject early so R, C, L build grounded representations.
# R↔C connect mid-late so C evaluates R before L speaks.
# L reads all peers late — final synthesis before token generation.
ASYMMETRIC_WIRING_DEFAULT: dict[tuple[str, str], tuple[int, int]] = {
    # V → all: broadcast visual truth early
    ("V", "B"): (2, 18),
    ("V", "R"): (2, 18),
    ("V", "C"): (2, 27),
    ("V", "L"): (18, 62),  # L reads V late — visual grounding for speech
    # B → all: broadcast factual memory early
    ("B", "R"): (2, 18),
    ("B", "C"): (2, 27),
    ("B", "L"): (18, 62),
    ("B", "V"): (2, 18),
    # R ↔ C: mid-late sync — C evaluates R's plan before L commits
    ("R", "C"): (14, 62),
    ("C", "R"): (14, 62),  # C steers R if veto fires
    ("R", "L"): (18, 62),  # L reads R's execution state late
    ("R", "B"): (2, 18),
    ("R", "V"): (2, 18),
    # C → L: verification signal reaches L late only
    ("C", "L"): (22, 62),
    ("C", "B"): (2, 18),
    ("C", "V"): (2, 18),
    # L → all: speech intent broadcast (L can influence peers mid-late)
    ("L", "R"): (14, 62),
    ("L", "C"): (14, 62),
    ("L", "B"): (2, 18),
    ("L", "V"): (2, 18),
}

# For the simpler 3-lobe topology (L, R, Embedder), use uniform wiring.
# Asymmetric wiring kicks in automatically when lobe names match the 5-lobe set.
_FIVE_LOBE_NAMES = frozenset({"L", "R", "B", "V", "C"})


@dataclass(frozen=True)
class LobePrompt:
    lobe: str
    text: str


def entropy_demands(
    logits: dict[str, Any],
    peer_names: list[str],
    threshold: float = ENTROPY_DEMAND_THRESHOLD,
    top_k: int = 200,
) -> list[Demand]:
    """Auto-generate Demand.attention from output logit entropy.

    When a lobe's output distribution is flat (high entropy), it means the
    current slot content isn't resolving its uncertainty. We automatically
    widen the slot bands from every other peer for the next pass.

    Uses top-K entropy (k=200) rather than full-vocab entropy to avoid
    numerical noise from near-zero logits swamping the signal.

    Returns a list of Demand objects — one per lobe that exceeded the threshold.
    Pass these into the next call's attention_demands to widen channels.
    """
    demands = []
    for lobe_name, lobe_logits in logits.items():
        try:
            # top-K to avoid log(~0) noise
            topk = torch.topk(lobe_logits.float(), min(top_k, lobe_logits.shape[-1]))
            probs = torch.softmax(topk.values, dim=-1)
            entropy = -(probs * probs.log()).sum().item()
            if entropy > threshold:
                # Request more bandwidth from all peers except self
                peers = tuple(n for n in peer_names if n != lobe_name)
                demands.append(Demand(lobe=lobe_name, attention=peers))
        except Exception:
            pass  # logit capture failed — skip, don't crash
    return demands


def channel_prompt(text: str, peer_names: list[str],
                   slot_widths: dict[str, int],
                   default: int = SLOTS_PER_PEER) -> str:
    """A lobe's prompt with a variable-width slot band for each peer.

    Wider bands give attention more surface area for channels the lobe
    declared it wants more from.
    """
    band = "".join(SLOT_TOKEN * slot_widths.get(name, default)
                   for name in peer_names)
    return band + " " + text


class NDIFChannels:
    """An Inference where every lobe hears every peer, continuously.

    Channel bandwidth is content-driven: a norm gate modulates each write,
    and demand-based slot widening lets lobes request more from specific peers.
    """

    def __init__(self, *, model_name: str = "Qwen/Qwen3.8-27B",
                 slots: int = SLOTS_PER_PEER,
                 gate_strength: float = 2.0) -> None:
        nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
        self.model = TransformersModel(model_name, device_map="meta",
                                       task="text-generation")
        self.layers = list(self.model.model.layers)
        self.slots = slots
        self.gate_strength = gate_strength

    def _build_slot_map(self, prompts: list[LobePrompt],
                        attention_demands: dict[str, tuple[str, ...]]) -> dict[tuple[int, int], int]:
        """Resolve per-pair slot widths from demands."""
        names = [p.lobe for p in prompts]
        idx = {name: i for i, name in enumerate(names)}
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

    def _think_sync(self, prompts: list[LobePrompt],
                    attention_demands: dict[str, tuple[str, ...]] | None = None):
        model = self.model
        layers = self.layers
        count = len(prompts)
        names = [p.lobe for p in prompts]
        first, last = 2, len(layers) - 2
        demands = attention_demands or {}

        slots_map = self._build_slot_map(prompts, demands)
        default_w = self.slots
        gs = self.gate_strength

        # Token offsets: where each peer's slot band starts in receiver's prompt
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
        texts = [
            channel_prompt(
                prompts[ri].text,
                peer_names_per_lobe[ri],
                slot_widths_per_lobe[ri],
                default_w,
            )
            for ri in range(count)
        ]

        # Flat order list: (receiver, sender, slot_offset, slot_width, wire_first, wire_last)
        # wire_first/wire_last: the layer range this pair is active.
        # For 5-lobe topology, use asymmetric wiring. Otherwise uniform (first..last).
        is_five_lobe = frozenset(names) <= _FIVE_LOBE_NAMES and len(names) >= 4
        order = []
        for ri in range(count):
            for si in range(count):
                if si == ri:
                    continue
                pair = (names[si], names[ri])  # (sender_role, receiver_role)
                if is_five_lobe and pair in ASYMMETRIC_WIRING_DEFAULT:
                    wire_first, wire_last = ASYMMETRIC_WIRING_DEFAULT[pair]
                    # Clamp to actual model depth
                    wire_first = max(wire_first, first)
                    wire_last = min(wire_last, last)
                else:
                    wire_first, wire_last = first, last
                order.append((ri, si, slots_offset[(ri, si)], slots_map[(ri, si)],
                               wire_first, wire_last))

        max_width = max(slots_map.values())
        snap9 = snap18 = snap27 = None  # Tier-1 captures; named vars survive trace

        hdim_sqrt = math.sqrt(model.config.hidden_size)

        # Precompute demand boosts as a plain nested list (no dataclass access inside trace)
        demand_boost_grid = [
            [
                gs if (si != ri and names[si] in demands.get(names[ri], ()))
                else 1.0
                for si in range(count)
            ]
            for ri in range(count)
        ]

        with model.trace(texts, remote=True):
            for index in range(first, last):
                layer = layers[index]
                # Receiver RMS scale: the magnitude of the receiver's own reasoning
                # at this layer. We normalize peer slots TO this scale, not by it.
                # This prevents residual-norm bias: loud senders don't dominate,
                # quiet ones don't vanish. Direction is preserved; only scale changes.
                recv_rms = layer.output[:, -1, :].norm(dim=-1) / hdim_sqrt

                for ri, si, offset, width, wire_first, wire_last in order:
                    # Asymmetric wiring: skip this pair outside its active range
                    if index < wire_first or index >= wire_last:
                        continue
                    peer_slots = layer.output[si, -default_w:, :].clone()
                    demand_boost = demand_boost_grid[ri][si]
                    for slot in range(width):
                        src_slot = slot % default_w
                        # Per-slot RMS: Frobenius norm would shrink each slot
                        # by sqrt(num_slots), so compute norm per token separately.
                        send_rms = peer_slots[src_slot].norm() / hdim_sqrt
                        # Clamp to [0, 5]: prevents noise amplification when
                        # a slot is near-zero (idle lobe, early layers).
                        scale = (recv_rms[ri] / (send_rms + 1e-8)).clamp(0.0, 5.0)
                        layer.output[ri, offset + slot, :] = peer_slots[src_slot] * scale * demand_boost

                # Capture mid-layer slot activity + Tier-1 snapshots inline
                # (must be here, not outside loop — OutOfOrderError otherwise)
                if index == len(layers) // 2:
                    slot_act = layer.output[:, :max_width, :].abs().mean(dim=-1).save()
                if index == 9:
                    snap9 = layer.output[:, :max_width, :].save()
                if index == 18:
                    snap18 = layer.output[:, :max_width, :].save()
                if index == 27:
                    snap27 = layer.output[:, :max_width, :].save()

            shared = model.lm_head.output[:, -1, :].save()

        # Measure crossing magnitude for exchange reporting
        with model.trace(texts, remote=True):
            mid = layers[len(layers) // 2]
            crossing = mid.output[:, -default_w:, :].clone().abs().sum(dim=(1, 2)).save()

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
        # Build Tier-1 snapshot: {lobe_name: {layer_idx: tensor(slots, hidden)}}
        raw_snaps = {li: s for li, s in [(9, snap9), (18, snap18), (27, snap27)] if s is not None}
        tier1: dict[str, dict[int, Any]] = {
            prompt.lobe: {li: raw_snaps[li][idx, :, :] for li in raw_snaps}
            for idx, prompt in enumerate(prompts)
        }

        # Entropy-based auto-demands: measure uncertainty in output distribution.
        # High entropy → lobe didn't have enough peer context → auto-widen next pass.
        auto_demands = entropy_demands(logits, names)

        return Decision(exchanges=tuple(exchanges)), logits, slot_act, tier1, auto_demands

    async def think(self, prompts: list[LobePrompt],
                    attention_demands: dict[str, tuple[str, ...]] | None = None):
        """Returns (Decision, logits, slot_act, tier1_snapshot, auto_demands).

        tier1_snapshot: {lobe_name: {layer_idx: tensor}} — raw pre-RoPE slots
        at layers 9/18/27 for same-session Tier-1 memory. Pass to
        WorkingMemory.save(). Never inject into future passes.

        auto_demands: list[Demand] — entropy-triggered bandwidth requests.
        Any lobe whose output entropy exceeded ENTROPY_DEMAND_THRESHOLD
        appears here with attention=(<all_peers>,). Pass these into the next
        call's attention_demands to auto-widen channels. Zero-prompt, zero
        roleplay — the model's own probability distribution declares need.
        """
        return await asyncio.to_thread(self._think_sync, prompts, attention_demands)

    async def reliance(self, prompts: list[LobePrompt], bands: int = 3,
                       attention_demands: dict[str, tuple[str, ...]] | None = None):
        """Emergent routing by depth band: how much does silencing one channel move output?"""
        return await asyncio.to_thread(self._reliance_sync, prompts, bands, attention_demands)

    def _reliance_sync(self, prompts, bands, attention_demands=None):
        model = self.model
        layers = self.layers
        count = len(prompts)
        demands = attention_demands or {}
        default_w = self.slots
        gs = self.gate_strength
        first, last = 2, len(layers) - 2

        slots_map = self._build_slot_map(prompts, demands)
        slots_offset = {}
        for ri in range(count):
            pos = 0
            for si in range(count):
                if si == ri:
                    continue
                slots_offset[(ri, si)] = pos
                pos += slots_map[(ri, si)]

        names = [p.lobe for p in prompts]
        peer_names_per_lobe = [[names[si] for si in range(count) if si != ri]
                               for ri in range(count)]
        slot_widths_per_lobe = [{names[si]: slots_map[(ri, si)]
                                  for si in range(count) if si != ri}
                                 for ri in range(count)]
        texts = [
            channel_prompt(prompts[ri].text, peer_names_per_lobe[ri],
                           slot_widths_per_lobe[ri], default_w)
            for ri in range(count)
        ]
        is_five_lobe = frozenset(names) <= _FIVE_LOBE_NAMES and len(names) >= 4
        order = []
        for ri in range(count):
            for si in range(count):
                if si == ri:
                    continue
                pair = (names[si], names[ri])
                if is_five_lobe and pair in ASYMMETRIC_WIRING_DEFAULT:
                    wf, wl = ASYMMETRIC_WIRING_DEFAULT[pair]
                    wf, wl = max(wf, first), min(wl, last)
                else:
                    wf, wl = first, last
                order.append((ri, si, slots_offset[(ri, si)], slots_map[(ri, si)], wf, wl))

        hdim_sqrt = math.sqrt(model.config.hidden_size)
        demand_boost_grid = [
            [
                gs if (si != ri and names[si] in demands.get(names[ri], ()))
                else 1.0
                for si in range(count)
            ]
            for ri in range(count)
        ]

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
                        peer_slots = layer.output[si, -default_w:, :].clone()
                        demand_boost = demand_boost_grid[ri][si]
                        for slot in range(width):
                            src_slot = slot % default_w
                            send_rms = peer_slots[src_slot].norm() / hdim_sqrt
                            scale = (recv_rms[ri] / (send_rms + 1e-8)).clamp(0.0, 5.0)
                            layer.output[ri, offset + slot, :] = peer_slots[src_slot] * scale * demand_boost
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
