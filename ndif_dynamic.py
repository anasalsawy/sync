"""Dynamic emergent channel gating.

The vanilla write is unconditional: every peer's activations go into every
receiver's slots at full strength. That means the transformer's attention
mechanism is the only thing regulating relevance, and it can only do so
*after* the write happens.

Here we add a second loop: before writing, we measure how "loud" each
sender is relative to the receiver's own last-token activation magnitude.
When the sender's representation is strong compared to the receiver's
current state, we write at full strength. When it is weak (the sender has
nothing to say this layer), we scale the write down.

This creates an emergent feedback loop:
  1. Receiver's attention at layer L amplifies useful slots.
  2. Amplified slots produce stronger gradients → higher norms next layer.
  3. High norms at layer L+1 raise the gate → signal gets stronger.
  4. Weak, ignored channels decay away on their own.

No routing logic is coded. The only rule is: match the sender's energy
to the receiver's context. Everything else emerges from attention.

Additionally, Demand.attention: a lobe can explicitly request more slots
from a specific peer. The runtime widens that peer's slot band for the
next pass, giving attention more surface area to land on.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import nnsight
from nnsight import TransformersModel

from runtime import Decision, Exchange

SLOTS_PER_PEER = 4
SLOT_TOKEN = " ?"


@dataclass(frozen=True)
class LobePrompt:
    lobe: str
    text: str
    # Optional per-peer slot overrides. {"Embedder": 8} gives 8 slots to
    # Embedder's channel and SLOTS_PER_PEER to everyone else.
    slot_overrides: dict[str, int] | None = None


def channel_prompt(text: str, peer_names: list[str],
                   slots_per_peer: dict[str, int],
                   default: int = SLOTS_PER_PEER) -> str:
    """Slot bands for each peer, variable width per peer."""
    band = "".join(SLOT_TOKEN * slots_per_peer.get(name, default)
                   for name in peer_names)
    return band + " " + text


class NDIFDynamic:
    """An Inference where channel bandwidth is content-driven."""

    def __init__(self, *, model_name: str = "Qwen/Qwen3.5-122B-A10B",
                 slots: int = SLOTS_PER_PEER) -> None:
        nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
        self.model = TransformersModel(model_name, device_map="meta",
                                       task="text-generation")
        self.layers = list(self.model.model.layers)
        self.default_slots = slots

    def _think_dynamic(self, prompts: list[LobePrompt],
                       gate_strength: float = 2.0):
        """Forward pass with norm-gated channel writes.

        gate_strength: how much to let energy differences modulate the channel.
        0 = unconditional (identical to original).
        2 = soft gating (default).
        """
        model = self.model
        layers = self.layers
        count = len(prompts)
        names = [p.lobe for p in prompts]
        first, last = 2, len(layers) - 2

        # Resolve slot widths: allow each lobe to widen specific peers
        slots_map = {}  # {(receiver_idx, sender_idx): width}
        for ri, rp in enumerate(prompts):
            overrides = rp.slot_overrides or {}
            for si, sp in enumerate(prompts):
                if si == ri:
                    continue
                slots_map[(ri, si)] = overrides.get(sp.lobe, self.default_slots)

        peer_names_per_lobe = [[names[si] for si in range(count) if si != ri]
                               for ri in range(count)]
        texts = [
            channel_prompt(
                prompts[ri].text,
                peer_names_per_lobe[ri],
                prompts[ri].slot_overrides or {},
                self.default_slots,
            )
            for ri in range(count)
        ]

        # Offset where each peer's slots start for a given receiver
        # slots_offset[(ri, si)] = token position of sender si's first slot in receiver ri
        slots_offset = {}
        for ri in range(count):
            pos = 0
            for si in range(count):
                if si == ri:
                    continue
                slots_offset[(ri, si)] = pos
                pos += slots_map[(ri, si)]

        order = [
            (ri, si, slots_offset[(ri, si)], slots_map[(ri, si)])
            for ri in range(count)
            for si in range(count) if si != ri
        ]
        default_w = self.default_slots
        gs = gate_strength

        mid_idx = len(layers) // 2
        max_width = max(slots_map.values())
        with model.trace(texts, remote=True):
            for index in range(first, last):
                layer = layers[index]
                # Read each lobe's last-token activation magnitude (the "self" signal)
                recv_mag = layer.output[:, -1, :].abs().mean(dim=-1)  # shape (N,)

                for ri, si, offset, width in order:
                    # Sender's slot region (always at the END of the sender's seq)
                    peer_slots = layer.output[si, -default_w:, :].clone()
                    # Sender's energy vs receiver's self signal
                    send_mag = peer_slots.abs().mean()
                    # Gate: 1.0 when magnitudes match, >1 when sender louder,
                    #        <1 when sender quieter. Clamp to [0.1, 2.0].
                    gate = (send_mag / (recv_mag[ri] + 1e-8) * gs).clamp(0.1, 2.0)
                    for slot in range(width):
                        src_slot = slot % default_w  # wrap if wider than default
                        layer.output[ri, offset + slot, :] = (
                            gate * peer_slots[src_slot]
                        )
                # Must capture inline here, not outside the layer loop
                if index == mid_idx:
                    slot_mags = layer.output[:, :max_width, :].abs().mean(dim=-1).save()

            shared = model.lm_head.output[:, -1, :].save()

        return shared, slot_mags

    def run(self, prompts, gate_strength=2.0):
        return self._think_dynamic(prompts, gate_strength)

    async def close(self, mind) -> None:
        pass


async def demo():
    """Show gate activity across three different tasks.

    For each task, print the slot magnitudes at the mid-layer for each lobe.
    Dynamic emergent behavior = these numbers change task to task without
    any routing code.
    """
    adapter = NDIFDynamic(model_name="Qwen/Qwen3.5-122B-A10B")
    print(f"model: Qwen/Qwen3.5-122B-A10B, {len(adapter.layers)} layers")
    print(f"gating: norm-based adaptive (gate_strength=2.0)")
    print()

    tasks = [
        ("memory-heavy",
         [LobePrompt("L", "What room should I book for Thursday?"),
          LobePrompt("R", "Execute booking for Thursday."),
          LobePrompt("Embedder",
                     "Memory: Anas prefers mornings. Room 4 double-booked.")]),
        ("execution-only",
         [LobePrompt("L", "Click the submit button."),
          LobePrompt("R", "Execute: click submit."),
          LobePrompt("Embedder", "Memory: form at step 3 of 5.")]),
        ("pure-chat",
         [LobePrompt("L", "Tell me about quantum entanglement."),
          LobePrompt("R", "No action needed; pure conversation."),
          LobePrompt("Embedder", "Memory: user is a physics student.")]),
        # Demand-based: L explicitly wants more from Embedder
        ("demand: L wants Embedder x2",
         [LobePrompt("L", "What did I say I preferred last time?",
                     slot_overrides={"Embedder": 8}),
          LobePrompt("R", "No action needed."),
          LobePrompt("Embedder",
                     "Memory: Anas prefers mornings. Zurich Thursday 10am.")]),
    ]

    for label, prompts in tasks:
        print(f"── {label} ──")
        shared, slot_mags = adapter.run(prompts)
        for idx, p in enumerate(prompts):
            mag = float(slot_mags[idx].mean())
            print(f"  {p.lobe:<10} slot activity at mid-layer: {mag:.4f}")
        print()


if __name__ == "__main__":
    asyncio.run(demo())
