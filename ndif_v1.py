"""NDIF adapter: continuous peer channels, nothing scheduled, nothing averaged.

Every lobe runs in one batched forward pass. Every lobe carries a dedicated
slot band for every peer, and each peer's live activations are written into
that band at every layer. No means, no summaries, no truncation: each peer's
full vectors go into their own channel.

Nothing here decides when lobes share. The channels are always open and
attention decides, per layer and per head, how much to read from whom. What
emerges can be measured afterwards with ``reliance``, which silences one
channel over a band of layers and reports how much the receiver's output
moved. That number is what the lobes chose, not a plan this module imposed.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import nnsight
from nnsight import TransformersModel

from runtime import Decision, Exchange

# Channel width per peer, in token slots. Measured on Qwen3-0.6B: one slot
# cannot carry a fact (a held-out name stayed at rank 855), four carries it
# (rank 1, the receiving lobe says the word it never saw), eight destabilises.
SLOTS_PER_PEER = 4
SLOT_TOKEN = " ?"


@dataclass(frozen=True)
class LobePrompt:
    lobe: str
    text: str


def channel_prompt(text: str, peers: int, slots: int = SLOTS_PER_PEER) -> str:
    """A lobe's prompt, with a slot band in front for each peer.

    The bands sit at the front so causal attention can read them from every
    later position: a lobe can consult its peers at any point in its own
    reasoning.
    """
    return SLOT_TOKEN * (slots * peers) + " " + text


class NDIFChannels:
    """An ``Inference`` where every lobe hears every peer, continuously."""

    def __init__(self, *, model_name: str = "Qwen/Qwen3.8-27B",
                 slots: int = SLOTS_PER_PEER) -> None:
        nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
        self.model = TransformersModel(model_name, device_map="meta",
                                       task="text-generation")
        self.layers = list(self.model.model.layers)
        self.slots = slots

    def _think_sync(self, prompts):
        model = self.model
        layers = self.layers
        width = self.slots
        count = len(prompts)
        texts = [channel_prompt(item.text, count - 1, width) for item in prompts]
        first, last = 2, len(layers) - 2
        # Precomputed so the trace body references only plain values. nnsight
        # compiles this block's source and serialises what it closes over, so
        # nothing in here may touch `self`.
        order = [(receiver, position, sender)
                 for receiver in range(count)
                 for position, sender in enumerate(
                     s for s in range(count) if s != receiver)]

        with model.trace(texts, remote=True):
            for index in range(first, last):
                layer = layers[index]
                peer = layer.output[:, -width:, :].clone()
                for receiver, position, sender in order:
                    for slot in range(width):
                        layer.output[receiver, position * width + slot, :] = (
                            peer[sender, slot]
                        )
            shared = model.lm_head.output[:, -1, :].save()

        with model.trace(texts, remote=True):
            mid = layers[len(layers) // 2]
            crossing = mid.output[:, -width:, :].clone().abs().sum(dim=(1, 2)).save()

        site = f"layers.{first}-{last - 1}"
        exchanges = []
        for receiver in range(count):
            for sender in range(count):
                if sender == receiver:
                    continue
                exchanges.append(Exchange(
                    source=prompts[sender].lobe, target=prompts[receiver].lobe,
                    site=site, magnitude=float(crossing[sender]),
                ))
        for item in exchanges:
            if item.magnitude <= 0:
                raise RuntimeError(f"Nothing crossed the channel: {item}")
        logits = {prompt.lobe: shared[index]
                  for index, prompt in enumerate(prompts)}
        return Decision(exchanges=tuple(exchanges)), logits

    async def think(self, prompts: list[LobePrompt]):
        return await asyncio.to_thread(self._think_sync, prompts)

    async def reliance(self, prompts: list[LobePrompt], bands: int = 3):
        """What the lobes actually chose: per-pair reliance by depth band.

        Silences one channel across a band of layers and reports how far the
        receiver's output moves. Nothing about the sharing is scheduled, so
        this measures emergent routing, not a design.
        """
        return await asyncio.to_thread(self._reliance_sync, prompts, bands)

    def _reliance_sync(self, prompts, bands):
        model = self.model
        layers = self.layers
        width = self.slots
        count = len(prompts)
        texts = [channel_prompt(item.text, count - 1, width) for item in prompts]
        first, last = 2, len(layers) - 2
        order = [(receiver, position, sender)
                 for receiver in range(count)
                 for position, sender in enumerate(
                     s for s in range(count) if s != receiver)]

        def pass_with(mute_receiver, mute_sender, low, high):
            with model.trace(texts, remote=True):
                for index in range(first, last):
                    layer = layers[index]
                    peer = layer.output[:, -width:, :].clone()
                    for receiver, position, sender in order:
                        if (receiver == mute_receiver and sender == mute_sender
                                and low <= index < high):
                            continue
                        for slot in range(width):
                            layer.output[receiver, position * width + slot, :] = (
                                peer[sender, slot]
                            )
                out = model.lm_head.output[:, -1, :].save()
            return out

        full = pass_with(-1, -1, 0, 0)
        edges = [first + (last - first) * step // bands
                 for step in range(bands + 1)]
        spans = list(zip(edges, edges[1:]))
        table = {}
        for receiver in range(count):
            for sender in range(count):
                if sender == receiver:
                    continue
                moved = []
                for low, high in spans:
                    out = pass_with(receiver, sender, low, high)
                    moved.append(
                        (out[receiver] - full[receiver]).abs().sum().item()
                    )
                table[(prompts[receiver].lobe, prompts[sender].lobe)] = moved
        return spans, table

    async def close(self, mind) -> None:
        pass
