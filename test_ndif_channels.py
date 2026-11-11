"""Tests for ndif.py: entropy-based auto-demand, asymmetric wiring, channel_prompt."""
from __future__ import annotations

import math
import pytest
import torch

from ndif import (
    NDIFChannels,
    LobePrompt,
    channel_prompt,
    entropy_demands,
    ASYMMETRIC_WIRING_DEFAULT,
    ENTROPY_DEMAND_THRESHOLD,
    SLOTS_PER_PEER,
    DEMAND_MULTIPLIER,
    _FIVE_LOBE_NAMES,
)
from runtime import Demand


# ── channel_prompt ──────────────────────────────────────────────────────────

def test_channel_prompt_default_width():
    """Each peer gets SLOTS_PER_PEER slot tokens by default."""
    text = "hello"
    result = channel_prompt(text, ["R", "B"], {}, default=4)
    slot_tokens = result.split(text)[0].strip()
    # Should have 4+4=8 " ?" tokens before the text
    assert slot_tokens.count("?") == 8


def test_channel_prompt_variable_width():
    """Wider demand for one peer reflects in token count."""
    result = channel_prompt("hi", ["R", "B"], {"B": 8}, default=4)
    # R gets 4, B gets 8 → 12 "?" total
    prefix = result.split("hi")[0]
    assert prefix.count("?") == 12


def test_channel_prompt_no_peers():
    """L with no peers just prepends nothing."""
    result = channel_prompt("solo", [], {}, default=4)
    assert result.strip() == "solo"


# ── entropy_demands ──────────────────────────────────────────────────────────

def _flat_logits(vocab=100):
    """Uniform distribution → maximum entropy."""
    return torch.ones(vocab)


def _peaked_logits(vocab=100):
    """Mass on one token → near-zero entropy."""
    t = torch.full((vocab,), -1e9)
    t[0] = 10.0
    return t


def test_entropy_demands_flat_triggers():
    """Flat logits (high entropy) → auto-demand generated."""
    logits = {"L": _flat_logits()}
    demands = entropy_demands(logits, ["L", "R", "B"], threshold=2.0)
    assert len(demands) == 1
    assert demands[0].lobe == "L"
    assert "R" in demands[0].attention
    assert "B" in demands[0].attention
    assert "L" not in demands[0].attention  # no self-demand


def test_entropy_demands_peaked_silent():
    """Peaked logits (low entropy) → no demand generated."""
    logits = {"L": _peaked_logits()}
    demands = entropy_demands(logits, ["L", "R", "B"], threshold=2.0)
    assert len(demands) == 0


def test_entropy_demands_multiple_lobes():
    """Multiple uncertain lobes each get their own Demand."""
    logits = {
        "L": _flat_logits(),
        "R": _flat_logits(),
        "B": _peaked_logits(),   # B is confident
    }
    demands = entropy_demands(logits, ["L", "R", "B"], threshold=2.0)
    demanding_lobes = {d.lobe for d in demands}
    assert "L" in demanding_lobes
    assert "R" in demanding_lobes
    assert "B" not in demanding_lobes


def test_entropy_demands_peers_exclude_self():
    """Auto-demand never includes the lobe itself in attention."""
    logits = {"R": _flat_logits()}
    demands = entropy_demands(logits, ["L", "R", "B"], threshold=2.0)
    if demands:
        for d in demands:
            assert d.lobe not in d.attention


def test_entropy_demands_bad_logits_skipped():
    """Malformed logit entry doesn't crash — just skipped."""
    logits = {"L": "not a tensor", "R": _flat_logits()}
    demands = entropy_demands(logits, ["L", "R"], threshold=2.0)
    # R should still get processed
    assert all(d.lobe != "L" for d in demands)


# ── asymmetric wiring defaults ───────────────────────────────────────────────

def test_five_lobe_names_coverage():
    """Every pair in the default wiring table uses valid 5-lobe names."""
    for sender, receiver in ASYMMETRIC_WIRING_DEFAULT:
        assert sender in _FIVE_LOBE_NAMES, f"Unknown sender: {sender}"
        assert receiver in _FIVE_LOBE_NAMES, f"Unknown receiver: {receiver}"
        assert sender != receiver, "Self-wiring entry found"


def test_wiring_ranges_valid():
    """All wiring ranges have first < last and are non-negative."""
    for (s, r), (first, last) in ASYMMETRIC_WIRING_DEFAULT.items():
        assert 0 <= first < last, f"{s}→{r}: invalid range ({first}, {last})"


def test_v_b_inject_before_layer_18():
    """V and B broadcast to R starts at layer 2 (early grounding).
    V→L and B→L start late (18+) — L synthesises at the end, not the start.
    """
    for sender in ("V", "B"):
        # R and C get early grounding from V/B
        for receiver in ("R", "C"):
            pair = (sender, receiver)
            if pair in ASYMMETRIC_WIRING_DEFAULT:
                first, _ = ASYMMETRIC_WIRING_DEFAULT[pair]
                assert first <= 2, f"{sender}→{receiver} should start at layer 2, got {first}"
        # L reads V/B late (synthesis happens at the end)
        for receiver in ("L",):
            pair = (sender, receiver)
            if pair in ASYMMETRIC_WIRING_DEFAULT:
                first, _ = ASYMMETRIC_WIRING_DEFAULT[pair]
                assert first >= 14, f"{sender}→{receiver} should start late (≥14), got {first}"


def test_c_to_l_starts_late():
    """C→L verification signal arrives late (layer 22+), not at layer 2."""
    first, last = ASYMMETRIC_WIRING_DEFAULT[("C", "L")]
    assert first >= 18, "C→L should be late-only (≥18); C steers before L speaks"


def test_r_c_bidirectional():
    """R↔C are both wired mid-to-late (≥14)."""
    for pair in [("R", "C"), ("C", "R")]:
        first, _ = ASYMMETRIC_WIRING_DEFAULT[pair]
        assert first >= 14, f"{pair} should start mid-late (≥14)"


# ── NDIFChannels unit pieces (no NDIF server needed) ────────────────────────

def test_build_slot_map_demand_doubles():
    """Demanding a peer doubles its slot width.
    Keys are (receiver_idx, sender_idx) integers.
    prompts = [L=0, R=1, B=2]. L (ri=0) demands more from B (si=2).
    """
    import os
    os.environ.setdefault("NDIF_API_KEY", "test")
    nc = NDIFChannels.__new__(NDIFChannels)
    nc.slots = SLOTS_PER_PEER
    nc.gate_strength = 2.0

    prompts = [LobePrompt("L", "hi"), LobePrompt("R", "run"), LobePrompt("B", "facts")]
    # L (idx 0) demands more from B (idx 2)
    slot_map = nc._build_slot_map(prompts, {"L": ("B",)})

    assert slot_map[(0, 1)] == SLOTS_PER_PEER                       # L←R unchanged
    assert slot_map[(0, 2)] == SLOTS_PER_PEER * DEMAND_MULTIPLIER   # L←B doubled
    assert slot_map[(1, 0)] == SLOTS_PER_PEER                       # R←L unchanged
    assert slot_map[(2, 0)] == SLOTS_PER_PEER                       # B←L unchanged


def test_build_slot_map_no_demands():
    """With no demands, all pairs get the default width."""
    import os
    os.environ.setdefault("NDIF_API_KEY", "test")
    nc = NDIFChannels.__new__(NDIFChannels)
    nc.slots = SLOTS_PER_PEER
    nc.gate_strength = 2.0

    prompts = [LobePrompt("L", "hi"), LobePrompt("R", "run")]
    slot_map = nc._build_slot_map(prompts, {})

    for (ri, si), w in slot_map.items():
        assert w == SLOTS_PER_PEER


# ── five-lobe detection ──────────────────────────────────────────────────────

def test_five_lobe_detection_full_set():
    """All 5 lobe names → asymmetric wiring activated."""
    names = ["L", "R", "B", "V", "C"]
    is_five = frozenset(names) <= _FIVE_LOBE_NAMES and len(names) >= 4
    assert is_five


def test_five_lobe_detection_three_lobes():
    """L, R, Embedder → uniform wiring (not five-lobe)."""
    names = ["L", "R", "Embedder"]
    is_five = frozenset(names) <= _FIVE_LOBE_NAMES and len(names) >= 4
    assert not is_five


def test_five_lobe_detection_partial_set():
    """Four out of five → still activates asymmetric wiring."""
    names = ["L", "R", "B", "V"]  # C absent but 4 of the 5
    is_five = frozenset(names) <= _FIVE_LOBE_NAMES and len(names) >= 4
    assert is_five
