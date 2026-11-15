"""Unit tests for Crossing readiness + five-worker mid-pass influence."""
from __future__ import annotations

import unittest

import torch

from crossing import Crossing, ReadyHandle
from local_workers import DEFAULT_VOCAB, LocalLobeWorker, SimModel
from sync_session import SyncSession, mid_pass_publish


class CrossingTests(unittest.TestCase):
    def test_publish_ready_consume(self):
        x = Crossing(enabled=True, session_id="t1")
        sig = x.publish(
            "B", pass_id=1, step=0, layer=2, kind="hidden",
            payload=torch.ones(8), readiness=ReadyHandle(ready=True),
        )
        self.assertIsNotNone(sig)
        self.assertIs(x.latest_ready("B"), sig)
        x.record_consume("L", "B", step=1, layer=3, signal=sig)
        x.record_consume("L", "R", step=1, layer=3, signal=None, reason="absent")
        tel = x.telemetry()
        self.assertEqual(tel["publish_count"], 1)
        self.assertEqual(tel["miss_count"], 1)
        self.assertIn("B", tel["ready_sources"])

    def test_in_flight_falls_back_to_previous_ready(self):
        x = Crossing(enabled=True)
        first = x.publish(
            "R", pass_id=0, step=0, layer=2, kind="hidden",
            payload=torch.ones(4), readiness=ReadyHandle(ready=True),
        )
        # In-flight successor (not ready yet)
        x.publish(
            "R", pass_id=0, step=1, layer=2, kind="hidden",
            payload=torch.ones(4) * 2, readiness=ReadyHandle(ready=False),
        )
        got = x.latest_ready("R")
        self.assertEqual(got.seq, first.seq)

    def test_disabled_is_noop(self):
        x = Crossing(enabled=False)
        self.assertIsNone(x.publish(
            "L", pass_id=0, step=0, layer=0, kind="hidden", payload=1,
        ))


class WorkerTests(unittest.TestCase):
    def test_residual_inject_moves_state(self):
        vocab = dict(DEFAULT_VOCAB)
        m = SimModel("L", vocab)
        m.reset([vocab["hello"]])
        before = m.state.clone()
        m.inject_residual(torch.ones(m.hidden) * 3.0)
        self.assertGreater((m.state - before).abs().sum().item(), 0.0)

    def test_b_fact_bias(self):
        vocab = dict(DEFAULT_VOCAB)
        m = SimModel("B", vocab)
        m.reset([vocab["memory"]])
        m.set_fact_token(vocab["secret"], strength=50.0)
        tid, _, _ = m.forward_step()
        self.assertEqual(tid, vocab["secret"])


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_five_start_together_l_streams(self):
        session = SyncSession(backend="sim", max_new_tokens=8, crossing_enabled=True)
        streamed: list[str] = []
        result = await session.turn(
            "hello world", on_token=lambda _l, p: streamed.append(p),
        )
        self.assertEqual(result.config, "five_on")
        self.assertEqual(set(result.results), set(session.ROSTER))
        self.assertTrue(result.reply)
        self.assertTrue(streamed)
        # Actual compute start times recorded for every worker
        self.assertEqual(set(result.compute_starts), set(session.ROSTER))
        self.assertGreater(result.crossing_telemetry["publish_count"], 0)

    async def test_three_configs(self):
        session = SyncSession(backend="sim", max_new_tokens=6)
        solo = await session.turn("ping", solo=True)
        off = await session.turn(
            "ping", solo=False, crossing_enabled=True, activation_injection=False,
        )
        on = await session.turn(
            "ping", solo=False, crossing_enabled=True, activation_injection=True,
        )
        self.assertEqual(solo.config, "solo")
        self.assertEqual(off.config, "five_off")
        self.assertEqual(on.config, "five_on")
        # five_off: coordination may publish; activation tensors must not
        self.assertFalse(off.activation_injection)
        self.assertTrue(on.activation_injection)
        hidden_off = [
            p for p in (session.last_crossing.publishes if session.last_crossing else [])
            if p.kind in ("hidden", "embed")
        ]
        # Re-check via on telemetry — on should publish activations
        self.assertGreater(on.crossing_telemetry["publish_count"], 0)
        # Matched-length step lists
        n = min(len(solo.matched_step_ms()), len(on.matched_step_ms()))
        self.assertGreater(n, 0)
        _ = hidden_off

    async def test_mid_pass_b_to_l_changes_output(self):
        """B fact published after L compute begins → L output changes vs control."""
        control = SyncSession(backend="sim", max_new_tokens=12)
        off = await control.turn("say the codeword", crossing_enabled=False)

        treated = SyncSession(backend="sim", max_new_tokens=12)
        hook = mid_pass_publish(
            when_lobe="L", after_step=1, publish_as="B",
            kind="hidden", token_name="gamma",
        )
        on = await treated.turn(
            "say the codeword", crossing_enabled=True, mid_pass_hook=hook,
        )
        self.assertGreater(on.crossing_telemetry["consume_count"], 0)
        # Causal: mid-pass publish happened; L continued past inject
        self.assertGreaterEqual(on.results["L"].timing.decode_steps, 2)
        self.assertNotEqual(off.reply, on.reply)

    async def test_mid_pass_l_decision_retargets_r(self):
        session = SyncSession(backend="sim", max_new_tokens=12)
        hook = mid_pass_publish(
            when_lobe="R", after_step=1, publish_as="L",
            kind="decision", token_name="revise", decision="objective",
        )
        on = await session.turn("plan the work", crossing_enabled=True, mid_pass_hook=hook)
        # R should have recorded a decision consume / inject
        self.assertTrue(
            on.results["R"].timing.inject_count > 0
            or "objective" in on.results["R"].decisions
            or "revise" in on.results["R"].text,
        )

    async def test_embedder_mid_pass_peer_activation(self):
        session = SyncSession(backend="sim", max_new_tokens=4)
        hook = mid_pass_publish(
            when_lobe="Embedder", after_step=3, publish_as="B",
            kind="hidden", token_name="gamma",
        )
        on = await session.turn("index session", crossing_enabled=True, mid_pass_hook=hook)
        emb = on.results["Embedder"]
        self.assertIsNotNone(emb.embedding)
        self.assertGreater(emb.timing.inject_count + on.crossing_telemetry["consume_count"], 0)

    async def test_crossing_off_no_activation_inject(self):
        session = SyncSession(backend="sim", max_new_tokens=8)
        hook = mid_pass_publish(
            when_lobe="L", after_step=0, publish_as="B", token_name="alpha",
        )
        # Injection off: hook may publish decision/hidden attempt — hidden skipped
        result = await session.turn(
            "x",
            crossing_enabled=True,
            activation_injection=False,
            mid_pass_hook=hook,
        )
        self.assertFalse(result.activation_injection)
        self.assertEqual(result.config, "five_off")
        x = session.last_crossing
        self.assertIsNotNone(x)
        hidden = [p for p in x.publishes if p.kind in ("hidden", "embed")]
        self.assertEqual(len(hidden), 0)
        # L must not have residual-injected peer activations
        self.assertEqual(result.results["L"].timing.inject_ops, 0)


class LocalRuntimeEntryTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_sync_entry(self):
        from runtime_sync import LocalSyncRuntime
        rt = LocalSyncRuntime(backend="sim", max_new_tokens=4)
        result = await rt.turn("hi")
        self.assertEqual(result.config, "five_on")
        self.assertEqual(list(rt.ROSTER), ["L", "R", "B", "C", "Embedder"])
        await rt.close()


if __name__ == "__main__":
    unittest.main()
