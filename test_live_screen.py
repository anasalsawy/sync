"""V live screen bus: shared into R lanes; observe/screenshot skipped."""
from __future__ import annotations

import asyncio
import unittest

from action_runtime import ActionProposal, execute_action
from batch_r import _task_prompt
from live_screen import LiveScreenBus
from local_workers import MAIN_WORKERS, role_prompt
from r_coordinator import RCoordinator
from task_record import SharedTaskRecord


class LiveScreenTests(unittest.TestCase):
    def test_v_in_roster(self):
        self.assertIn("V", MAIN_WORKERS)
        prompt = role_prompt("V", "user: hi", "live screen: demo")
        self.assertIn("live screen", prompt)

    def test_worker_prompt_includes_v_and_skip(self):
        bus = LiveScreenBus()
        bus.push(source="browser", url="https://example.com", title="Ex", summary="home")
        record = SharedTaskRecord.create("click next", constraints="")
        text = _task_prompt(record, "R0", bus)
        self.assertIn("V live screen", text)
        self.assertIn("skip_observe", text)
        self.assertIn("example.com", text)

    def test_execute_observe_skips_when_v_live(self):
        bus = LiveScreenBus()
        bus.push(url="https://news.ycombinator.com/", title="HN")
        # Patch module bus used by execute_action
        import live_screen as ls
        import action_runtime as ar

        old = ls.LIVE_SCREEN
        ls.LIVE_SCREEN = bus
        try:
            before = bus.skip_observe_count

            async def _run():
                return await execute_action(
                    ActionProposal(lobe="R", action_type="observe", target="full page")
                )

            result = asyncio.run(_run())
            self.assertTrue(result.success)
            self.assertIn("V live screen", str(result.output))
            self.assertEqual(bus.skip_observe_count, before + 1)
        finally:
            ls.LIVE_SCREEN = old

    def test_coordinator_skips_observe_dispatch(self):
        bus = LiveScreenBus()
        bus.push(url="https://example.com/app", title="App")
        coord = RCoordinator(n_workers=3, live_screen=bus, mode="sequential")
        record = SharedTaskRecord.create("read screen", constraints="")
        a, _ = record.propose(
            effect="observe", resource="screen", args={}, worker_id="seed",
        )
        record.claim(a.action_id, "R0")
        record.mark_prepared(a.action_id, "R0", {"effect": "observe"})
        record.mark_dispatched(a.action_id)
        log: list = []
        skipped = coord._skip_observe_with_v(record, record.actions[a.action_id], log)
        self.assertTrue(skipped)
        self.assertTrue(log[0]["skipped_screenshot"])
        self.assertEqual(record.actions[a.action_id].status, "verified")
        self.assertGreaterEqual(bus.skip_observe_count, 1)


if __name__ == "__main__":
    unittest.main()
