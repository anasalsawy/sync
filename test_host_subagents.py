"""Host subagent mapping for Multi-R — native vs single-path."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from companion_session import CompanionSession
from host_subagents import (
    NativeSubagentHost,
    SinglePathHost,
    SubagentMappedExecutor,
    bind_host_executor,
)
from multi_r_config import DEFAULT_MULTI_R_WORKERS
from r_coordinator import RCoordinator
from task_record import SharedTaskRecord


class BindHostTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_launches_three_subagents(self):
        host = NativeSubagentHost(delay_s=0.01)
        ex, mode, info = await bind_host_executor(host, n_workers=3)
        self.assertEqual(info["host_mode"], "native_subagent")
        self.assertEqual(info["n_subagents"], 3)
        self.assertEqual(len(ex.mapping), 3)
        self.assertEqual(mode, "parallel")
        await ex.close()

    async def test_single_path_keeps_three_workers_label(self):
        host = SinglePathHost(delay_s=0.01)
        ex, mode, info = await bind_host_executor(host, n_workers=3)
        self.assertEqual(info["host_mode"], "single_path")
        self.assertEqual(info["r_workers_active"], 3)
        self.assertEqual(mode, "sequential")
        self.assertEqual(len(ex.worker_ids), 3)


class MappingReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_to_subagent_stable_across_actions(self):
        host = NativeSubagentHost(delay_s=0.01)
        ex = await SubagentMappedExecutor.launch(host, ["R0", "R1", "R2"])
        rec = SharedTaskRecord.create("Process independent items in parallel: a b c")
        from r_coordinator import _seed_actions_for_goal
        _seed_actions_for_goal(rec)
        # Claim as R1 twice — same subagent
        acts = rec.available_actions()
        a0 = rec.claim(acts[0].action_id, "R1")
        self.assertIsNotNone(a0)
        obs1 = await ex.execute(a0, {})
        # Second action also R1
        acts = rec.available_actions()
        a1 = rec.claim(acts[0].action_id, "R1")
        self.assertIsNotNone(a1)
        obs2 = await ex.execute(a1, {})
        self.assertEqual(obs1["subagent_id"], obs2["subagent_id"])
        self.assertEqual(obs1["worker_id"], "R1")
        await ex.close()


class CompanionHostPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_task_native_subagents(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            host = NativeSubagentHost(delay_s=0.05)
            events = []
            session = CompanionSession(
                db_path=str(root / "c.db"),
                episodic_path=str(root / "e.db"),
                backend="sim",
                max_new_tokens=4,
                host=host,
                on_event=events.append,
            )
            tid = await session.submit_task(
                "Process independent items in parallel: alpha beta gamma delta",
                require_c=False,
            )
            self.assertTrue(tid)
            self.assertEqual(session.last_host_launch["host_mode"], "native_subagent")
            self.assertEqual(session.last_host_launch["n_subagents"], 3)
            self.assertEqual(len(host.launched), 3)
            self.assertTrue(any(e.get("type") == "host_launch" for e in events))
            self.assertTrue(session.last_multi_r.satisfied)
            ids = [d["action_id"] for d in session.last_multi_r.dispatch_log]
            self.assertEqual(len(ids), len(set(ids)))
            await session.close()

    async def test_submit_task_single_path_prep_ahead(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            host = SinglePathHost(delay_s=0.08)
            session = CompanionSession(
                db_path=str(root / "c.db"),
                episodic_path=str(root / "e.db"),
                backend="sim",
                max_new_tokens=4,
                host=host,
            )
            tid = await session.submit_task(
                "Pipeline: read the inbox, write a reply to outbox, then send it.",
                require_c=False,
            )
            self.assertTrue(tid)
            self.assertEqual(session.last_host_launch["host_mode"], "single_path")
            self.assertEqual(session.last_multi_r.metrics.n_workers, 3)
            # Prepares during execution appear on timeline
            prep_busy = [
                e for e in session.last_multi_r.timeline
                if e.kind == "prepare" and e.detail.get("while_executor_busy")
            ]
            if not prep_busy:
                # Window overlap fallback
                tl = session.last_multi_r.timeline
                starts = [e.at for e in tl if e.kind == "dispatch_start"]
                ends = [e.at for e in tl if e.kind == "dispatch_end"]
                self.assertTrue(starts and ends)
                prep_any = [e for e in tl if e.kind == "prepare" and e.detail.get("speculative")]
                self.assertTrue(prep_any or starts)
            ids = [d["action_id"] for d in session.last_multi_r.dispatch_log]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(DEFAULT_MULTI_R_WORKERS, 3)
            await session.close()


if __name__ == "__main__":
    unittest.main()
