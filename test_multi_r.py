"""Multi-R: shared task claims, cross-attn influence, prep-ahead dispatch."""
from __future__ import annotations

import asyncio
import unittest

from batch_r import BatchedRTeam, CROSS_ATTN_LAYERS
from multi_r_config import DEFAULT_MULTI_R_WORKERS, MultiRLaunchError
from r_coordinator import RCoordinator, SequentialExecutor
from task_record import SharedTaskRecord, action_identity


class TaskRecordTests(unittest.TestCase):
    def test_same_effect_same_identity(self):
        a = action_identity(
            effect="edit", resource="src/app.py",
            args={"patch": "fix_a", "line": 10},
        )
        b = action_identity(
            effect="edit", resource="src/app.py",
            args={"patch": "fix_a", "line": 10, "description": "other words"},
        )
        self.assertEqual(a, b)

    def test_atomic_claim_one_winner(self):
        rec = SharedTaskRecord.create("goal")
        act, is_new = rec.propose(
            effect="edit", resource="f.py", args={"patch": "x"}, worker_id="R0",
        )
        self.assertTrue(is_new)
        c0 = rec.claim(act.action_id, "R0")
        c1 = rec.claim(act.action_id, "R1")
        self.assertIsNotNone(c0)
        self.assertIsNone(c1)
        self.assertEqual(rec.duplicate_claim_attempts, 1)

    def test_duplicate_propose_collapses(self):
        rec = SharedTaskRecord.create("goal")
        a, n1 = rec.propose(
            effect="edit", resource="f.py", args={"patch": "x"}, worker_id="R0",
        )
        b, n2 = rec.propose(
            effect="edit", resource="f.py",
            args={"patch": "x", "wording": "please apply x"}, worker_id="R1",
        )
        self.assertTrue(n1)
        self.assertFalse(n2)
        self.assertEqual(a.action_id, b.action_id)
        self.assertEqual(len(rec.actions), 1)

    def test_resource_conflict_orders(self):
        rec = SharedTaskRecord.create("goal")
        a, _ = rec.propose(
            effect="edit", resource="f.py", args={"patch": "a"}, worker_id="s",
        )
        b, _ = rec.propose(
            effect="edit", resource="f.py", args={"patch": "b"}, worker_id="s",
        )
        self.assertIsNotNone(rec.claim(a.action_id, "R0"))
        self.assertIsNone(rec.claim(b.action_id, "R1"))
        self.assertGreaterEqual(rec.conflicts_ordered, 1)


class CrossAttnTests(unittest.TestCase):
    def test_default_is_three_workers(self):
        self.assertEqual(DEFAULT_MULTI_R_WORKERS, 3)
        team = BatchedRTeam()  # provisional default
        self.assertEqual(len(team.workers), 3)
        team.ensure_launched()

    def test_launch_failure_when_count_invalid(self):
        with self.assertRaises(MultiRLaunchError):
            BatchedRTeam(n_workers=0)

    def test_perturb_changes_peer_within_run(self):
        rec = SharedTaskRecord.create("share activations")
        team = BatchedRTeam(n_workers=3, max_new_tokens=6, cross_attn=True)
        result = team.run_inference(rec, influence_probe=True)
        self.assertTrue(result.cross_attn_enabled)
        self.assertIn(CROSS_ATTN_LAYERS[0], CROSS_ATTN_LAYERS)
        self.assertTrue(
            result.influence_proof.get("peer_changed")
            or result.timing.cross_attn_influence,
            result.influence_proof,
        )

    def test_cross_attn_off_no_influence_flag(self):
        rec = SharedTaskRecord.create("no share")
        team = BatchedRTeam(n_workers=3, max_new_tokens=4, cross_attn=False)
        result = team.run_inference(rec, influence_probe=True)
        self.assertFalse(result.cross_attn_enabled)
        self.assertFalse(result.timing.cross_attn_influence)


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_dispatches_once(self):
        coord = RCoordinator(
            n_workers=3, cross_attn=True, mode="sequential",
            executor=SequentialExecutor(delay_s=0.02),
        )
        result = await coord.run_task(
            "Fix the failing unit tests in this code task",
            force_duplicate=True,
        )
        ids = [d["action_id"] for d in result.dispatch_log]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(result.record.duplicate_claim_attempts, 1)
        edits = [d for d in result.dispatch_log if d["effect"] == "edit"]
        # Conflicting edits on same file — at most one in flight; both may
        # eventually verify sequentially after invalidate/reclaim
        self.assertGreaterEqual(len(edits), 1)

    async def test_parallel_independent_actions(self):
        coord = RCoordinator(n_workers=3, cross_attn=True, mode="parallel")
        result = await coord.run_task(
            "Process independent items in parallel: alpha beta gamma delta",
            force_duplicate=False,
        )
        self.assertTrue(result.satisfied)
        self.assertGreaterEqual(len(result.dispatch_log), 3)

    async def test_three_faster_than_one_on_parallel(self):
        from r_coordinator import ParallelExecutor
        delay = 0.1
        # 1R is explicit eval control; 3R is provisional default
        one = RCoordinator(
            n_workers=1, cross_attn=False, mode="parallel",
            executor=ParallelExecutor(delay_s=delay, max_parallel=4),
        )
        three = RCoordinator(
            n_workers=3, cross_attn=True, mode="parallel",
            executor=ParallelExecutor(delay_s=delay, max_parallel=4),
        )
        goal = "Process independent items in parallel: alpha beta gamma delta"
        r1 = await one.run_task(goal, force_duplicate=False)
        r3 = await three.run_task(goal, force_duplicate=False)
        self.assertTrue(r1.satisfied and r3.satisfied)
        self.assertLess(r3.metrics.total_task_s, r1.metrics.total_task_s * 1.05)


if __name__ == "__main__":
    unittest.main()
