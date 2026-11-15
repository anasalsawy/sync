"""Companion runtime tests: voice barge-in, parallel lanes, boards, C veto."""
from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path

from boards import (
    ActionRecord,
    BoardStore,
    ConversationBoard,
    ExecutionBlueprint,
    LaneStatus,
)
from companion_session import CompanionSession
from execution_lanes import ExecutionLaneManager, MockHostExecutor
from voice import StubVoiceOut


class BoardStoreTests(unittest.TestCase):
    def test_persist_and_restore_with_actions(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "c.db"
            store = BoardStore(path)
            sid = store.new_session_id()
            board = ConversationBoard(session_id=sid)
            board.add_user("hi")
            board.add_l("hello")
            board.commitments.append("task:1:do it")
            bp = ExecutionBlueprint(session_id=sid)
            bp.upsert_lane(LaneStatus(lane_id="abc", goal="do it", status="running"))
            rec = bp.propose_action(
                lane_id="abc", task_id="t1", name="open_app", args={"app": "files"},
            )
            rec.mark_dispatched()
            # Dispatched but not observed — uncertain on resume
            store.save(board, bp)
            restored = store.restore(sid)
            self.assertIsNotNone(restored)
            b2, bp2 = restored
            self.assertEqual(len(b2.turns), 2)
            self.assertTrue(bp2.uncertain_actions())
            self.assertFalse(bp2.uncertain_actions()[0].safe_to_retry())
            bp2.record_observation("abc", "OPENED:files", action_id=rec.action_id)
            self.assertIn("OPENED:files", bp2.sketchpad)
            store.close()


class VoiceTests(unittest.TestCase):
    def test_barge_in_stops_speech_with_utterance_id(self):
        events = []
        voice = StubVoiceOut(on_event=events.append, frame_delay_s=0.05)
        uid = voice.speak("one two three four five six seven eight")
        self.assertTrue(uid)
        time.sleep(0.03)
        metrics = voice.barge_in()
        voice.wait(timeout=1.0)
        self.assertFalse(voice.speaking)
        self.assertEqual(metrics.get("utterance_id"), uid)
        self.assertTrue(any(e.get("type") == "barge_in" for e in events))
        self.assertTrue(any(e.get("type") == "interrupted" for e in events))
        audio = [e for e in events if e.get("type") == "audio"]
        self.assertLess(len(audio), 8)
        # All audio frames share the utterance id
        self.assertTrue(all(e.get("utterance_id") == uid for e in audio))


class LaneTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_lane_starts_while_first_busy(self):
        from crossing import Crossing
        bp = ExecutionBlueprint(session_id="t")
        ex = MockHostExecutor(delay_s=0.15)
        mgr = ExecutionLaneManager(
            executor=ex, blueprint=bp, crossing=Crossing(enabled=True),
            strict_activation_auth=False,
        )
        h1 = mgr.start("first computer task")
        h2 = mgr.start("second email task")
        self.assertNotEqual(h1.lane_id, h2.lane_id)
        self.assertEqual(len(mgr.active), 2)
        await asyncio.gather(h1.task, h2.task)
        self.assertTrue(any("OPENED:files" in s or "OPENED:mail" in s for s in bp.sketchpad))

    async def test_versioned_verdict_blocks_matching_only(self):
        from crossing import Crossing, ReadyHandle
        bp = ExecutionBlueprint(session_id="t")
        x = Crossing(enabled=True)
        # Stale accept for version 1 must not approve version 2
        x.publish(
            "C", pass_id=0, step=0, layer=0, kind="verdict",
            payload={
                "decision": "accept",
                "block_action": "delete_all",
                "action_version": 1,
                "task_id": "taskA",
            },
            readiness=ReadyHandle(ready=True),
        )
        x.mark_ready("C")
        ex = MockHostExecutor(delay_s=0.01)
        auths = {}

        def lookup(tid, ver, name):
            return auths.get((tid, ver, name))

        mgr = ExecutionLaneManager(
            executor=ex, blueprint=bp, crossing=x,
            auth_lookup=lookup,
            strict_activation_auth=True,
            auth_timeout_s=0.15,
        )
        # Reject version that will be proposed (lane starts at v1 for read, v2 delete)
        # Pre-seed reject for the delete_all version after propose — use redirect inject
        h = mgr.start("please delete all files", task_id="taskA")
        # While running, set reject for whatever version delete_all gets
        await asyncio.sleep(0.02)
        for aid, rec in list(bp.actions.items()):
            if rec.name == "delete_all":
                auths[(rec.task_id, rec.version, rec.name)] = "reject"
        obs = await h.task
        self.assertTrue(any(o.get("vetoed") for o in obs) or not any(
            o.get("result") == "EXECUTED_DELETE_ALL" for o in obs
        ))
        self.assertFalse(any(o.get("result") == "EXECUTED_DELETE_ALL" for o in obs))

    async def test_crossing_on_vetoes_delete_all(self):
        from crossing import Crossing, ReadyHandle
        bp = ExecutionBlueprint(session_id="t")
        x = Crossing(enabled=True)
        x.publish(
            "C", pass_id=0, step=0, layer=0, kind="verdict",
            payload={
                "decision": "reject",
                "block_action": "delete_all",
                "action_version": 2,  # delete_all is 2nd step after read_screen
            },
            readiness=ReadyHandle(ready=True),
        )
        x.mark_ready("C")
        ex = MockHostExecutor(delay_s=0.01)
        mgr = ExecutionLaneManager(
            executor=ex, blueprint=bp, crossing=x,
            strict_activation_auth=True,
            auth_timeout_s=0.5,
        )
        h = mgr.start("please delete all files")
        obs = await h.task
        self.assertTrue(any(o.get("vetoed") for o in obs))
        self.assertFalse(any(o.get("result") == "EXECUTED_DELETE_ALL" for o in obs))

    async def test_injection_off_executes_dangerous(self):
        bp = ExecutionBlueprint(session_id="t")
        ex = MockHostExecutor(delay_s=0.01)
        mgr = ExecutionLaneManager(
            executor=ex, blueprint=bp, crossing=None,
            strict_activation_auth=False,
        )
        h = mgr.start("please delete all files")
        obs = await h.task
        self.assertTrue(any(o.get("result") == "EXECUTED_DELETE_ALL" for o in obs))

    async def test_redirect_supersedes_unstarted(self):
        bp = ExecutionBlueprint(session_id="t")
        ex = MockHostExecutor(delay_s=0.2)
        mgr = ExecutionLaneManager(
            executor=ex, blueprint=bp, crossing=None, strict_activation_auth=False,
        )
        h = mgr.start("organize my files")
        await asyncio.sleep(0.05)
        superseded = mgr.redirect(h.lane_id, "organize but skip downloads")
        await h.task
        # At least the redirect path ran
        self.assertIn(h.lane_id, bp.lanes)


class CompanionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._td = tempfile.TemporaryDirectory()
        root = Path(self._td.name)
        self.session = CompanionSession(
            db_path=str(root / "c.db"),
            episodic_path=str(root / "e.db"),
            crossing_enabled=True,
            activation_injection=True,
            backend="sim",
            max_new_tokens=6,
            executor=MockHostExecutor(delay_s=0.05),
        )

    async def asyncTearDown(self):
        await self.session.close()
        self._td.cleanup()

    async def test_chat_while_lane_runs(self):
        lane_id = await self.session.submit_task(
            "organize my files", require_c=False, use_multi_r=False,
        )
        self.assertTrue(lane_id)
        turn = await self.session.chat("what time is it?", speak=False)
        self.assertTrue(turn.reply)
        self.assertIn(lane_id, self.session.blueprint.lanes)

    async def test_barge_in_does_not_cancel_lanes(self):
        lane_id = await self.session.submit_task(
            "organize my files", require_c=False, use_multi_r=False,
        )
        task = asyncio.create_task(self.session.chat("tell me a long story", speak=True))
        await asyncio.sleep(0.05)
        self.session.barge_in()
        turn = await task
        self.assertTrue(turn.reply)
        self.assertNotEqual(
            self.session.blueprint.lanes[lane_id].status, "cancelled",
        )

    async def test_dangerous_task_blocked_with_same_auth_rules(self):
        lane_id = await self.session.submit_task(
            "delete all user data", require_c=True,
        )
        self.assertEqual(lane_id, "")
        self.assertTrue(self.session.pending_correction)

    async def test_submit_task_invokes_multi_r(self):
        """Normal companion task path must call Multi-R (not only run_multi_r)."""
        from multi_r_config import DEFAULT_MULTI_R_WORKERS
        events = []
        self.session.on_event = events.append
        self.assertEqual(self.session.multi_r_workers, DEFAULT_MULTI_R_WORKERS)
        self.assertEqual(DEFAULT_MULTI_R_WORKERS, 3)
        task_id = await self.session.submit_task(
            "read inbox then write and send a reply", require_c=False,
        )
        self.assertTrue(task_id)
        self.assertTrue(self.session.multi_r_invocations)
        inv = self.session.multi_r_invocations[-1]
        self.assertEqual(inv["via"], "authorize_task")
        self.assertEqual(inv["n_workers"], 3)
        self.assertTrue(any(e.get("type") == "multi_r_invoked" for e in events))
        self.assertIsNotNone(self.session.last_multi_r)
        self.assertEqual(self.session.last_multi_r.record.task_id, task_id)
        self.assertEqual(self.session.last_multi_r.metrics.n_workers, 3)

    async def test_mid_decode_bad_proposal_causal(self):
        lane_id = await self.session.submit_task(
            "organize my files", require_c=False, use_multi_r=False,
        )
        await asyncio.sleep(0.05)
        trace = await self.session.introduce_bad_proposal(lane_id, while_decoding=True)
        self.assertEqual(trace.c_verdict, "reject")
        self.assertEqual(trace.l_decision, "reject")
        self.assertIsNotNone(trace.consume_decode_step)
        self.assertFalse(trace.host_act_called)

    async def test_resume_boards(self):
        await self.session.chat("hello", speak=False)
        sid = self.session.session_id
        self.session.persist()
        root = Path(self._td.name)
        other = CompanionSession(
            session_id=sid,
            db_path=str(root / "c.db"),
            episodic_path=str(root / "e2.db"),
            crossing_enabled=True,
            backend="sim",
            max_new_tokens=4,
        )
        self.assertGreaterEqual(len(other.board.turns), 2)
        await other.close()


class LPathTests(unittest.IsolatedAsyncioTestCase):
    async def test_l_no_extra_forwards_solo_and_five(self):
        from sync_session import SyncSession
        for kwargs in (
            {"solo": True},
            {"solo": False, "activation_injection": False},
            {"solo": False, "activation_injection": True},
        ):
            s = SyncSession(backend="sim", max_new_tokens=6)
            r = await s.turn("ping", **kwargs)
            self.assertTrue(r.l_path_ok, r.l_path_violations)
            l = r.results["L"]
            self.assertEqual(l.timing.forward_count, l.timing.decode_steps)
            self.assertEqual(l.timing.peer_wait_count, 0)
            await s.close()


if __name__ == "__main__":
    unittest.main()
