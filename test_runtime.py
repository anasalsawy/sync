"""One persistent mind: sharing, addressing, and emergence from demand."""
import asyncio
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace

from runtime import (
    Action, BranchAction, Decision, Demand, Exchange, MAIN, Mind,
    Observation, Reply, Subtask, Sync,
)


def mesh(active, site="layers.14"):
    """A spanning chain of exchanges, so every active lobe shared this pass."""
    return tuple(Exchange(source, target, site, 0.5)
                 for source, target in zip(active, active[1:]))


class Scripted:
    """Drives the mind from a step function and always reports sharing."""

    def __init__(self, step):
        self.step = step
        self.seen = []
        self.closed = False

    async def coupled_pass(self, mind, links):
        self.seen.append((mind, mind.active, links))
        decision = self.step(mind, links)
        if asyncio.iscoroutine(decision):
            decision = await decision
        return replace(decision, exchanges=decision.exchanges or mesh(mind.active))

    async def close(self, mind):
        self.closed = True

    @property
    def actives(self):
        return [active for _, active, _ in self.seen]


class Raw(Scripted):
    """Reports exactly what the step function returned, sharing included."""

    async def coupled_pass(self, mind, links):
        self.seen.append((mind, mind.active, links))
        return self.step(mind, links)


class MemoryWorker:
    def __init__(self):
        self.current = "READY"
        self.clicks = []

    async def observe(self):
        return Observation(None, (("output", self.current),), self.current)

    async def execute(self, action):
        if any(self.current != was for _, was in action.seen):
            return None
        target = action.payload["target"]
        self.clicks.append(target)
        self.current = target
        return await self.observe()


class FakeHost:
    supports_subagents = True

    def __init__(self):
        self.workers = {}
        self.released = []

    @asynccontextmanager
    async def open_subagent(self, subtask):
        worker = MemoryWorker()
        self.workers[subtask.key] = worker
        try:
            yield worker
        finally:
            self.released.append(subtask.key)

    @asynccontextmanager
    async def open_instance(self, subtask):  # pragma: no cover - native host
        raise AssertionError("This host delegates natively")
        yield


def keyed(mind):
    return {item.text: key for key, item in mind.stimuli.items()}


class MindTests(unittest.IsolatedAsyncioTestCase):

    # ------------------------------------------------------- one mind

    async def test_a_second_prompt_lands_in_the_same_mind(self):
        arrived, release = asyncio.Event(), asyncio.Event()

        async def step(mind, links):
            if mind.passes == 0:
                return Decision(demands=(Demand("L", another_pass=True),))
            arrived.set()
            await release.wait()
            self.assertEqual(len(mind.stimuli), 2)
            keys = keyed(mind)
            return Decision(replies=(
                Reply(keys["second"], "answered second"),
                Reply(keys["first"], "answered first"),
            ))

        adapter = Scripted(step)
        sync = Sync(adapter)
        first = sync.send("first")
        await asyncio.wait_for(arrived.wait(), 5)
        second = sync.send("second")
        release.set()
        self.assertEqual(await asyncio.wait_for(first, 5), "answered first")
        self.assertEqual(await asyncio.wait_for(second, 5), "answered second")
        minds = {id(mind) for mind, _, _ in adapter.seen}
        self.assertEqual(len(minds), 1, "a second prompt must not build a second mind")
        await sync.aclose()
        self.assertTrue(adapter.closed)

    async def test_every_lobe_shares_with_every_other_lobe(self):
        captured = {}

        def step(mind, links):
            if mind.passes == 0:
                return Decision(demands=(Demand("L", split=1), Demand("R", split=1)))
            captured.update(links)
            keys = keyed(mind)
            return Decision(replies=(Reply(keys["go"], "done"),))

        sync = Sync(Scripted(step))
        self.assertEqual(await asyncio.wait_for(sync.send("go"), 5), "done")
        everyone = set(MAIN) | {"L1", "R1"}
        self.assertEqual(set(captured), everyone)
        for lobe, peers in captured.items():
            self.assertEqual(set(peers), everyone - {lobe}, lobe)

    # ------------------------------------------------- demand, not schedule

    async def test_the_next_pass_is_the_sum_of_what_the_lobes_wanted(self):
        def step(mind, links):
            if mind.passes == 0:
                keys = keyed(mind)
                return Decision(
                    replies=(Reply(keys["go"], "working on it"),),
                    demands=(Demand("R", another_pass=True, split=2),),
                )
            return Decision()

        adapter = Scripted(step)
        sync = Sync(adapter)
        self.assertEqual(await asyncio.wait_for(sync.send("go"), 5), "working on it")
        await asyncio.wait_for(sync._pump, 5)
        # Nobody wrote this list. L spoke and stood down; R asked to keep
        # thinking and for two lanes of itself; the lanes woke because they
        # were born. That is the whole schedule.
        self.assertEqual(adapter.actives, [MAIN, ("R", "R1", "R2")])

    async def test_lanes_are_elastic_and_uncapped(self):
        def step(mind, links):
            if mind.passes == 0:
                return Decision(demands=(Demand("R", another_pass=True, split=3),))
            if mind.passes == 1:
                return Decision(demands=(Demand("R", another_pass=True, split=5),))
            keys = keyed(mind)
            return Decision(replies=(Reply(keys["go"], "spread out"),))

        sync = Sync(Scripted(step))
        self.assertEqual(await asyncio.wait_for(sync.send("go"), 5), "spread out")
        self.assertEqual(sync.mind.offspring["R"],
                         tuple(f"R{index}" for index in range(1, 9)))

    async def test_a_lane_holding_work_is_never_put_down(self):
        host = FakeHost()

        def step(mind, links):
            if mind.passes == 0:
                keys = keyed(mind)
                return Decision(
                    replies=(Reply(keys["go"], "on it"),),
                    split=(Subtask("a", "do a"),),
                    demands=(Demand("R", another_pass=True, split=1),),
                )
            if mind.passes == 1:
                return Decision(assignments=(("a", "R1"),), branch_actions=(
                    BranchAction("a", "R1", Action(
                        {"kind": "click", "target": "Save"},
                        claims=frozenset({"output"}), seen=(("output", "READY"),),
                    )),
                ))
            return Decision(complete_branches=("a",))

        adapter = Scripted(step)
        sync = Sync(adapter, host=host)
        self.assertEqual(await asyncio.wait_for(sync.send("go"), 5), "on it")
        await asyncio.wait_for(sync._pump, 5)
        # No demand named R1 on the third pass. It stayed awake because it
        # was still holding an unfinished subtask.
        self.assertEqual(adapter.actives[2], ("R1",))
        self.assertEqual(host.workers["a"].clicks, ["Save"])
        self.assertEqual(sync.mind.branch_results["a"].result, "Save")
        self.assertEqual(host.released, ["a"])

    # ------------------------------------------------------------ the rules

    async def test_only_l_and_r_open_lanes_of_their_own(self):
        def step(mind, links):
            return Decision(demands=(Demand("V", split=1),))

        with self.assertRaisesRegex(PermissionError, "Only L and R open lanes"):
            await asyncio.wait_for(Sync(Scripted(step)).send("go"), 5)

    async def test_only_l_speaks(self):
        def step(mind, links):
            keys = keyed(mind)
            return Decision(replies=(Reply(keys["look"], "I am not L"),))

        with self.assertRaisesRegex(PermissionError, "Only L speaks"):
            await asyncio.wait_for(Sync(Scripted(step)).send("look", origin="V"), 5)

    async def test_a_lobe_initiated_stimulus_proceeds_without_l(self):
        adapter = Scripted(lambda mind, links: Decision(
            demands=(Demand("B", another_pass=True),) if mind.passes == 0 else (),
        ))
        sync = Sync(adapter)
        await asyncio.wait_for(sync.send("remember this", origin="B"), 5)
        await asyncio.wait_for(sync._pump, 5)
        self.assertEqual(adapter.actives, [("B",), ("B",)])
        self.assertTrue(all("L" not in active for active in adapter.actives))

    async def test_r_must_be_active_to_delegate(self):
        def step(mind, links):
            if mind.passes == 0:
                keys = keyed(mind)
                return Decision(replies=(Reply(keys["go"], "spoken"),),
                                demands=(Demand("B", another_pass=True),))
            return Decision(split=(Subtask("a", "do a"),))

        sync = Sync(Scripted(step), host=FakeHost())
        await asyncio.wait_for(sync.send("go"), 5)
        with self.assertRaises(PermissionError):
            await asyncio.wait_for(sync._pump, 5)

    # -------------------------------------------------------- live sharing

    async def test_a_pass_without_live_sharing_is_refused(self):
        def step(mind, links):
            keys = keyed(mind)
            return Decision(replies=(Reply(keys["go"], "silent"),))

        with self.assertRaisesRegex(PermissionError, "did not share live signals"):
            await asyncio.wait_for(Sync(Raw(step)).send("go"), 5)

    async def test_an_exchange_that_moved_nothing_is_refused(self):
        def step(mind, links):
            keys = keyed(mind)
            return Decision(replies=(Reply(keys["go"], "flat"),),
                            exchanges=mesh(MAIN) + (Exchange("L", "C", "layers.14", 0.0),))

        with self.assertRaisesRegex(ValueError, "moved nothing"):
            await asyncio.wait_for(Sync(Raw(step)).send("go"), 5)

    async def test_an_exchange_cannot_name_an_inactive_lobe(self):
        def step(mind, links):
            keys = keyed(mind)
            return Decision(replies=(Reply(keys["go"], "ghost"),),
                            exchanges=mesh(MAIN) + (Exchange("L", "R9", "layers.14", 0.5),))

        with self.assertRaisesRegex(PermissionError, "not active"):
            await asyncio.wait_for(Sync(Raw(step)).send("go"), 5)

    # ------------------------------------------------------------- lifetime

    async def test_interruption_stops_the_mind(self):
        entered = asyncio.Event()

        async def step(mind, links):
            entered.set()
            await asyncio.Future()

        adapter = Scripted(step)
        sync = Sync(adapter)
        answer = sync.send("hold")
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(sync.interrupt(), 5)
        self.assertTrue(answer.cancelled())
        await sync.aclose()
        self.assertTrue(adapter.closed)


if __name__ == "__main__":
    unittest.main()
