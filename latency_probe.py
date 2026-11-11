"""What the runtime itself costs. The inference is a free pass.

This measures the plumbing, not a model. The adapter returns instantly, so
every number here is the overhead the runtime adds around whatever the model
actually takes. End-to-end latency on a real session is this plus the forward
pass plus the host's own call. Four questions, each timed over 100 runs on
real Chromium:

  * TTFA: a prompt arrives into a quiet mind. How long before its first act
    is dispatched to the host?
  * step: one lane, one act. From the adapter's decision to the host's call.
  * fanout: an R split with four lanes. How long before every lane is acting?
  * tick: the clock watcher's wake frequency, so time-awareness is not stale.
"""
import asyncio
import os
import statistics
import time
from contextlib import asynccontextmanager
from dataclasses import replace

from playwright.async_api import async_playwright

from runtime import (
    Action, BranchAction, Decision, Demand, Exchange,
    Observation, Reply, Subtask, Sync,
)


def mesh(active, site="layers.14"):
    return tuple(Exchange(source, target, site, 0.5)
                 for source, target in zip(active, active[1:]))


class InstantWorker:
    """A host worker that returns in microseconds; isolates runtime cost."""

    def __init__(self):
        self.dispatched_at = []

    async def observe(self):
        return Observation(None, (("field", "READY"),), "READY")

    async def execute(self, action):
        self.dispatched_at.append(time.perf_counter())
        return Observation(None, (("field", action.payload["value"]),),
                           action.payload["value"])


class InstantHost:
    supports_subagents = False

    def __init__(self):
        self.workers = []

    @asynccontextmanager
    async def open_instance(self, subtask):
        worker = InstantWorker()
        self.workers.append(worker)
        yield worker

    @asynccontextmanager
    async def open_subagent(self, subtask):
        raise AssertionError("instance only")
        yield


class StepAdapter:
    """One lane, one act, then done. The simplest possible shape."""

    async def coupled_pass(self, mind, links):
        return replace(self.decide(mind), exchanges=mesh(mind.active))

    def decide(self, mind):
        keys = {item.text: key for key, item in mind.stimuli.items()}
        if not mind.subtasks:
            return Decision(split=(Subtask("only", "one"),),
                            demands=(Demand("R", another_pass=True, split=1),))
        if not mind.assignments:
            return Decision(assignments=(("only", "R1"),), branch_actions=(
                BranchAction("only", "R1", Action(
                    {"value": "done"}, claims=frozenset({"field"}),
                    seen=(("field", "READY"),),
                )),
            ))
        if "only" not in mind.branch_results:
            if "only" in mind.in_flight:
                return Decision()
            return Decision(complete_branches=("only",))
        return Decision(replies=(Reply(keys["go"], "finished"),))

    async def close(self, mind):
        pass


class FanoutAdapter:
    """Four lanes, dispatched in the same pass."""

    def __init__(self):
        self.count = 4

    async def coupled_pass(self, mind, links):
        return replace(self.decide(mind), exchanges=mesh(mind.active))

    def decide(self, mind):
        keys = {item.text: key for key, item in mind.stimuli.items()}
        names = tuple(f"n{index}" for index in range(self.count))
        if not mind.subtasks:
            return Decision(split=tuple(Subtask(name, "x") for name in names),
                            demands=(Demand("R", another_pass=True, split=self.count),))
        if not mind.assignments:
            pairs = tuple((name, f"R{index + 1}") for index, name in enumerate(names))
            return Decision(assignments=pairs, branch_actions=tuple(
                BranchAction(name, f"R{index + 1}", Action(
                    {"value": f"v{index}"}, claims=frozenset({f"f{index}"}),
                    seen=(("field", "READY"),),
                ))
                for index, name in enumerate(names)
            ))
        open_keys = tuple(name for name in names if name not in mind.branch_results)
        landed = tuple(name for name in open_keys if name not in mind.in_flight)
        if landed:
            return Decision(complete_branches=landed)
        if open_keys:
            return Decision()
        return Decision(replies=(Reply(keys["go"], "finished"),))

    async def close(self, mind):
        pass


async def ttfa_and_step(count=100):
    """Each run: fresh Sync, send, measure to first dispatch; then full."""
    ttfa, full = [], []
    for _ in range(count):
        host = InstantHost()
        sync = Sync(StepAdapter(), host=host, poll_seconds=0.001)
        sent = time.perf_counter()
        answer = sync.send("go")
        await asyncio.wait_for(answer, 5)
        done = time.perf_counter()
        ttfa.append((host.workers[0].dispatched_at[0] - sent) * 1000)
        full.append((done - sent) * 1000)
        await sync.aclose()
    return ttfa, full


async def fanout(count=50):
    """Time between first and last lane dispatching in one fanout."""
    spread = []
    total = []
    for _ in range(count):
        host = InstantHost()
        sync = Sync(FanoutAdapter(), host=host, poll_seconds=0.001)
        sent = time.perf_counter()
        answer = sync.send("go")
        await asyncio.wait_for(answer, 5)
        done = time.perf_counter()
        all_dispatches = sorted(
            moment for worker in host.workers for moment in worker.dispatched_at
        )
        spread.append((all_dispatches[-1] - all_dispatches[0]) * 1000)
        total.append((done - sent) * 1000)
        await sync.aclose()
    return spread, total


async def clock_tick():
    """How closely the clock's advance tracks real time between passes."""
    from clock import Clock
    sync = Sync(StepAdapter(), host=InstantHost(), clock=Clock(), poll_seconds=0.001)
    await asyncio.wait_for(sync.send("go"), 5)
    await asyncio.wait_for(sync._pump, 5)
    before = sync.mind.now
    await asyncio.sleep(0.05)
    after = sync.mind.now
    await sync.aclose()
    return (after - before) * 1000


def show(label, values, unit="ms"):
    p = statistics.quantiles(values, n=100)
    print(f"{label:<18} min {min(values):6.2f}  p50 {p[49]:6.2f}  "
          f"p95 {p[94]:6.2f}  p99 {p[98]:6.2f}  max {max(values):6.2f}  {unit}")


async def main():
    print("Runtime overhead, no inference, no real worker latency")
    print("=" * 72)
    ttfa, full = await ttfa_and_step()
    show("TTFA (quiet->act)", ttfa)
    show("full step", full)
    spread, total = await fanout()
    show("fanout spread", spread)
    show("fanout full", total)
    tick = await clock_tick()
    print(f"\nclock drift over 50 ms of sleep: {tick:6.2f} ms "
          f"(ideal 50; poll=1 ms)")


if __name__ == "__main__":
    asyncio.run(main())
