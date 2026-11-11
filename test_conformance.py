"""Host conformance: one scenario, differently shaped hosts, identical outcome.

The runtime must not know what any host is. Each host below speaks its own
vocabulary and payload shape. The same scenario must produce the same reply
and the same lane behaviour on all of them, and the runtime source must not
name any host.
"""
import asyncio
import pathlib
import re
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace

from runtime import (
    Action, BranchAction, Decision, Demand, Exchange, MAIN,
    Observation, Reply, Subtask, Sync,
)

SUBTASKS = ("alpha", "beta")


def mesh(active, site="layers.14"):
    return tuple(Exchange(source, target, site, 0.5)
                 for source, target in zip(active, active[1:]))


class Ledger:
    """A tiny world each host exposes in its own way."""

    def __init__(self):
        self.values = {name: "empty" for name in SUBTASKS}

    def revisions(self):
        return tuple(sorted((name, value) for name, value in self.values.items()))


class NativeHost:
    """Delegates natively; the worker speaks key/value operations."""

    supports_subagents = True

    def __init__(self):
        self.world = Ledger()
        self.opened = []

    @asynccontextmanager
    async def open_subagent(self, subtask):
        self.opened.append(subtask.key)
        yield KeyValueWorker(self.world, subtask.key)

    @asynccontextmanager
    async def open_instance(self, subtask):  # pragma: no cover
        raise AssertionError("native host")
        yield


class KeyValueWorker:
    def __init__(self, world, key):
        self.world, self.key = world, key

    async def observe(self):
        return Observation({"store": dict(self.world.values)},
                           self.world.revisions(), self.world.values[self.key])

    async def execute(self, action):
        if any(self.world.values[name] != was for name, was in action.seen):
            return None
        self.world.values[self.key] = action.payload["op"]["set"]
        return await self.observe()


class ShellHost:
    """Opens an instance per lane; the worker speaks shell-like commands."""

    supports_subagents = False

    def __init__(self):
        self.world = Ledger()
        self.opened = []

    @asynccontextmanager
    async def open_instance(self, subtask):
        self.opened.append(subtask.key)
        yield ShellWorker(self.world, subtask.key)

    @asynccontextmanager
    async def open_subagent(self, subtask):  # pragma: no cover
        raise AssertionError("instance host")
        yield


class ShellWorker:
    def __init__(self, world, key):
        self.world, self.key = world, key

    async def observe(self):
        return Observation(f"$ cat {self.key}\n{self.world.values[self.key]}",
                           self.world.revisions(), self.world.values[self.key])

    async def execute(self, action):
        if any(self.world.values[name] != was for name, was in action.seen):
            return None
        command = action.payload["cmd"]
        verb, _, value = command.partition("=")
        if verb != f"write {self.key}":
            raise ValueError("unknown command")
        self.world.values[self.key] = value
        return await self.observe()


class SharedWorker:
    """A single surface every lane writes to. It knows no lane's name."""

    def __init__(self, world):
        self.world = world

    async def observe(self):
        return Observation({"store": dict(self.world.values)},
                           self.world.revisions(), "")

    async def execute(self, action):
        if any(self.world.values[name] != was for name, was in action.seen):
            return None
        key = next(iter(action.claims))
        value = action.payload["op"]["set"]
        self.world.values[key] = value
        # The act reports what it produced. The surface cannot know which
        # lane asked, so the result comes from the act, as it would on a host.
        return Observation({"store": dict(self.world.values)},
                           self.world.revisions(), value)


class SharedSurfaceHost:
    """One surface shared by every lane: the same worker is handed out twice."""

    supports_subagents = True

    def __init__(self):
        self.world = Ledger()
        self.shared = SharedWorker(self.world)
        self.opened = []

    @asynccontextmanager
    async def open_subagent(self, subtask):
        self.opened.append(subtask.key)
        yield self.shared

    @asynccontextmanager
    async def open_instance(self, subtask):  # pragma: no cover
        raise AssertionError("native host")
        yield


def action_for(host_kind, key, value, seen_value):
    """Each host gets an action in its own payload shape; the claims are shared."""
    if host_kind == "shell":
        payload = {"cmd": f"write {key}={value}"}
    else:
        payload = {"op": {"set": value}}
    return Action(payload, claims=frozenset({key}),
                  seen=((key, seen_value),))


class Scenario:
    """The same mind behaviour for every host. Nothing here names a host."""

    def __init__(self, host_kind):
        self.host_kind = host_kind
        self.result = None

    async def coupled_pass(self, mind, links):
        return replace(self.decide(mind), exchanges=mesh(mind.active))

    def decide(self, mind):
        keys = {item.text: key for key, item in mind.stimuli.items()}
        if not mind.subtasks:
            return Decision(
                split=tuple(Subtask(name, f"set {name}") for name in SUBTASKS),
                demands=(Demand("R", another_pass=True, split=len(SUBTASKS)),),
            )
        if not mind.assignments:
            pairs = tuple(zip(SUBTASKS, ("R1", "R2")))
            return Decision(assignments=pairs, branch_actions=tuple(
                BranchAction(name, actor,
                             action_for(self.host_kind, name, f"{name}-done", "empty"))
                for name, actor in pairs
            ))
        open_keys = tuple(name for name in SUBTASKS if name not in mind.branch_results)
        landed = tuple(name for name in open_keys if name not in mind.in_flight)
        if landed:
            return Decision(complete_branches=landed)
        if open_keys:
            return Decision()
        results = " | ".join(mind.branch_results[name].result for name in SUBTASKS)
        return Decision(replies=(Reply(keys["do both"], results),))

    async def close(self, mind):
        pass


class HostConformance(unittest.IsolatedAsyncioTestCase):

    async def run_scenario(self, host, host_kind):
        sync = Sync(Scenario(host_kind), host=host)
        try:
            reply = await asyncio.wait_for(sync.send("do both"), 10)
        finally:
            await sync.aclose()
        return reply, sync

    async def test_native_host_completes_the_scenario(self):
        host = NativeHost()
        reply, sync = await self.run_scenario(host, "kv")
        self.assertEqual(reply, "alpha-done | beta-done")
        self.assertCountEqual(host.opened, SUBTASKS)
        self.assertEqual(host.world.values, {"alpha": "alpha-done", "beta": "beta-done"})

    async def test_instance_host_completes_the_same_scenario(self):
        host = ShellHost()
        reply, sync = await self.run_scenario(host, "shell")
        self.assertEqual(reply, "alpha-done | beta-done")
        self.assertCountEqual(host.opened, SUBTASKS)
        self.assertEqual(host.world.values, {"alpha": "alpha-done", "beta": "beta-done"})

    async def test_shared_surface_host_completes_the_same_scenario(self):
        host = SharedSurfaceHost()
        reply, sync = await self.run_scenario(host, "kv")
        self.assertEqual(reply, "alpha-done | beta-done")
        self.assertEqual(host.world.values, {"alpha": "alpha-done", "beta": "beta-done"})
        self.assertEqual(sync._surfaces, {}, "the shared surface closed once")

    async def test_the_runtime_names_no_host(self):
        here = pathlib.Path(__file__).parent
        source = (here / "runtime.py").read_text() + (here / "control.py").read_text()
        for forbidden in ("click", "browser", "playwright", "screenshot", "dom",
                          "keyboard", "shell", "cursor", "hermes", "ui-tars",
                          "subprocess", "http"):
            with self.subTest(word=forbidden):
                self.assertIsNone(re.search(rf"\b{forbidden}\b", source, re.IGNORECASE),
                                  f"runtime names {forbidden!r}")


if __name__ == "__main__":
    unittest.main()
