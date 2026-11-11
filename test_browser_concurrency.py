"""Real Chromium, one mind: the conversation answers while two lanes act.

Each lane is its own Chromium process, opened by the host, because this host
declares no native delegation. Nothing in the runtime knows that.
"""
import asyncio
import os
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace

from playwright.async_api import async_playwright

from runtime import (
    Action, BranchAction, Decision, Demand, Exchange, MAIN,
    Observation, Reply, Subtask, Sync,
)

OUTCOMES = {"order": "SHIPPED", "invoice": "PAID"}
FOLLOW_UP = {"SHIPPED": "Track delivery", "PAID": "Open receipt"}


def mesh(active, site="layers.14"):
    return tuple(Exchange(source, target, site, 0.5)
                 for source, target in zip(active, active[1:]))


class FixtureBrowser:
    def __init__(self, page, name, entered, release, trace):
        self.page = page
        self.name = name
        self.entered = entered
        self.release = release
        self.trace = trace

    async def observe(self):
        status = await self.page.locator("output").inner_text()
        return Observation(await self.page.screenshot(),
                           (("output", status),), status)

    async def execute(self, action):
        current = await self.observe()
        if any(current.revision(name) != was for name, was in action.seen):
            return None
        target = action.payload["target"]
        self.trace.append(("start", self.name, target))
        if target == "Inspect":
            self.entered[self.name].set()
            await self.release.wait()
        await self.page.get_by_role("button", name=target, exact=True).click()
        self.trace.append(("finish", self.name, target))
        return await self.observe()


@asynccontextmanager
async def chromium_worker(key, entered, release, trace):
    """One Chromium process of its own, for one lane."""
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=os.getenv("SYNC_CHROMIUM", "/usr/bin/chromium")
        )
        try:
            page = await browser.new_page(viewport={"width": 480, "height": 240})
            status = OUTCOMES[key]
            await page.set_content(f"""
                <output>READY</output>
                <button onclick="document.querySelector('output').textContent='{status}';
                    this.hidden=true; document.querySelector('#followup').hidden=false">Inspect</button>
                <button id="followup" hidden onclick="
                    document.querySelector('output').textContent=this.textContent">
                    {FOLLOW_UP[status]}</button>
            """)
            yield FixtureBrowser(page, key, entered, release, trace)
        finally:
            await browser.close()


class ChromiumHost:
    """A host with no native delegation: concurrency is extra instances."""

    supports_subagents = False

    def __init__(self, entered, release, trace):
        self.entered, self.release, self.trace = entered, release, trace
        self.opened, self.released = [], []

    @asynccontextmanager
    async def open_instance(self, subtask):
        self.opened.append(subtask.key)
        async with chromium_worker(subtask.key, self.entered,
                                   self.release, self.trace) as worker:
            try:
                yield worker
            finally:
                self.released.append(subtask.key)

    @asynccontextmanager
    async def open_subagent(self, subtask):  # pragma: no cover - not native
        raise AssertionError("This host has no native delegation")
        yield


class Adapter:
    def __init__(self, test):
        self.test = test
        self.actives = []
        self.minds = set()

    async def coupled_pass(self, mind, links):
        self.actives.append(mind.active)
        self.minds.add(id(mind))
        decision = self.decide(mind)
        return replace(decision, exchanges=mesh(mind.active))

    def decide(self, mind):
        keys = {item.text: key for key, item in mind.stimuli.items()}
        update = keys.get("Can I get an update?")
        if update is not None:
            # Both lanes are mid-click and nothing has come back yet.
            self.test.assertEqual(mind.branch_results, {})
            self.test.assertEqual(mind.in_flight, frozenset(OUTCOMES))
            return Decision(replies=(Reply(update, "Both checks are in progress."),))
        if not mind.subtasks:
            return Decision(
                split=tuple(Subtask(key, f"Inspect the {key}") for key in OUTCOMES),
                demands=(Demand("R", another_pass=True, split=len(OUTCOMES)),),
            )
        if not mind.assignments:
            pairs = tuple(zip(OUTCOMES, ("R1", "R2")))
            return Decision(assignments=pairs, branch_actions=tuple(
                BranchAction(key, actor, Action(
                    {"kind": "click", "target": "Inspect"},
                    claims=frozenset({f"{key}/output"}),
                    seen=(("output", "READY"),),
                ))
                for key, actor in pairs
            ))
        pending = {key: seen for key, seen in mind.branch_observations.items()
                   if key not in mind.branch_results}
        landed = {key: seen for key, seen in pending.items()
                  if key not in mind.in_flight}
        follow = {key: seen for key, seen in landed.items()
                  if seen.result in FOLLOW_UP}
        if follow:
            return Decision(branch_actions=tuple(
                BranchAction(key, mind.assignments[key], Action(
                    {"kind": "click", "target": FOLLOW_UP[seen.result]},
                    claims=frozenset({f"{key}/output"}),
                    seen=(("output", seen.result),),
                ))
                for key, seen in follow.items()
            ))
        finished = tuple(key for key, seen in landed.items()
                         if seen.result in FOLLOW_UP.values())
        if finished:
            return Decision(complete_branches=finished)
        if pending:
            # Hands are still moving. Nothing to want; wait for them.
            return Decision()
        return Decision(replies=(Reply(
            keys["Check the order and the invoice"],
            " | ".join(mind.branch_results[key].result for key in OUTCOMES),
        ),))

    async def close(self, mind):
        pass


class OneMindBrowserTest(unittest.IsolatedAsyncioTestCase):
    async def test_the_conversation_answers_while_two_lanes_are_mid_action(self):
        entered = {key: asyncio.Event() for key in OUTCOMES}
        release = asyncio.Event()
        trace = []
        host = ChromiumHost(entered, release, trace)
        adapter = Adapter(self)
        sync = Sync(adapter, host=host)
        merged = sync.send("Check the order and the invoice")
        try:
            # Both lanes are inside a real click in their own Chromium.
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in entered.values())), 20)
            self.assertFalse(merged.done())
            self.assertEqual({name for _, name, _ in trace}, set(OUTCOMES))
            self.assertTrue(all(step == "start" for step, _, _ in trace))

            # A second prompt, answered while both hands are still moving.
            update = sync.send("Can I get an update?")
            self.assertEqual(await asyncio.wait_for(update, 20),
                             "Both checks are in progress.")
            self.assertFalse(merged.done())

            release.set()
            self.assertEqual(await asyncio.wait_for(merged, 30),
                             "Track delivery | Open receipt")
        finally:
            await sync.aclose()

        self.assertEqual(len(adapter.minds), 1, "both prompts must share one mind")
        self.assertCountEqual(host.opened, OUTCOMES)
        self.assertCountEqual(host.released, OUTCOMES)
        self.assertEqual([step for step, _, _ in trace].count("finish"), 4)
        print("Browser trace:", trace)
        print("Active lobes per pass:", adapter.actives)


if __name__ == "__main__":
    unittest.main()
