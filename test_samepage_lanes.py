"""Several lanes working one real page at once, through the runtime.

The host hands the same surface to every subtask, so the lanes share one
Chromium page. They stay out of each other's way by claiming different
elements on it, and staleness is judged per claim, so one lane's write never
invalidates another lane's pending act.

Covers the three things that decide whether this is real:
  * four lanes filling disjoint fields concurrently, nothing lost
  * two lanes reaching for the same element, refused before dispatch
  * two lanes that both need the single focus, taken in turn and not fought
    over, which is the worst case and still correct
"""
import asyncio
import os
import time
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace

from playwright.async_api import async_playwright

from runtime import (
    Action, BranchAction, Decision, Demand, Exchange, MAIN,
    Observation, Reply, Subtask, Sync,
)

FIELDS = 12
VALUES = {f"f{index}": f"value-{index}" for index in range(FIELDS)}

PAGE = """
<form id="big"></form>
<script>
  const form = document.getElementById('big');
  const state = {};
  for (let index = 0; index < %d; index++) {
    const input = document.createElement('input');
    input.id = 'f' + index;
    input.addEventListener('input', event => {
      state[input.id] = event.target.value;
      input.value = state[input.id];
    });
    form.appendChild(input);
  }
</script>
""" % FIELDS

NODE_WRITE = """
([ids, values]) => {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype, 'value').set;
  ids.forEach((id, slot) => {
    const element = document.getElementById(id);
    setter.call(element, values[slot]);
    element.dispatchEvent(new Event('input', {bubbles: true}));
  });
}
"""

READ_ALL = """
() => Object.fromEntries(Array.from(document.querySelectorAll('input'))
  .map(element => [element.id, element.value]))
"""


def mesh(active, site="layers.14"):
    return tuple(Exchange(source, target, site, 0.5)
                 for source, target in zip(active, active[1:]))


def lanes_of(count):
    ids = list(VALUES)
    size = len(ids) // count
    return [tuple(ids[index * size:(index + 1) * size]) for index in range(count)]


class PageWorker:
    """One page. Its claims are its input elements."""

    def __init__(self, page, trace):
        self.page = page
        self.trace = trace

    async def observe(self):
        values = await self.page.evaluate(READ_ALL)
        filled = sum(1 for value in values.values() if value)
        return Observation(None, tuple(sorted(values.items())), f"{filled} filled")

    async def execute(self, action):
        current = await self.observe()
        if any(current.revision(name) != was for name, was in action.seen):
            return None
        fields = action.payload["fields"]
        label = action.payload.get("lane", "?")
        self.trace.append(("start", label, time.perf_counter()))
        if action.payload["kind"] == "focus":
            # Real keystroke typing: takes the page's single focus.
            for field, value in fields.items():
                await self.page.fill(f"#{field}", value)
        else:
            # Focus-free node writes: no contention with other lanes.
            await self.page.evaluate(
                NODE_WRITE, [list(fields), [fields[name] for name in fields]]
            )
        self.trace.append(("finish", label, time.perf_counter()))
        return await self.observe()


class SharedPageHost:
    """A host with one surface. Every lane gets the same page."""

    supports_subagents = False

    def __init__(self, page, trace):
        self.worker = PageWorker(page, trace)
        self.handouts = []

    @asynccontextmanager
    async def open_instance(self, subtask):
        self.handouts.append(subtask.key)
        yield self.worker

    @asynccontextmanager
    async def open_subagent(self, subtask):  # pragma: no cover - not native
        raise AssertionError("This host has no native delegation")
        yield


class Filler:
    """Splits the form across lanes and lets them all write at once."""

    def __init__(self, test, count, kind="write", collide=False):
        self.test = test
        self.groups = lanes_of(count)
        self.kind = kind
        self.collide = collide
        self.count = count

    async def coupled_pass(self, mind, links):
        decision = self.decide(mind)
        return replace(decision, exchanges=mesh(mind.active))

    def decide(self, mind):
        keys = {item.text: key for key, item in mind.stimuli.items()}
        if not mind.subtasks:
            return Decision(
                split=tuple(Subtask(f"lane{index}", f"fill {group}")
                            for index, group in enumerate(self.groups)),
                demands=(Demand("R", another_pass=True, split=self.count),),
            )
        if not mind.assignments:
            pairs = tuple((f"lane{index}", f"R{index + 1}")
                          for index in range(self.count))
            seen = dict(mind.branch_observations[pairs[0][0]].revisions)
            actions = []
            for index, (key, actor) in enumerate(pairs):
                group = self.groups[0] if self.collide else self.groups[index]
                actions.append(BranchAction(key, actor, Action(
                    {"kind": self.kind, "lane": key,
                     "fields": {name: VALUES[name] for name in group}},
                    claims=frozenset(group),
                    seen=tuple((name, seen[name]) for name in group),
                    exclusive=self.kind == "focus",
                )))
            return Decision(assignments=pairs, branch_actions=tuple(actions))
        open_keys = tuple(key for key in mind.subtasks
                          if key not in mind.branch_results)
        landed = tuple(key for key in open_keys if key not in mind.in_flight)
        if landed:
            return Decision(complete_branches=landed)
        if open_keys:
            return Decision()
        return Decision(replies=(Reply(keys["fill the form"],
                                       mind.branch_results[open_keys[0] if open_keys
                                                           else "lane0"].result),))

    async def close(self, mind):
        pass


@asynccontextmanager
async def one_page(trace):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=os.getenv("SYNC_CHROMIUM", "/usr/bin/chromium")
        )
        try:
            page = await browser.new_page(viewport={"width": 900, "height": 700})
            await page.set_content(PAGE)
            yield page
        finally:
            await browser.close()


class SamePageLaneTest(unittest.IsolatedAsyncioTestCase):

    async def drive(self, count, *, kind="write", collide=False):
        trace = []
        async with one_page(trace) as page:
            host = SharedPageHost(page, trace)
            sync = Sync(Filler(self, count, kind, collide), host=host)
            answer = sync.send("fill the form")
            try:
                started = time.perf_counter()
                await asyncio.wait_for(answer, 30)
                elapsed = (time.perf_counter() - started) * 1000
                final = await page.evaluate(READ_ALL)
            finally:
                await sync.aclose()
            return host, sync, trace, final, elapsed

    async def test_four_lanes_fill_one_page_at_once(self):
        host, sync, trace, final, elapsed = await self.drive(4)
        self.assertEqual(final, VALUES, "every field must land exactly once")
        self.assertEqual(sync.mind.events, [],
                         "per-claim staleness must not discard any lane's act")
        self.assertEqual(len(host.handouts), 4)
        self.assertEqual(len(sync._surfaces), 0, "the shared page closed once")
        starts = [when for step, _, when in trace if step == "start"]
        finishes = [when for step, _, when in trace if step == "finish"]
        self.assertEqual(len(starts), 4)
        # Overlap: every lane had started before the first one finished.
        self.assertLess(max(starts), min(finishes))
        print(f"4 lanes, one page: {elapsed:.1f} ms, all {FIELDS} fields correct")

    async def test_two_lanes_cannot_claim_the_same_element(self):
        with self.assertRaisesRegex(PermissionError, "Two lanes claimed"):
            await self.drive(2, collide=True)

    async def test_the_single_focus_is_taken_in_turn_not_fought_over(self):
        host, sync, trace, final, elapsed = await self.drive(2, kind="focus")
        # The worst case: both lanes need the one focus. They serialise, and
        # nothing is corrupted the way concurrent unarbitrated typing is.
        self.assertEqual(final, VALUES)
        spans = {}
        for step, lane, when in trace:
            spans.setdefault(lane, {})[step] = when
        first, second = sorted(spans.values(), key=lambda span: span["start"])
        self.assertGreaterEqual(second["start"], first["finish"] - 1e-6)
        print(f"2 focus lanes, taken in turn: {elapsed:.1f} ms, "
              f"all {FIELDS} fields correct")


if __name__ == "__main__":
    unittest.main()
