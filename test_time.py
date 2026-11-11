"""The clock never stops: the world moves between passes, and each pass sees it."""
import asyncio
import unittest

from clock import ManualClock
from clock import Event
from runtime import Decision, Exchange, Reply, Sync


def mesh(active, site="layers.14"):
    return tuple(Exchange(source, target, site, 0.5)
                 for source, target in zip(active, active[1:]))


class Ticker:
    """Answers each stimulus on the pass that sees it and records what it was shown."""

    def __init__(self):
        self.passes = []

    async def coupled_pass(self, mind, links):
        self.passes.append({"now": mind.now, "elapsed": mind.elapsed,
                            "news": tuple((event.at, event.text)
                                          for event in mind.since_last_look())})
        keys = {item.text: key for key, item in mind.stimuli.items()}
        replies = tuple(Reply(key, f"answered {text}") for text, key in keys.items())
        return Decision(replies=replies, exchanges=mesh(mind.active))

    async def close(self, mind):
        pass


async def settle(seconds=0.02):
    await asyncio.sleep(seconds)


class ContinuousTimeTests(unittest.IsolatedAsyncioTestCase):

    async def test_every_pass_is_stamped_with_the_clock(self):
        clock = ManualClock(1_000.0)
        adapter = Ticker()
        sync = Sync(adapter, clock=clock, poll_seconds=0.001)
        await asyncio.wait_for(sync.send("first"), 5)
        clock.advance(30)
        await asyncio.wait_for(sync.send("second"), 5)
        self.assertEqual(adapter.passes[0]["now"], 1_000.0)
        self.assertEqual(adapter.passes[1]["now"], 1_030.0)
        self.assertEqual(adapter.passes[1]["elapsed"], 30.0)
        await sync.aclose()

    async def test_the_clock_keeps_ticking_with_no_lobe_thinking(self):
        clock = ManualClock(1_000.0)
        adapter = Ticker()
        sync = Sync(adapter, clock=clock, poll_seconds=0.001)
        await asyncio.wait_for(sync.send("one"), 5)
        await asyncio.wait_for(sync._pump, 5)
        passes_before = len(adapter.passes)
        clock.advance(45)
        await settle()
        # Nothing was asked and no lobe ran, yet the mind's present moved on.
        self.assertEqual(len(adapter.passes), passes_before)
        self.assertEqual(sync.mind.now, 1_045.0)
        await sync.aclose()

    async def test_the_world_happens_between_passes_and_keeps_its_own_times(self):
        clock = ManualClock(1_000.0)
        adapter = Ticker()
        # The world's own script: each event carries the time it happened.
        world = [Event(1_010.0, "order shipped"), Event(1_030.0, "news: rates moved")]
        released = []

        async def ambient():
            due = [event for event in world
                   if event.at <= clock.now() and event not in released]
            released.extend(due)
            return due

        sync = Sync(adapter, clock=clock, ambient=ambient, poll_seconds=0.001)
        await asyncio.wait_for(sync.send("start"), 5)
        await asyncio.wait_for(sync._pump, 5)
        clock.advance(20)   # now 1020: the shipment has happened
        await settle()
        clock.advance(20)   # now 1040: the news has happened
        await settle()
        await asyncio.wait_for(sync.send("what's new?"), 5)

        news = adapter.passes[-1]["news"]
        self.assertEqual(news, ((1_010.0, "order shipped"), (1_030.0, "news: rates moved")))
        await sync.aclose()

    async def test_a_pass_sees_only_what_happened_since_it_last_looked(self):
        clock = ManualClock(1_000.0)
        adapter = Ticker()
        fired = set()

        async def ambient():
            if clock.now() >= 1_010 and "a" not in fired:
                fired.add("a")
                return (Event(1_010.0, "first event"),)
            return ()

        sync = Sync(adapter, clock=clock, ambient=ambient, poll_seconds=0.001)
        await asyncio.wait_for(sync.send("look once"), 5)
        clock.advance(15)
        await settle()
        await asyncio.wait_for(sync.send("look again"), 5)
        # The second pass shows the event that came after the first pass,
        # and does not repeat anything the first pass already saw.
        self.assertEqual(adapter.passes[1]["news"], ((1_010.0, "first event"),))
        await sync.aclose()


if __name__ == "__main__":
    unittest.main()
