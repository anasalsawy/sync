"""Live NDIF decisions through the Sync runtime and real Chromium."""
import asyncio
import os
import unittest
from contextlib import asynccontextmanager

import nnsight
from nnsight import TransformersModel

from runtime import Action, BranchAction, Decision, Subtask, Sync
from test_browser_concurrency import BrowserConcurrencyTest


class NDIFInference:
    def __init__(self):
        nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
        self.model = TransformersModel(
            os.getenv("SYNC_MODEL", "Qwen/Qwen3-8B"),
            device_map="meta", task="text-generation",
        )

    def _choose(self, questions):
        model = self.model
        labels = [model.tokenizer.encode(" " + label, add_special_tokens=False)
                  for label in "ABCD"]
        if any(len(ids) != 1 for ids in labels):
            raise ValueError("Expected single-token choices")
        ids = [tokens[0] for tokens in labels]
        prompts = [
            f"{question}\n" + "\n".join(
                f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)
            ) + "\nThe correct option is"
            for question, choices in questions
        ]
        with model.trace(prompts, remote=True):
            scores = model.lm_head.output[:, -1, ids].float().save()
        results = [choices[row[:len(choices)].argmax().item()]
                   for row, (_, choices) in zip(scores, questions, strict=True)]
        print("NDIF decisions:", results, flush=True)
        return results

    async def choose(self, *questions):
        return await asyncio.to_thread(self._choose, questions)

    async def coupled_pass(self, run, links):
        if run.instruction == "Can I get an update?":
            answer, = await self.choose((
                "The order and invoice checks have started and neither has finished. What is their status?",
                ("Both checks are in progress.", "Both checks finished.", "No checks have started."),
            ))
            return Decision(done=True, reply=answer)
        if run.passes == 0:
            plan, = await self.choose((
                f"User request: {run.instruction}. Which independent subtasks cover the whole request?",
                (("invoice",), ("order", "invoice"), ("order",)),
            ))
            return Decision(split=tuple(Subtask(key, f"Check the {key}") for key in plan),
                            r_width=len(plan))
        if len(run.assignments) < len(run.subtasks):
            routes = await self.choose(*[
                (f"Delegate the {key} check. R is the parent. {actor} is the available worker for {key}. Who should act?",
                 ("R", actor))
                for key, actor in zip(run.subtasks, (name for name in run.active if name.startswith("R") and name != "R"))
            ])
            assignments = tuple(zip(run.subtasks, routes, strict=True))
            selected = await self.choose(*[
                (f"The {key} is uninspected. Choose the next browser action.",
                 ("Open receipt", "Inspect", "Track delivery")) for key in run.subtasks
            ])
            return Decision(assignments=assignments, branch_actions=tuple(
                BranchAction(key, actor, Action(run.branch_observations[key].revision, "click", target))
                for (key, actor), target in zip(assignments, selected, strict=True)
            ))
        if not run.branch_results:
            observed = tuple(run.branch_observations.items())
            if all(obs.result in ("Track delivery", "Open receipt") for _, obs in observed):
                return Decision(complete_branches=tuple(key for key, _ in observed))
            selected = await self.choose(*[
                (f"The {key} browser reports {obs.result}. A paid invoice needs a receipt. A shipped order needs tracking. Choose the next action.",
                 ("Track delivery", "Inspect", "Open receipt")) for key, obs in observed
            ])
            return Decision(branch_actions=tuple(
                BranchAction(key, run.assignments[key], Action(obs.revision, "click", target))
                for (key, obs), target in zip(observed, selected, strict=True)
            ))
        facts = {key: obs.result for key, obs in run.branch_results.items()}
        reply, = await self.choose((
            f"Verified browser results: {facts}. Which final report includes both results accurately?",
            ("Track delivery | Open receipt", "Track delivery only", "Open receipt only"),
        ))
        return Decision(done=True, reply=reply)

    async def close(self, run):
        pass


class LiveNDIFTest(unittest.IsolatedAsyncioTestCase):
    async def test_split_route_followup_merge(self):
        entered = {key: asyncio.Event() for key in ("order", "invoice")}
        release = asyncio.Event()
        trace = []
        adapter = NDIFInference()
        fixtures = BrowserConcurrencyTest().fixtures(
            entered, release, trace, {"SHIPPED": "Track delivery", "PAID": "Open receipt"},
        )
        async with fixtures as make_fixture:
            outcomes = {"order": "SHIPPED", "invoice": "PAID"}
            verified = {}

            class ChromiumHost:
                supports_subagents = False

                @asynccontextmanager
                async def open_instance(self, subtask):
                    fixture = BrowserConcurrencyTest().fixtures(
                        entered, release, trace,
                        {"SHIPPED": "Track delivery", "PAID": "Open receipt"},
                    )
                    async with fixture as make_worker:
                        worker = await make_worker(subtask.key, outcomes[subtask.key])
                        yield worker
                        verified[subtask.key] = (await worker.observe()).result

            sync = Sync(adapter, await make_fixture("parent"), host=ChromiumHost())
            parent = sync.start("Check order and invoice")
            try:
                async def verify_overlap():
                    await asyncio.gather(*(event.wait() for event in entered.values()))
                overlap = asyncio.create_task(verify_overlap())
                try:
                    completed, _ = await asyncio.wait((parent, overlap), timeout=120,
                                                       return_when=asyncio.FIRST_COMPLETED)
                    if parent in completed:
                        await parent
                    self.assertIn(overlap, completed)
                    self.assertTrue(all(event[0] == "start" for event in trace))
                    update = sync.start("Can I get an update?", browser=await make_fixture("update"))
                    self.assertEqual(await asyncio.wait_for(update, 60), "Both checks are in progress.")
                    self.assertFalse(parent.done())
                    release.set()
                    self.assertEqual(await asyncio.wait_for(parent, 90), "Track delivery | Open receipt")
                    self.assertEqual(verified, {"order": "Track delivery", "invoice": "Open receipt"})
                    print("Verified Chromium trace:", trace)
                finally:
                    overlap.cancel()
                    await asyncio.gather(overlap, return_exceptions=True)
            finally:
                await sync.interrupt()
