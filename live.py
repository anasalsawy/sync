"""Live NDIF and Chromium integration for a controlled browser task."""
import asyncio
import os
from getpass import getpass

import nnsight
from nnsight import TransformersModel
from playwright.async_api import async_playwright

from runtime import Action, Decision, Observation, Sync


class PageBrowser:
    def __init__(self, page):
        self.page = page

    async def observe(self):
        status = await self.page.locator("#status").inner_text()
        return Observation(status, await self.page.screenshot(), status)

    async def execute(self, action):
        if (await self.observe()).revision != action.revision:
            return None
        if action.kind != "click" or action.target not in {"Save", "Cancel"}:
            raise ValueError("Action outside the test fixture")
        await self.page.get_by_role("button", name=action.target, exact=True).click()
        print(f"Chromium clicked {action.target}", flush=True)
        return await self.observe()


class NDIF:
    def __init__(self):
        self.model = TransformersModel("Qwen/Qwen3-0.6B", device_map="meta", task="text-generation")

    async def coupled_pass(self, run, links):
        if run.passes == 0:
            route = await asyncio.to_thread(self.classify, (
                f"User instruction: {run.instruction}\n"
                "A = respond without clicking. B = delegate a browser click.\n"
                "The appropriate route is"
            ))
            return Decision(r_width=1) if route == "B" else Decision(done=True, reply="No action.")
        if run.passes == 1:
            if "R" not in links["R1"] or "R1" not in links["R"]:
                raise RuntimeError("R family sharing link missing")
            target = await asyncio.to_thread(self.choose, run.instruction)
            verdict = await asyncio.to_thread(self.classify, (
                f"User instruction: {run.instruction}\nProposed click: {target}\n"
                "A = approve if the click follows the instruction. B = reject otherwise.\n"
                "The correct verdict is"
            ))
            if verdict == "B":
                return Decision(done=True, reply="C rejected the action.")
            return Decision(delegate="R1", action=Action(run.observation.revision, "click", target))
        return Decision(done=True, reply=f"Browser status: {run.observation.result}")

    def classify(self, prompt):
        model = self.model
        labels = [model.tokenizer.encode(f" {label}", add_special_tokens=False)[0] for label in ("A", "B")]
        with model.trace(prompt, remote=True):
            scores = model.lm_head.output[0, -1, labels].float().save()
        choice = ("A", "B")[scores.argmax().item()]
        print(f"NDIF control readout: {choice}", flush=True)
        return choice

    def choose(self, instruction):
        model = self.model
        context = f"User instruction: {instruction}\nAvailable buttons: A = Save, B = Cancel."
        tokens = [model.tokenizer.encode(f" {label}", add_special_tokens=False) for label in ("A", "B")]
        if any(len(token) != 1 for token in tokens):
            raise RuntimeError("Expected single-token action labels")
        labels = [token[0] for token in tokens]
        with model.trace(remote=True) as tracer:
            ready = tracer.barrier(2)
            with tracer.invoke(context + "\nThe supervisor recommends button"):
                r = model.model.layers[0].output.clone()
                ready()
                model.model.layers[0].output[:, -1, :] = child[:, -1, :]
            with tracer.invoke(context + "\nTo follow the user instruction, click button"):
                child = model.model.layers[0].output.clone()
                ready()
                model.model.layers[0].output[:, -1, :] = r[:, -1, :]
                scores = model.lm_head.output[0, -1, labels].float().save()
        print(f"NDIF action scores [Save, Cancel]: {scores.tolist()}", flush=True)
        return ("Save", "Cancel")[scores.argmax().item()]

    async def close(self, run):
        pass


async def main():
    nnsight.CONFIG.API.APIKEY = os.getenv("NDIF_API_KEY") or getpass("NDIF API key: ")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=os.getenv("SYNC_CHROMIUM", "/usr/bin/chromium"))
        try:
            page = await browser.new_page()
            await page.set_content("""
                <button onclick="document.querySelector('#status').textContent='Saved'">Save</button>
                <button onclick="document.querySelector('#status').textContent='Cancelled'">Cancel</button>
                <p id="status">Unsaved</p>
            """)
            sync = Sync(NDIF(), PageBrowser(page))
            print(await sync.start("Save the form. Do not cancel."), flush=True)
            assert await page.locator("#status").inner_text() == "Saved"
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
