"""Chromium workflow with no generated text passed between vision and action."""
import asyncio
import os
from getpass import getpass

import nnsight
from nnsight import TransformersModel
from playwright.async_api import async_playwright

from vision_share import probe


async def main():
    nnsight.CONFIG.API.APIKEY = os.getenv("NDIF_API_KEY") or getpass("NDIF API key: ")
    model = TransformersModel("Qwen/Qwen3.5-9B", device_map="meta", task="image-text-to-text")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=os.environ["SYNC_CHROMIUM"])
        try:
            page = await browser.new_page(viewport={"width": 560, "height": 250})
            await page.set_content("<h1>&nbsp;</h1>")
            neutral = await page.screenshot()
            await page.set_content("""
            <h1 style="font-size:60px">UNINSPECTED</h1>
            <button onclick="
                document.querySelector('h1').textContent =
                  crypto.getRandomValues(new Uint8Array(1))[0] % 2 ? 'REFUNDED' : 'SHIPPED';
                this.hidden = true;
                document.querySelector('section').hidden = false;
            ">Inspect order</button>
            <section hidden>
              <button onclick="document.querySelector('output').textContent=this.textContent">Refund review</button>
              <button onclick="document.querySelector('output').textContent=this.textContent">Delivery tracking</button>
            </section>
            <output></output>
            """)
            snapshot = previous_target = None
            for _ in range(2):
                target, snapshot, replayed = await asyncio.to_thread(
                    probe, model, await page.screenshot(), neutral,
                    workflow=True, prior=snapshot, return_snapshot=True,
                )
                if previous_target is not None and replayed != previous_target:
                    raise RuntimeError("Replayed activation did not reproduce the previous decision")
                previous_target = target
                await page.get_by_role("button", name=target, exact=True).click(timeout=3000)
                print("Clicked:", target, flush=True)
            actual = await page.locator("h1").inner_text()
            outcome = await page.locator("output").inner_text()
            assert outcome == {"REFUNDED": "Refund review", "SHIPPED": "Delivery tracking"}[actual]
            print("Verified:", actual, outcome, flush=True)
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
