"""Live image-to-action hidden-state probe with no inter-lobe text output."""
import asyncio
import io
import os
from getpass import getpass

import nnsight
from nnsight import TransformersModel
from PIL import Image
from playwright.async_api import async_playwright


def probe(model, screenshot, neutral, *, workflow=False, prior=None, return_snapshot=False):
    question = (
        "Choose the next action. If the status is REFUNDED choose A. If SHIPPED choose B. If the order is UNINSPECTED choose C. Reply with only the letter."
        if workflow else
        "Read the visible order status. Reply with A if REFUNDED or B if SHIPPED. Output only the letter."
    )
    def conversation(png):
        image = Image.open(io.BytesIO(png)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        return messages

    conversations = [conversation(screenshot), conversation(neutral), conversation(neutral)]
    if prior is not None:
        conversations.append(conversation(neutral))
    batch = model.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=False, padding=True,
        )

    labels = [model.processor.tokenizer.encode(x, add_special_tokens=False)[0] for x in ("ABC" if workflow else "AB")]
    layer = model.model.language_model.layers[-1]
    with model.trace(batch, remote=True):
        snapshot = layer.output[0, -1, :].clone().save()
        layer.output[1, -1, :] = snapshot
        if prior is not None:
            layer.output[3, -1, :] = prior.to(layer.output.device)
            replayed = model.lm_head.output[3, -1, labels].float().save()
        patched = model.lm_head.output[1, -1, labels].float().save()
        baseline = model.lm_head.output[2, -1, labels].float().save()
    print("Baseline and activation-patched action scores:", baseline.tolist(), patched.tolist(), flush=True)
    assert not patched.equal(baseline)
    actions = ("Refund review", "Delivery tracking", "Inspect order")
    target = actions[patched.argmax().item()]
    if return_snapshot:
        previous = actions[replayed.argmax().item()] if prior is not None else None
        return target, snapshot.detach().cpu(), previous
    return target


async def main():
    nnsight.CONFIG.API.APIKEY = os.getenv("NDIF_API_KEY") or getpass("NDIF API key: ")
    model = TransformersModel("Qwen/Qwen3.5-9B", device_map="meta", task="image-text-to-text")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=os.environ["SYNC_CHROMIUM"])
        try:
            page = await browser.new_page(viewport={"width": 480, "height": 240})
            await page.set_content('<h1 style="font-size:60px">&nbsp;</h1>')
            neutral = await page.screenshot()
            for status, expected in (("REFUNDED", "Refund review"), ("SHIPPED", "Delivery tracking")):
                await page.set_content(f"""<h1 style="font-size:60px">{status}</h1>
                <button onclick="document.querySelector('output').textContent=this.textContent">Refund review</button>
                <button onclick="document.querySelector('output').textContent=this.textContent">Delivery tracking</button>
                <output></output>""")
                target = await asyncio.to_thread(probe, model, await page.screenshot(), neutral)
                await page.get_by_role("button", name=target, exact=True).click()
                outcome = await page.locator("output").inner_text()
                print(status, outcome, flush=True)
                assert outcome == expected
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
