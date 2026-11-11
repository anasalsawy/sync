import asyncio, io, os
from PIL import Image
import nnsight
from live import TransformersModel, async_playwright

async def main():
    nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
    vision = TransformersModel("Qwen/Qwen3.5-9B", device_map="meta", task="image-text-to-text")
    def read(png, question="Read the large order status word. Answer with that word only."):
        messages = [{"role":"user","content":[{"type":"image","image":Image.open(io.BytesIO(png)).convert("RGB")},{"type":"text","text":question}]}]
        inputs=vision.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt",enable_thinking=False)
        with vision.generate(inputs,max_new_tokens=12,do_sample=False,remote=True) as tracer:
            ids=tracer.result.save()
        return vision.processor.tokenizer.decode(ids[0,inputs["input_ids"].shape[-1]:],skip_special_tokens=True)
    async with async_playwright() as p:
        b=await p.chromium.launch(executable_path=os.environ["SYNC_CHROMIUM"])
        page=await b.new_page(viewport={"width":500,"height":300})
        await page.set_content("""<button onclick="document.querySelector('h1').textContent = crypto.getRandomValues(new Uint8Array(1))[0]%2 ? 'REFUNDED':'SHIPPED'; this.hidden=true;document.querySelector('section').hidden=false">Inspect order</button>
        <h1 style="font-size:64px"></h1><section hidden><button onclick="document.querySelector('p').textContent='Refund review'">Refund review</button><button onclick="document.querySelector('p').textContent='Delivery tracking'">Delivery tracking</button></section><p></p>""")
        await page.get_by_role("button",name="Inspect order").click()
        seen=(await asyncio.to_thread(read,await page.screenshot())).upper()
        print("Screenshot interpretation:",seen,flush=True)
        assert ("REFUNDED" in seen) != ("SHIPPED" in seen)
        target="Refund review" if "REFUNDED" in seen else "Delivery tracking"
        wrong="Delivery tracking" if target=="Refund review" else "Refund review"
        prompt=f"Order status: {seen}. Rule: REFUNDED orders go to Refund review. SHIPPED orders go to Delivery tracking."
        for proposal,expected in ((wrong,"B"),(target,"A")):
            response = await asyncio.to_thread(read, await page.screenshot(), f"{prompt} Proposed click: {proposal}. Does that follow the rule? Answer only APPROVE or REJECT.")
            verdict = {"APPROVE":"A","REJECT":"B"}.get(response.strip().upper().rstrip("."))
            print("Verifier:",proposal,response,flush=True)
            assert verdict==expected,(proposal,verdict)
            if verdict=="A":
                await page.get_by_role("button",name=proposal,exact=True).click()
            else:
                assert await page.locator("p").inner_text()==""
        status=await page.locator("h1").inner_text()
        outcome=await page.locator("p").inner_text()
        assert outcome=={"REFUNDED":"Refund review","SHIPPED":"Delivery tracking"}[status]
        print("Verified browser outcome:",status,outcome,flush=True)
asyncio.run(main())
