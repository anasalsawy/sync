"""Experimental UI-TARS model transport for NDIF."""
import base64
import io
import json
import os
import sys

import nnsight
from nnsight import TransformersModel
from PIL import Image

nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
nnsight.CONFIG.APP.REMOTE_LOGGING = False
model = TransformersModel("Qwen/Qwen3.5-9B", device_map="meta", task="image-text-to-text")

def predict(payload):
    text, images = [], []
    for message in payload["messages"]:
        content = message["content"]
        if isinstance(content, str):
            text.append(f'{message["role"]}: {content}')
        else:
            for part in content:
                if part["type"] == "text":
                    text.append(part["text"])
                elif part["type"] == "image_url":
                    uri = part["image_url"]["url"]
                    images.append(Image.open(io.BytesIO(base64.b64decode(uri.split(",",1)[1]))).convert("RGB"))
    text.append("Browser observation: " + json.dumps(payload["browser_observation"]))
    text.append("Use the current page and last action. If a URL failed, choose a different action or source.")
    if payload.get("action_retry"):
        text.append(payload["action_retry"] + " Use the listed UI-TARS syntax and begin with Action:.")
    image = images[-1]
    def conversation(pixels):
        return [{"role":"user","content":[
            {"type":"image","image":pixels},
            {"type":"text","text":"\n".join(text) + "\nReturn the next UI-TARS action."},
        ]}]
    batch = model.processor.apply_chat_template(
        [conversation(image), conversation(Image.new("RGB", image.size))],
        tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", padding=True, enable_thinking=False,
    )
    layer = model.model.language_model.layers[-1]
    with model.generate(batch, max_new_tokens=96, do_sample=False, remote=True) as tracer:
        layer.output[1, -1, :] = layer.output[0, -1, :].clone()
        tokens = tracer.result.save()
    return model.processor.tokenizer.decode(
        tokens[1, batch["input_ids"].shape[-1]:], skip_special_tokens=True,
    ).strip()

for line in sys.stdin:
    try:
        answer = predict(json.loads(line))
        print(json.dumps({"choices":[{"index":0,"message":{"role":"assistant","content":answer},"finish_reason":"stop"}]}), flush=True)
    except Exception as exc:
        print(json.dumps({"error":{"message":f"{type(exc).__name__}: {exc}"}}), flush=True)
