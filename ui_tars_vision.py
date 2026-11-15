"""UI-TARS observations inside the existing Sync NDIF lobe batch."""
from __future__ import annotations

import base64
import binascii
from io import BytesIO
import os
import re

from PIL import Image

LOBE_ROWS = ("L", "R", "B", "V", "C", "R0", "R1", "R2")
V_ROW = LOBE_ROWS.index("V")
FIRST_WORKER = LOBE_ROWS.index("R0")
IMAGE_LIMIT = 8 * 1024 * 1024
_FRAME = re.compile(r"data:image/(?:png|jpeg|webp);base64,([A-Za-z0-9+/=]+)\Z")


def action_context(messages):
    """Retain the UI-TARS transcript and only the newest screen observation."""
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    clean, latest = [], None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system", "user", "assistant",
        }:
            raise ValueError("invalid GUI message")
        content = message.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("invalid GUI content part")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"type": "text", "text": part["text"]})
                elif message["role"] == "user" and part.get("type") == "image_url":
                    image = part.get("image_url")
                    latest = (parts, len(parts), image.get("url") if isinstance(image, dict) else None)
                else:
                    raise ValueError("unsupported GUI content part")
        else:
            raise ValueError("invalid GUI message content")
        clean.append({"role": message["role"], "content": parts})
    if latest is None or not isinstance(latest[2], str):
        raise ValueError("a screenshot data URL is required")
    parts, index, url = latest
    if len(url) > IMAGE_LIMIT * 4 // 3 + 64 or not (match := _FRAME.fullmatch(url)):
        raise ValueError("invalid or oversized screenshot")
    try:
        raw = base64.b64decode(match[1], validate=True)
        if len(raw) > IMAGE_LIMIT:
            raise ValueError("oversized screenshot")
        with Image.open(BytesIO(raw)) as image:
            if image.width * image.height > 16_777_216:
                raise ValueError("screenshot exceeds the pixel limit")
            screenshot = image.convert("RGB")
    except (OSError, binascii.Error, Image.DecompressionBombError) as exc:
        raise ValueError("invalid screenshot image") from exc
    parts.insert(index, {"type": "image"})
    return [row for row in clean if row["content"]], screenshot


def lobe_conversations(messages, screenshot):
    """Only V receives pixels. All rows retain the task and action history."""
    rows = []
    for lobe in LOBE_ROWS:
        conversation = []
        for message in messages:
            content = [
                {"type": "image", "image": screenshot} if part["type"] == "image" else part
                for part in message["content"]
                if part["type"] != "image" or lobe == "V"
            ]
            if content:
                conversation.append({"role": message["role"], "content": content})
        directions = {
            "L": "Track user intent and task progress.",
            "B": "Track relevant context and previous actions.",
            "V": "Attend to the current screenshot and ground the action in visible evidence.",
            "C": "Check the next action against the task and visible state.",
        }
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": directions.get(lobe,
                        "Use the visual state from V and emit one valid UI-TARS action.")}],
        })
        rows.append(conversation)
    return rows


class UITarsVisionRuntime:
    def __init__(self, model_name):
        from threading import local

        self.model_name = model_name
        self._request_models = local()

    def _model(self):
        import nnsight
        from nnsight import TransformersModel

        model = getattr(self._request_models, "model", None)
        if model is None:
            nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
            model = TransformersModel(
                self.model_name, device_map="meta", task="image-text-to-text",
            )
            self._request_models.model = model
        return model

    def predict(self, messages, screenshot, max_tokens):
        model = self._model()
        batch = model.processor.apply_chat_template(
            lobe_conversations(messages, screenshot),
            tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True, enable_thinking=False,
        )
        layers = model.model.language_model.layers
        layer = layers[-2]
        with model.generate(batch, max_new_tokens=max_tokens,
                            do_sample=False, remote=True) as tracer:
            for _ in tracer.iter[:]:
                # Clone before writing so each receiver sees the same sender state.
                visual = layer.output[V_ROW, -1, :].clone()
                peers = layer.output[FIRST_WORKER:, -1, :].clone()
                peer_mean = peers.mean(dim=0)
                # A residual keeps each worker's own state while adding V and its peers.
                layer.output[FIRST_WORKER:, -1, :] = (
                    peers * 0.25 + visual * 0.65 + (3 * peer_mean - peers) * 0.05
                )
                for row in (0, 1, 2, 4):
                    layer.output[row, -1, :] = (
                        layer.output[row, -1, :] * 0.75 + visual * 0.25
                    )
            tokens = tracer.result.save()

        prompt_length = batch["input_ids"].shape[-1]
        tokenizer = model.processor.tokenizer
        for row in range(FIRST_WORKER, len(LOBE_ROWS)):
            reply = tokenizer.decode(tokens[row, prompt_length:], skip_special_tokens=True).strip()
            if re.search(r"(?m)^Action:\s*[a-z_]+\(", reply):
                return reply
        raise ValueError("the Sync workers produced no UI-TARS action")
