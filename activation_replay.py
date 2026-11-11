from getpass import getpass
import os
import nnsight
from nnsight import TransformersModel

nnsight.CONFIG.API.APIKEY = os.getenv("NDIF_API_KEY") or getpass()
model = TransformersModel("Qwen/Qwen3-0.6B", device_map="meta", task="text-generation")
layer = model.model.layers[-1]
labels = [model.tokenizer.encode(" "+word, add_special_tokens=False)[0] for word in ("Paris", "Rome")]
with model.trace(["The capital of France is", "The capital of Italy is"], remote=True):
    snapshot = layer.output[:, -1, :].clone().save()
prompt = "The capital of a country is"
with model.trace([prompt, prompt, prompt], remote=True):
    layer.output[:2, -1, :] = snapshot.to(layer.output.device)
    scores = model.lm_head.output[:, -1, labels].float().save()
print("Scores [Paris, Rome]:", scores.tolist())
assert scores[:2].argmax(dim=-1).tolist() == [0, 1]
