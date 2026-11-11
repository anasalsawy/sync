import os
from getpass import getpass
import nnsight
from nnsight import TransformersModel

nnsight.CONFIG.API.APIKEY = os.getenv("NDIF_API_KEY") or getpass("NDIF API key: ")
model = TransformersModel("Qwen/Qwen3-0.6B", device_map="meta", task="text-generation")
prompts = ["The capital of France is", "The capital of Italy is"]
with model.trace(remote=True) as tracer:
    ready = tracer.barrier(2)
    with tracer.invoke(prompts[0]):
        left_signal = model.model.layers[0].output.clone()
        ready()
        model.model.layers[0].output[:, -1, :] = right_signal[:, -1, :]
        left_logits = model.lm_head.output[:, -1, :].save()
    with tracer.invoke(prompts[1]):
        right_signal = model.model.layers[0].output.clone()
        ready()
        model.model.layers[0].output[:, -1, :] = left_signal[:, -1, :]
        right_logits = model.lm_head.output[:, -1, :].save()
    with tracer.invoke(prompts[0]):
        left_baseline = model.lm_head.output[:, -1, :].save()
    with tracer.invoke(prompts[1]):
        right_baseline = model.lm_head.output[:, -1, :].save()

for label, patched, baseline in [
    ("A", left_logits, left_baseline), ("B", right_logits, right_baseline)
]:
    shift = (patched.softmax(-1) - baseline.softmax(-1)).abs().max().item()
    print(f"{label} maximum next-token probability change: {shift:.8f}")
    assert shift > 1e-6
