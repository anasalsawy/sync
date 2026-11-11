"""The decision itself: how much attention mass lands on each peer's slots.

Computes query.key over the positions a lobe can see, softmaxes, and reports
the share of attention the lobe spends on each peer's slot band versus its
own tokens. This is the literal mechanism, not a downstream consequence.
"""
import asyncio, os, sys
import torch
import nnsight
from nnsight import TransformersModel
from ndif import channel_prompt, SLOTS_PER_PEER

LOBES = ("L", "R", "Embedder")
TASK = {
    "L": "Conversational voice. The user asks: book my usual room, you know how I like it.",
    "R": "Execution supervisor. Carry out: book the usual room per the user's preferences.",
    "Embedder": "Memory: Anas prefers mornings. Room 4 was double-booked. He hates room 7.",
}


def main():
    nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-8B"
    model = TransformersModel(name, device_map="meta", task="text-generation")
    layers = list(model.model.layers)
    W = SLOTS_PER_PEER
    N = len(LOBES)
    texts = [channel_prompt(TASK[l], N - 1, W) for l in LOBES]
    probe_layers = [len(layers) // 4, len(layers) // 2, 3 * len(layers) // 4]

    # One trace per probe layer: assignments into an outer dict do not survive
    # the compiled trace body, so each capture gets its own run.
    saved = {}
    for probe in probe_layers:
        with model.trace(texts, remote=True):
            for li in range(2, len(layers) - 2):
                layer = layers[li]
                if li == probe:
                    q_at = layer.self_attn.q_proj.output.save()
                    k_at = layer.self_attn.k_proj.output.save()
                peer = layer.output[:, -W:, :].clone()
                for receiver in range(N):
                    senders = [s for s in range(N) if s != receiver]
                    for pos, sender in enumerate(senders):
                        for slot in range(W):
                            layer.output[receiver, pos * W + slot, :] = peer[sender, slot]
        saved[probe] = (q_at, k_at)

    cfg = model.config
    heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // heads)

    print(f"{name}: {heads} heads, {kv_heads} kv-heads, head_dim {hd}, "
          f"{W} slots/peer")
    print("\nShare of each lobe's attention at its final token:")
    for li in probe_layers:
        q, k = saved[li]
        B, S, _ = q.shape
        qh = q.view(B, S, heads, hd).transpose(1, 2)        # [B, H, S, hd]
        kh = k.view(B, S, kv_heads, hd).transpose(1, 2)     # [B, KV, S, hd]
        rep = heads // kv_heads
        kh = kh.repeat_interleave(rep, dim=1)
        print(f"\n  layer {li}")
        for r, lobe in enumerate(LOBES):
            scores = (qh[r, :, -1, :].unsqueeze(1) @ kh[r].transpose(-1, -2)).squeeze(1)
            weights = torch.softmax(scores.float() / hd ** 0.5, dim=-1).mean(0)
            senders = [s for s in range(N) if s != r]
            parts = []
            for pos, s in enumerate(senders):
                share = weights[pos * W:(pos + 1) * W].sum().item()
                parts.append(f"{LOBES[s]} {share*100:5.1f}%")
            own = weights[W * (N - 1):].sum().item()
            print(f"    {lobe:>9} reads:  " + "   ".join(parts)
                  + f"   | own tokens {own*100:5.1f}%")


if __name__ == "__main__":
    main()
