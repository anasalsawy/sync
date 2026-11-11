"""Same model, same lobes, different tasks. Does the routing actually change?"""
import asyncio, os, sys
import nnsight
from ndif import LobePrompt, NDIFChannels

SCENARIOS = {
    "memory-heavy": [
        ("L", "Conversational voice. The user asks: book my usual room, you know how I like it."),
        ("R", "Execution supervisor. Carry out: book the usual room per the user's known preferences."),
        ("Embedder", "Memory: Anas prefers mornings. Room 4 was double-booked last week. He hates room 7."),
    ],
    "execution-only": [
        ("L", "Conversational voice. The user asks: click the blue Submit button at the bottom."),
        ("R", "Execution supervisor. Carry out: click the blue Submit button at coordinates 400,900."),
        ("Embedder", "Memory: no stored preferences are relevant to this request."),
    ],
    "pure-chat": [
        ("L", "Conversational voice. The user asks: what do you think about the weather today?"),
        ("R", "Execution supervisor. There is no task to execute right now."),
        ("Embedder", "Memory: Anas lives in Cairo. He dislikes small talk about weather."),
    ],
}


async def main():
    nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-27B"
    adapter = NDIFChannels(model_name=model_name)
    results = {}
    for name, spec in SCENARIOS.items():
        prompts = [LobePrompt(lobe, text) for lobe, text in spec]
        spans, table = await adapter.reliance(prompts)
        results[name] = table
        print(f"\n=== {name} ===")
        header = "".join(f"{f'{a}-{b}':>12}" for a, b in spans)
        print(f"{'':>16}{header}")
        for (receiver, sender), moved in table.items():
            print(f"{receiver + ' <- ' + sender:>16}" +
                  "".join(f"{v:>12.0f}" for v in moved))

    print("\n=== Does it change? Per-pair, per-band, across scenarios ===")
    names = list(results)
    keys = list(results[names[0]])
    for key in keys:
        rows = [results[n][key] for n in names]
        spread = []
        for band in range(len(rows[0])):
            vals = [r[band] for r in rows]
            lo, hi = min(vals), max(vals)
            spread.append((hi - lo) / (hi + 1e-9) * 100)
        print(f"{key[0] + ' <- ' + key[1]:>16}  "
              f"variation across tasks: " +
              "".join(f"{s:>7.0f}%" for s in spread))


if __name__ == "__main__":
    asyncio.run(main())
