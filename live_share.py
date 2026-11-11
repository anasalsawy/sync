"""Live: continuous peer channels on Qwen3.8-27B, and the routing that emerged."""
import asyncio, os, sys
import nnsight
from ndif import LobePrompt, NDIFChannels, SLOTS_PER_PEER


async def main():
    nnsight.CONFIG.API.APIKEY = os.environ["NDIF_API_KEY"]
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-27B"
    goal = "Book the Zurich meeting room for Thursday."
    prompts = [
        LobePrompt("L", f"Conversational voice. The user asks: {goal}"),
        LobePrompt("R", f"Execution supervisor. Carry out: {goal}"),
        LobePrompt("Embedder",
                   "Memory: Anas prefers mornings. Room 4 was double-booked last week."),
    ]
    adapter = NDIFChannels(model_name=model_name)
    print(f"model: {model_name}, {len(adapter.layers)} layers, "
          f"{SLOTS_PER_PEER} slots per peer, channels open on every layer")

    decision, _ = await adapter.think(prompts)
    print(f"\n{len(decision.exchanges)} channels carried signal "
          f"({decision.exchanges[0].site}):")
    for ex in decision.exchanges:
        print(f"  {ex.source:>8} -> {ex.target:<8}  crossing {ex.magnitude:12.0f}")

    print("\nEmergent routing: reliance on each channel by depth")
    print("(output shift when that one channel is silenced there)")
    spans, table = await adapter.reliance(prompts)
    header = "".join(f"{f'{a}-{b}':>12}" for a, b in spans)
    print(f"{'':>16}{header}")
    for (receiver, sender), moved in table.items():
        label = f"{receiver} <- {sender}"
        print(f"{label:>16}" + "".join(f"{v:>12.0f}" for v in moved))


if __name__ == "__main__":
    asyncio.run(main())
