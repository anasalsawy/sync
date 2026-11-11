"""One scenario's reliance table, printed as JSON so runs can be combined."""
import asyncio, json, os, sys
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
    name = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.8-27B"
    adapter = NDIFChannels(model_name=model_name)
    prompts = [LobePrompt(lobe, text) for lobe, text in SCENARIOS[name]]
    spans, table = await adapter.reliance(prompts)
    print("RESULT " + json.dumps({
        "scenario": name, "spans": spans,
        "table": {f"{r}<-{s}": v for (r, s), v in table.items()},
    }))


if __name__ == "__main__":
    asyncio.run(main())
