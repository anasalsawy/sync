"""Can several lanes fill one page at once? Measured, not assumed.

Four arms against the same 12-field form in one real Chromium page:

  sequential      one lane, real keystroke typing, field after field
  focus x2        two lanes, real keystroke typing, at the same time
  nodes x2        two lanes, focus-free node writes, at the same time
  nodes x4        four lanes, focus-free node writes, at the same time

The form carries both plain inputs and a controlled input that re-renders
its value from JS state, the way a React-controlled field does, so a write
that skips the native setter and the input event is visibly lost.
"""
import asyncio
import os
import time

from playwright.async_api import async_playwright

FIELDS = 12
VALUES = {f"f{index}": f"value-{index}" for index in range(FIELDS)}

PAGE = """
<form id="big"></form>
<script>
  const state = {};
  const form = document.getElementById('big');
  for (let index = 0; index < %d; index++) {
    const input = document.createElement('input');
    input.id = 'f' + index;
    // Controlled: the field's value is re-rendered from state on every input,
    // so a bare `el.value = x` with no input event is discarded.
    input.addEventListener('input', event => {
      state[input.id] = event.target.value;
      input.value = state[input.id];
    });
    form.appendChild(input);
  }
  window.__state = state;
  window.__calls = 0;
</script>
""" % FIELDS

# Sets the value through the native setter, then fires the input event the
# page's own handler listens for. No focus is taken, so lanes do not collide.
NODE_WRITE = """
([ids, values]) => {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype, 'value').set;
  ids.forEach((id, slot) => {
    const element = document.getElementById(id);
    setter.call(element, values[slot]);
    element.dispatchEvent(new Event('input', {bubbles: true}));
  });
  window.__calls += 1;
  return ids.length;
}
"""


async def fresh(browser):
    page = await browser.new_page(viewport={"width": 900, "height": 700})
    await page.set_content(PAGE)
    return page


async def read_back(page):
    return await page.evaluate(
        "() => Object.fromEntries(Array.from(document.querySelectorAll('input'))"
        ".map(element => [element.id, element.value]))"
    )


def lanes_of(count):
    ids = list(VALUES)
    size = len(ids) // count
    return [ids[index * size:(index + 1) * size] for index in range(count)]


async def sequential(page):
    for field, value in VALUES.items():
        await page.fill(f"#{field}", value)
    return FIELDS


async def focus_typing(page, count):
    async def lane(ids):
        for field in ids:
            await page.fill(f"#{field}", VALUES[field])

    await asyncio.gather(*(lane(ids) for ids in lanes_of(count)))
    return FIELDS


async def node_writes(page, count):
    async def lane(ids):
        await page.evaluate(NODE_WRITE, [ids, [VALUES[field] for field in ids]])

    await asyncio.gather(*(lane(ids) for ids in lanes_of(count)))
    return await page.evaluate("() => window.__calls")


async def main():
    arms = [
        ("sequential", lambda page: sequential(page)),
        ("focus x2", lambda page: focus_typing(page, 2)),
        ("nodes x2", lambda page: node_writes(page, 2)),
        ("nodes x4", lambda page: node_writes(page, 4)),
    ]
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=os.getenv("SYNC_CHROMIUM", "/usr/bin/chromium")
        )
        try:
            print(f"{'arm':<12} {'wall ms':>8} {'calls':>6}  correct  note")
            for name, arm in arms:
                page = await fresh(browser)
                start = time.perf_counter()
                calls = await arm(page)
                elapsed = (time.perf_counter() - start) * 1000
                final = await read_back(page)
                wrong = {field: final[field] for field in VALUES
                         if final[field] != VALUES[field]}
                note = "all fields landed" if not wrong else f"{len(wrong)} lost/wrong"
                print(f"{name:<12} {elapsed:>8.1f} {calls:>6}  "
                      f"{len(VALUES) - len(wrong):>3}/{len(VALUES)}   {note}")
                if wrong:
                    print(f"             damage: {dict(list(wrong.items())[:4])}")
                await page.close()
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
