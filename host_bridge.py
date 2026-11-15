"""host_bridge.py — Auto-discovery and plug-and-play host integration.

On startup, inspects whatever host is connected and auto-registers action
handlers. No manual setup. Connect a host, call attach(), done.

Supported auto-discovery paths:
  1. MCP tools  — reads the MCP tool list, maps tool names → action handlers
  2. Playwright — detects a live browser page, maps DOM actions automatically
  3. Generic callable dict — any {action_name: async_fn} mapping

Usage:
    from host_bridge import attach

    # MCP host (Cursor, any MCP server)
    await attach(mcp_tools=mcp_tool_list)

    # Playwright browser
    await attach(playwright_page=page)

    # Both at once
    await attach(mcp_tools=tools, playwright_page=page)
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, Awaitable

from action_runtime import (
    ActionProposal, ActionResult, register_handler, _HANDLERS
)


# ---------------------------------------------------------------------------
# MCP auto-discovery
# ---------------------------------------------------------------------------

# Maps MCP tool name fragments → canonical action type
# Extend this list to cover new tool namespaces automatically.
_MCP_ACTION_MAP: list[tuple[re.Pattern, str]] = [
    # More specific patterns first — first match wins
    (re.compile(r"screenshot|snapshot|capture", re.I), "observe"),
    (re.compile(r"navigate|goto|open_url|load_url", re.I), "navigate"),
    (re.compile(r"scroll|swipe",             re.I), "scroll"),
    (re.compile(r"hover|mouse_over",         re.I), "hover"),
    (re.compile(r"keyboard|hotkey",          re.I), "key"),
    (re.compile(r"select|choose|pick",       re.I), "select"),
    (re.compile(r"fill|type_text|input",     re.I), "fill"),
    (re.compile(r"click|tap|press",          re.I), "click"),
    (re.compile(r"retrieve|search|query|lookup|find", re.I), "retrieve"),
    (re.compile(r"observe|inspect|read_page", re.I), "observe"),
    (re.compile(r"wait|sleep|pause",         re.I), "wait"),
]

def _infer_action_type(tool_name: str) -> str | None:
    for pattern, action_type in _MCP_ACTION_MAP:
        if pattern.search(tool_name):
            return action_type
    return None


def _build_mcp_handler(
    tool_name: str,
    call_tool: Callable[[str, dict], Awaitable[Any]],
    action_type: str,
) -> Callable[[ActionProposal], Awaitable[ActionResult]]:
    """Build an async handler that calls an MCP tool with the proposal's target/payload."""
    async def handler(p: ActionProposal) -> ActionResult:
        # Build the MCP tool arguments from the proposal
        args: dict[str, Any] = {}

        if action_type in ("click", "hover", "observe"):
            # target is a CSS selector or coordinate
            args["selector"] = p.target
        elif action_type == "fill":
            args["selector"] = p.target
            args["value"]    = str(p.payload or "")
        elif action_type in ("navigate", "goto"):
            args["url"] = p.target
        elif action_type in ("retrieve", "search", "query"):
            args["query"] = p.target
        elif action_type == "scroll":
            args["selector"] = p.target
            args["direction"] = str(p.payload or "down")
        elif action_type in ("wait", "sleep"):
            args["ms"] = int(float(p.payload or 500))
        elif action_type in ("key", "keyboard"):
            args["key"] = p.target
        else:
            args["target"] = p.target
            if p.payload is not None:
                args["value"] = p.payload

        try:
            result = await call_tool(tool_name, args)
            output = result if isinstance(result, str) else json.dumps(result)
            return ActionResult(
                lobe=p.lobe, action_type=action_type, target=p.target,
                success=True, output=output[:500],
            )
        except Exception as e:
            return ActionResult(
                lobe=p.lobe, action_type=action_type, target=p.target,
                success=False, error=str(e),
            )
    return handler


def register_mcp_tools(
    tools: list[dict],
    call_tool: Callable[[str, dict], Awaitable[Any]],
):
    """
    Auto-register handlers from an MCP tool list.

    tools:     list of tool dicts, each with at least {"name": str}
    call_tool: async (tool_name, args) -> result
    """
    registered = 0
    for tool in tools:
        name = tool.get("name", "")
        action_type = _infer_action_type(name)
        if not action_type:
            continue
        # Register — later tool for same action_type wins (more specific)
        handler = _build_mcp_handler(name, call_tool, action_type)
        _HANDLERS[action_type] = handler
        registered += 1

    print(f"[host-bridge] MCP: registered {registered}/{len(tools)} tools "
          f"→ {sorted(set(_HANDLERS.keys()))}", flush=True)


# ---------------------------------------------------------------------------
# Playwright auto-discovery
# ---------------------------------------------------------------------------

async def push_playwright_to_v(page, *, source: str = "browser") -> None:
    """Refresh V's live screen bus from the current Playwright page (no PNG)."""
    from live_screen import LIVE_SCREEN
    try:
        title = await page.title()
        url = page.url
        links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.slice(0,12).map(e => ({text:(e.innerText||'').trim().slice(0,60), href:e.href}))",
        )
        controls = await page.eval_on_selector_all(
            "input,textarea,button,select",
            "els => els.slice(0,12).map(e => ({tag:e.tagName.toLowerCase(), "
            "id:e.id||'', name:e.name||'', text:(e.innerText||e.value||'').trim().slice(0,40)}))",
        )
        LIVE_SCREEN.push(
            source=source,
            url=url or "",
            title=title or "",
            summary=f"{title} @ {url}",
            links=list(links or []),
            controls=list(controls or []),
        )
    except Exception:
        # Best-effort: still mark a minimal frame so workers can skip screenshots
        try:
            LIVE_SCREEN.push(
                source=source,
                url=getattr(page, "url", "") or "",
                title="",
                summary="page state refresh failed partially",
            )
        except Exception:
            pass


def register_playwright_page(page):
    """
    Auto-register handlers from a live Playwright page object.
    No imports at module level — Playwright is optional.

    After mutating actions, V's LiveScreenBus is refreshed so worker lanes
    already hold live state — observe/screenshot is skipped.
    """
    from live_screen import LIVE_SCREEN

    async def _after_mutate() -> None:
        await push_playwright_to_v(page)

    async def _click(p: ActionProposal) -> ActionResult:
        try:
            target = (p.target or "").strip()
            # Models often emit href URLs for click — not valid CSS; resolve to link.
            if re.match(r"^https?://", target, re.I):
                link = page.locator(f'a[href="{target}"]').first
                if await link.count() == 0:
                    # HN and others may use relative hrefs; try suffix match
                    link = page.locator(f'a[href*="{target.split("://", 1)[-1]}"]').first
                if await link.count() > 0:
                    await link.click(timeout=5000)
                else:
                    # Last resort: navigate to the URL (still advances the task)
                    await page.goto(target, timeout=15000)
                await _after_mutate()
                return ActionResult(
                    lobe=p.lobe, action_type="click", target=target, success=True,
                    output=f"clicked/opened {target} → {page.url}",
                )
            await page.click(target, timeout=5000)
            await _after_mutate()
            return ActionResult(lobe=p.lobe, action_type="click",
                                target=target, success=True,
                                output=f"clicked {target}")
        except Exception as e:
            return ActionResult(lobe=p.lobe, action_type="click",
                                target=p.target, success=False, error=str(e))

    async def _fill(p: ActionProposal) -> ActionResult:
        try:
            await page.fill(p.target, str(p.payload or ""), timeout=5000)
            await _after_mutate()
            return ActionResult(lobe=p.lobe, action_type="fill",
                                target=p.target, success=True,
                                output=f"filled {p.target}")
        except Exception as e:
            return ActionResult(lobe=p.lobe, action_type="fill",
                                target=p.target, success=False, error=str(e))

    async def _navigate(p: ActionProposal) -> ActionResult:
        try:
            await page.goto(p.target, timeout=15000)
            await _after_mutate()
            return ActionResult(lobe=p.lobe, action_type="navigate",
                                target=p.target, success=True,
                                output=page.url)
        except Exception as e:
            return ActionResult(lobe=p.lobe, action_type="navigate",
                                target=p.target, success=False, error=str(e))

    async def _observe(p: ActionProposal) -> ActionResult:
        # V already shared live state with worker lanes — no screenshot trip.
        if LIVE_SCREEN.should_skip_observe():
            LIVE_SCREEN.note_observe_skipped()
            return ActionResult(
                lobe=p.lobe, action_type="observe", target=p.target,
                success=True, output=LIVE_SCREEN.worker_context(),
            )
        try:
            await push_playwright_to_v(page)
            return ActionResult(
                lobe=p.lobe, action_type="observe", target=p.target,
                success=True, output=LIVE_SCREEN.worker_context(),
            )
        except Exception as e:
            return ActionResult(lobe=p.lobe, action_type="observe",
                                target=p.target, success=False, error=str(e))

    async def _scroll(p: ActionProposal) -> ActionResult:
        try:
            direction = str(p.payload or "down")
            delta = 300 if direction == "down" else -300
            await page.evaluate(f"window.scrollBy(0, {delta})")
            await _after_mutate()
            return ActionResult(lobe=p.lobe, action_type="scroll",
                                target=p.target, success=True,
                                output=f"scrolled {direction}")
        except Exception as e:
            return ActionResult(lobe=p.lobe, action_type="scroll",
                                target=p.target, success=False, error=str(e))

    async def _select(p: ActionProposal) -> ActionResult:
        try:
            await page.select_option(p.target, str(p.payload or ""), timeout=5000)
            return ActionResult(lobe=p.lobe, action_type="select",
                                target=p.target, success=True,
                                output=f"selected {p.payload}")
        except Exception as e:
            return ActionResult(lobe=p.lobe, action_type="select",
                                target=p.target, success=False, error=str(e))

    async def _hover(p: ActionProposal) -> ActionResult:
        try:
            await page.hover(p.target, timeout=5000)
            return ActionResult(lobe=p.lobe, action_type="hover",
                                target=p.target, success=True,
                                output=f"hovered {p.target}")
        except Exception as e:
            return ActionResult(lobe=p.lobe, action_type="hover",
                                target=p.target, success=False, error=str(e))

    async def _key(p: ActionProposal) -> ActionResult:
        try:
            await page.keyboard.press(p.target)
            return ActionResult(lobe=p.lobe, action_type="key",
                                target=p.target, success=True,
                                output=f"pressed {p.target}")
        except Exception as e:
            return ActionResult(lobe=p.lobe, action_type="key",
                                target=p.target, success=False, error=str(e))

    _HANDLERS.update({
        "click":    _click,
        "fill":     _fill,
        "navigate": _navigate,
        "observe":  _observe,
        "scroll":   _scroll,
        "select":   _select,
        "hover":    _hover,
        "key":      _key,
    })

    print(f"[host-bridge] Playwright: registered {len(_HANDLERS)} handlers", flush=True)


# ---------------------------------------------------------------------------
# Generic callable dict
# ---------------------------------------------------------------------------

def register_callables(mapping: dict[str, Callable[[ActionProposal], Awaitable[ActionResult]]]):
    """Register any {action_type: async_handler} dict directly."""
    _HANDLERS.update(mapping)
    print(f"[host-bridge] Generic: registered {sorted(mapping.keys())}", flush=True)


# ---------------------------------------------------------------------------
# attach() — single entry point, plug-and-play
# ---------------------------------------------------------------------------

async def attach(
    mcp_tools: list[dict] | None = None,
    call_tool: Callable[[str, dict], Awaitable[Any]] | None = None,
    playwright_page=None,
    callables: dict | None = None,
):
    """
    Auto-discover and register all available action handlers.

    Call once at startup. Accepts any combination of:
      mcp_tools + call_tool  — MCP server integration
      playwright_page        — Playwright browser integration
      callables              — custom {action_type: handler} dict

    After this returns, action_runtime.py has live handlers for every
    discovered action type. No further setup needed.
    """
    if mcp_tools and call_tool:
        register_mcp_tools(mcp_tools, call_tool)

    if playwright_page is not None:
        register_playwright_page(playwright_page)

    if callables:
        register_callables(callables)

    if not _HANDLERS:
        print("[host-bridge] Warning: no host attached — actions will no-op", flush=True)
    else:
        print(f"[host-bridge] Ready. Action types: {sorted(_HANDLERS.keys())}", flush=True)
