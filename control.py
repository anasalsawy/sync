"""sawii/sync scheduling and execution permissions."""
import asyncio
from collections.abc import Awaitable, Callable, Iterable

MAIN = ("L", "V", "R", "B", "C", "Embedder")

# Only L and R lead lanes of their own. An offspring never spawns offspring.
SPAWNERS = ("L", "R")


def leader(lobe: str) -> str:
    return lobe.rstrip("0123456789")


def can_share(source: str, target: str) -> bool:
    """Every lobe shares with every other lobe.

    There are no family boundaries on sharing. A main lobe, an L offspring and
    an R offspring all exchange live signals directly: one mind, not a relay
    tree. ``leader`` still governs parentage and who may act, never who may
    share.
    """
    return source != target


class SyncControl:
    """Schedules a supplied inference adapter. The adapter owns model state."""

    def __init__(self, step: Callable[[str], Awaitable[None]]) -> None:
        self.step = step
        self.tasks: dict[str, asyncio.Task] = {}
        self.spawn_count = {"L": 0, "R": 0}

    def start(self) -> tuple[asyncio.Task, ...]:
        return self.continue_round("L", MAIN)

    def continue_round(self, actor: str, lobes: Iterable[str]) -> tuple[asyncio.Task, ...]:
        if actor != "L":
            raise PermissionError("Only L controls the main schedule")
        names = tuple(dict.fromkeys(lobes))
        if any(name not in MAIN and name not in self.tasks for name in names):
            raise ValueError("Unknown lobe")
        return tuple(self._activate(name) for name in names)

    def request_parent(self, actor: str) -> asyncio.Task:
        if self.tasks.get(actor) is not asyncio.current_task():
            raise PermissionError("Only a running lobe can request its parent")
        if actor == "L":
            raise ValueError("L has no parent")
        parent = "L" if actor in MAIN else leader(actor)
        return self._activate(parent)

    def _activate(self, name: str) -> asyncio.Task:
        task = self._run(name)
        for child in tuple(self.tasks):
            if child != name and (name == "L" or leader(child) == name):
                self._run(child)
        return task

    def spawn(self, actor: str) -> str:
        if actor not in self.spawn_count:
            raise PermissionError("Only L and R spawn their own offspring")
        self.spawn_count[actor] += 1
        name = f"{actor}{self.spawn_count[actor]}"
        self._run(name)
        return name

    async def interrupt(self, actor: str, lobe: str) -> None:
        if actor != "L":
            raise PermissionError("Only L interrupts the schedule")
        names = [name for name in self.tasks if name == lobe or leader(name) == lobe]
        tasks = [self.tasks[name] for name in names if not self.tasks[name].done()]
        if asyncio.current_task() in tasks:
            raise ValueError("A running lobe must stop itself by returning")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def dispatch(self, actor: str, action: Callable[[], Awaitable[object]]) -> object:
        if actor == "R" or leader(actor) != "R" or actor not in self.tasks:
            raise PermissionError("Only R offspring can execute an action")
        if self.tasks[actor] is not asyncio.current_task():
            raise PermissionError("The active offspring must dispatch its own action")
        return await action()

    def _run(self, lobe: str) -> asyncio.Task:
        task = self.tasks.get(lobe)
        if task is None or task.done():
            task = asyncio.create_task(self.step(lobe), name=lobe)
            self.tasks[lobe] = task
        return task
