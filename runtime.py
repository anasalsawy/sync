"""sawii/sync: one persistent shared mind.

The mind is singular. L and R are one each for the whole life of a session,
and a new prompt is a stimulus injected into the live shared state, never a
fresh run with a fresh L. The backend supplies coupled passes over that one
state; this module is the referee, not the mind.

Nothing here schedules the lobes. Each pass the lobes report what they want
of the next one, and the shape of that pass is whatever they asked for.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from uuid import uuid4
from dataclasses import dataclass, field
from typing import Protocol

from clock import Clock, Event
from control import MAIN, SPAWNERS, can_share, leader


@dataclass(frozen=True)
class Observation:
    """What a host reported back, in the host's own vocabulary.

    ``payload`` is whatever the host says a percept is. The mind reads it;
    this module never interprets it. ``revisions`` gives the current revision
    of each thing a lane might claim, so staleness is per claim rather than
    per surface: one lane filling a field does not invalidate another lane's
    pending act on a different field.
    """

    payload: object = None
    revisions: tuple[tuple[str, str], ...] = ()
    result: str = ""

    def revision(self, claim: str) -> str | None:
        return dict(self.revisions).get(claim)


@dataclass(frozen=True)
class Action:
    """What a lane wants done, in the host's own vocabulary.

    ``payload`` is the host's own action and is opaque here: the mind decides
    which lane pursues which goal, never what an action is. ``claims`` names
    the things in the host's world this act touches, and ``seen`` is the
    revision the lane last had for each of them. ``exclusive`` marks an act
    that needs the host's single focus, a resource with one holder at a
    time, which is taken in turn rather than fought over.
    """

    payload: object
    claims: frozenset[str] = frozenset()
    seen: tuple[tuple[str, str], ...] = ()
    exclusive: bool = False


@dataclass(frozen=True)
class Exchange:
    """One live hidden-signal patch that happened inside the completed pass.

    ``site`` names where in the stack the signal crossed (``"layers.14"``), and
    ``magnitude`` is the measured effect of that patch on the recipient. A
    patch that moved nothing is not an exchange.
    """

    source: str
    target: str
    site: str
    magnitude: float


@dataclass(frozen=True)
class Subtask:
    key: str
    goal: str


@dataclass(frozen=True)
class BranchAction:
    subtask: str
    actor: str
    action: Action
    veto: bool = False


@dataclass(frozen=True)
class Stimulus:
    """Something that arrived from outside and is still unanswered."""

    key: str
    text: str
    origin: str = "L"


@dataclass(frozen=True)
class Demand:
    """What one lobe wants of the next pass.

    This is the whole scheduling vocabulary. No lobe declares a plan, a pass
    count or a width: it says it wants to keep thinking, that it wants more
    lanes of itself, or whose attention it wants, and the next pass is the sum
    of what everyone asked for.
    """

    lobe: str
    another_pass: bool = False
    split: int = 0
    attention: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reply:
    """L speaking, addressed to the stimulus it answers."""

    stimulus: str
    text: str


@dataclass(frozen=True)
class Decision:
    """Internal readouts from the lobes for one completed coupled pass."""

    demands: tuple[Demand, ...] = ()
    exchanges: tuple[Exchange, ...] = ()
    replies: tuple[Reply, ...] = ()
    split: tuple[Subtask, ...] = ()
    assignments: tuple[tuple[str, str], ...] = ()
    branch_actions: tuple[BranchAction, ...] = ()
    complete_branches: tuple[str, ...] = ()


@dataclass
class Mind:
    """The one shared state every lobe reads and writes."""

    observation: Observation = field(default_factory=Observation)
    history: tuple[tuple[str, str], ...] = ()
    stimuli: dict[str, Stimulus] = field(default_factory=dict)
    active: tuple[str, ...] = ()
    passes: int = 0
    events: list[str] = field(default_factory=list)
    offspring: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {name: () for name in SPAWNERS}
    )
    subtasks: dict[str, Subtask] = field(default_factory=dict)
    assignments: dict[str, str] = field(default_factory=dict)
    branch_observations: dict[str, Observation] = field(default_factory=dict)
    branch_results: dict[str, Observation] = field(default_factory=dict)
    in_flight: frozenset[str] = frozenset()
    claimed: dict[str, str] = field(default_factory=dict)
    # Time the mind lives inside. The clock keeps ticking whether or not a
    # lobe is thinking: ``now`` is refreshed continuously, ``timeline`` is
    # everything the world did as it happened, and ``seen_until`` marks how far
    # into that timeline the lobes have already been shown.
    now: float = 0.0
    born: float = 0.0
    elapsed: float = 0.0
    timeline: list[Event] = field(default_factory=list)
    seen_until: float = 0.0
    fresh: tuple[Event, ...] = ()

    def since_last_look(self) -> tuple[Event, ...]:
        """What happened between the previous pass and this one."""
        return self.fresh

    def lanes(self) -> tuple[str, ...]:
        return tuple(name for family in SPAWNERS for name in self.offspring[family])

    def knows(self, lobe: str) -> bool:
        return lobe in MAIN or lobe in self.lanes()


class Inference(Protocol):
    async def coupled_pass(
        self, mind: Mind, links: dict[str, tuple[str, ...]]
    ) -> Decision:
        """Think one pass over the one shared mind.

        Exchange live signals directly on the given links and report every
        exchange. Generated text never substitutes for a hidden signal.
        """

    async def close(self, mind: Mind) -> None:
        """Release backend state for this mind, including retired lanes."""


class Worker(Protocol):
    """One surface of a connected host."""

    async def observe(self) -> Observation: ...

    async def execute(self, action: Action) -> Observation | None:
        """Atomically reject a stale claim with None, or act and observe."""


class Host(Protocol):
    """The connected host owns its native worker and instance lifecycles."""

    @property
    def supports_subagents(self) -> bool: ...

    def open_subagent(self, subtask: Subtask) -> AbstractAsyncContextManager[Worker]: ...

    def open_instance(self, subtask: Subtask) -> AbstractAsyncContextManager[Worker]: ...

    # A host may hand back a surface it has already given out. Lanes then
    # share it, which is how several lanes work one page at once.


class Sync:
    """Keeps one mind alive and holds every pass to the rules."""

    def __init__(self, inference: Inference, worker: Worker | None = None, *,
                 host: Host | None = None, clock: Clock | None = None,
                 ambient=None, poll_seconds: float = 0.05) -> None:
        self.inference = inference
        self.worker = worker
        self.clock = clock or Clock()
        # The world keeps moving while the mind sleeps. ``ambient`` is an
        # async callable the host supplies, returning what changed since the
        # last call, as short descriptions. Nothing is scheduled by the mind.
        self.ambient = ambient
        self.poll_seconds = poll_seconds
        self._watch: asyncio.Task[None] | None = None
        self.host = host
        self.mind: Mind | None = None
        self.event_path = os.getenv("SYNC_EVENT_LOG")
        self._pump: asyncio.Task[None] | None = None
        self._answers: dict[str, asyncio.Future[str]] = {}
        self._workers: dict[str, Worker] = {}
        self._surfaces: dict[int, dict[str, object]] = {}
        self._held: dict[str, frozenset[str]] = {}
        self._woken: tuple[str, ...] = ()
        self._demands: tuple[Demand, ...] = ()
        self.failure: BaseException | None = None
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._nudge = asyncio.Event()

    # ---------------------------------------------------------------- events

    def _emit(self, event: str, **details: object) -> None:
        if not self.event_path:
            return
        entry = {"time": time.time(), "run": "mind", "event": event, **details}
        try:
            with Path(self.event_path).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logging.getLogger(__name__).exception("Cannot write Sync monitor event")

    # ------------------------------------------------------------- stimulus

    def send(self, text: str, *, origin: str = "L") -> asyncio.Future[str]:
        """Inject a stimulus into the live mind and wait for L to answer it.

        A second prompt does not start a second mind. It lands in the same
        shared state as everything already in flight, and the lobes see it on
        their next pass.
        """
        if origin not in MAIN:
            raise ValueError("Unknown initiating lobe")
        if self.mind is None:
            self.mind = Mind(now=self.clock.now(), born=self.clock.now())
        key = uuid4().hex[:8]
        self.mind.stimuli[key] = Stimulus(key, text, origin)
        if origin == "L":
            self.mind.history += (("user", text),)
        answer: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._answers[key] = answer
        self._emit("stimulus", key=key, origin=origin, text=text)
        self._nudge.set()
        self._start_pump()
        return answer

    def _start_pump(self) -> None:
        if self._watch is None:
            self._watch = asyncio.create_task(self._watching(), name="world")
        if self._pump is None or self._pump.done():
            self._pump = asyncio.create_task(self._live(), name="mind")
            # Mark any failure retrieved here; it is kept on self.failure and
            # still raised to anyone who awaits the pump.
            self._pump.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )

    async def interrupt(self) -> None:
        pump, self._pump = self._pump, None
        if pump is not None and not pump.done():
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
        for key, answer in self._answers.items():
            if not answer.done():
                answer.cancel()
        self._answers.clear()

    async def aclose(self) -> None:
        if self._watch is not None:
            self._watch.cancel()
            await asyncio.gather(self._watch, return_exceptions=True)
            self._watch = None
        await self.interrupt()
        for task in self._inflight.values():
            task.cancel()
        await asyncio.gather(*self._inflight.values(), return_exceptions=True)
        self._inflight.clear()
        for key in tuple(self._workers):
            await self._retire_worker(key)
        if self.mind is not None:
            await self.inference.close(self.mind)

    # ------------------------------------------------------------- the pump

    async def _watching(self) -> None:
        """The world ticks on, always, between every pass.

        The clock and the outside world are never paused: each poll refreshes
        ``now`` and appends whatever happened to the timeline, time-stamped.
        A lobe that is not thinking at that moment still stands inside a
        present that is moving.
        """
        assert self.mind is not None
        while True:
            now = self.clock.now()
            self.mind.now = now
            self.mind.elapsed = now - self.mind.born
            changes = await self.ambient() if self.ambient is not None else ()
            for item in changes:
                # The world may stamp its own events; a bare string is seen now.
                if isinstance(item, Event):
                    self.mind.timeline.append(item)
                else:
                    self.mind.timeline.append(Event(now, str(item)))
            await asyncio.sleep(self.poll_seconds)

    async def _live(self) -> None:
        mind = self.mind
        assert mind is not None
        await self._pump_loop(mind)

    async def _pump_loop(self, mind: Mind) -> None:
        if self.worker is not None and not mind.observation.revisions:
            mind.observation = await self.worker.observe()
        try:
            while True:
                self._reap()
                mind.in_flight = frozenset(
                    key for key, task in self._inflight.items() if not task.done()
                )
                mind.claimed = {name: holder
                                for holder, names in self._held.items()
                                for name in names}
                mind.active = self._wake(mind)
                if not mind.active:
                    return
                links = {
                    lobe: tuple(peer for peer in mind.active
                                if peer != lobe and can_share(peer, lobe))
                    for lobe in mind.active
                }
                mind.now = self.clock.now()
                mind.elapsed = mind.now - mind.born
                mind.fresh = tuple(event for event in mind.timeline
                                   if mind.seen_until < event.at <= mind.now)
                mind.seen_until = mind.now
                self._emit("pass started", number=mind.passes + 1, lobes=mind.active,
                           elapsed=round(mind.elapsed, 3))
                result = await self.inference.coupled_pass(mind, links)
                mind.passes += 1
                self._emit("pass finished", number=mind.passes)
                self._audit_exchanges(mind, result, links)
                self._audit_roles(mind, result)
                self._woken = self._spawn(mind, result)
                await self._route_branches(mind, result)
                self._speak(mind, result)
                self._settle(mind)
                self._demands = result.demands
                await asyncio.sleep(0)
                if not result.demands:
                    await self._idle()
        except asyncio.CancelledError:
            self._emit("mind interrupted")
            raise
        except Exception as exc:
            # A broken rule must never vanish just because every prompt was
            # already answered: it is raised to anyone waiting, kept on the
            # Sync, and raised again by the pump itself.
            self._emit("mind failed", error=type(exc).__name__)
            self.failure = exc
            for answer in self._answers.values():
                if not answer.done():
                    answer.set_exception(exc)
            self._answers.clear()
            raise

    def _wake(self, mind: Mind) -> tuple[str, ...]:
        """Who thinks on the next pass. Nobody wrote this list; it is a sum."""
        wanted: set[str] = set()
        for item in mind.stimuli.values():
            if item.origin == "L":
                # L owes an answer, so L stays up, and the mind with it.
                wanted.update(MAIN)
            else:
                # A lobe-initiated stimulus proceeds without waking L.
                wanted.add(item.origin)
        for demand in self._demands:
            if demand.another_pass:
                wanted.add(demand.lobe)
            wanted.update(demand.attention)
        # A lane holding unfinished work is never put down.
        wanted.update(actor for key, actor in mind.assignments.items()
                      if key not in mind.branch_results)
        # Activation flows downward: a lane just spawned thinks next pass.
        wanted.update(self._woken)
        self._woken = ()
        order = MAIN + mind.lanes()
        return tuple(name for name in order if name in wanted)

    # ---------------------------------------------------------------- audits

    def _audit_exchanges(self, mind: Mind, result: Decision,
                         links: dict[str, tuple[str, ...]]) -> None:
        """Hold the pass to the one-shared-mind rule.

        A lobe that generated text instead of sharing signals is the failure
        this guards against, so every active lobe must appear in at least one
        reported exchange. The runtime does not run the model and so cannot
        observe the patch itself: what it enforces is that the backend account
        for each lobe per pair, per site, with a nonzero effect, which makes
        silence unpassable and a fabricated claim a visible entry in the log.
        """
        for shared in result.exchanges:
            if shared.source not in mind.active or shared.target not in mind.active:
                raise PermissionError("An exchange named a lobe that is not active")
            if shared.target not in links[shared.source]:
                raise PermissionError("An exchange crossed a link that is not open")
            if not shared.site:
                raise ValueError("An exchange must name its site in the stack")
            if not shared.magnitude > 0:
                raise ValueError("An exchange that moved nothing is not an exchange")
        if len(mind.active) < 2:
            return
        shared_lobes = {end for item in result.exchanges
                        for end in (item.source, item.target)}
        silent = tuple(lobe for lobe in mind.active if lobe not in shared_lobes)
        if silent:
            raise PermissionError(
                "Lobes did not share live signals this pass: " + ", ".join(silent)
            )
        self._emit("signals exchanged", count=len(result.exchanges),
                   sites=sorted({item.site for item in result.exchanges}))

    def _audit_roles(self, mind: Mind, result: Decision) -> None:
        if result.replies and "L" not in mind.active:
            raise PermissionError("Only L speaks, and L must be active to speak")
        if (result.split or result.assignments) and "R" not in mind.active:
            raise PermissionError("R must be active to plan or delegate execution")
        for reply in result.replies:
            if reply.stimulus not in mind.stimuli:
                raise ValueError("L answered a stimulus that is not open")
        for demand in result.demands:
            if demand.lobe not in mind.active:
                raise PermissionError("Only an active lobe states what it wants")
            if demand.split < 0:
                raise ValueError("A lobe cannot want a negative number of lanes")
            if demand.split and demand.lobe not in SPAWNERS:
                raise PermissionError("Only L and R open lanes of their own")
            for name in demand.attention:
                if not mind.knows(name):
                    raise ValueError("A lobe asked for the attention of nobody")

    # ------------------------------------------------------------- mechanics

    def _spawn(self, mind: Mind, result: Decision) -> tuple[str, ...]:
        """Lanes are elastic: a lobe asks for more and gets them. No cap."""
        opened: list[str] = []
        for demand in result.demands:
            if not demand.split:
                continue
            family = demand.lobe
            existing = mind.offspring[family]
            born = tuple(f"{family}{len(existing) + index + 1}"
                         for index in range(demand.split))
            mind.offspring[family] = existing + born
            opened.extend(born)
            self._emit("lanes opened", family=family, lanes=born)
        return tuple(opened)

    def _speak(self, mind: Mind, result: Decision) -> None:
        for reply in result.replies:
            mind.stimuli.pop(reply.stimulus, None)
            mind.history += (("assistant", reply.text),)
            answer = self._answers.pop(reply.stimulus, None)
            if answer is not None and not answer.done():
                answer.set_result(reply.text)
            self._emit("reply", stimulus=reply.stimulus, text=reply.text)

    def _reap(self) -> None:
        """Collect hands that have landed. A host that broke is not swallowed."""
        for key, task in tuple(self._inflight.items()):
            if task.done():
                del self._inflight[key]
                task.result()

    async def _idle(self) -> None:
        """Nobody wants to think and hands are still moving: wait for one.

        This is the only place the mind waits, and a new prompt breaks it
        immediately, so the conversation is never held up by execution.
        """
        moving = [task for task in self._inflight.values() if not task.done()]
        if not moving:
            return
        self._nudge.clear()
        waiting = asyncio.ensure_future(self._nudge.wait())
        try:
            await asyncio.wait([*moving, waiting],
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiting.cancel()

    def _settle(self, mind: Mind) -> None:
        """Retire stimuli nobody owes a spoken answer to.

        Only L speaks, so a stimulus raised by another lobe is an internal
        event: it is delivered on one pass and then done. Leaving it open
        would keep its lobe awake for a reply that can never come.
        """
        for key, item in tuple(mind.stimuli.items()):
            if item.origin != "L" and item.origin in mind.active:
                mind.stimuli.pop(key)
                answer = self._answers.pop(key, None)
                if answer is not None and not answer.done():
                    answer.set_result("")

    async def _route_branches(self, mind: Mind, result: Decision) -> None:
        if result.split:
            keys = [subtask.key for subtask in result.split]
            if len(set(keys)) != len(keys) or set(keys) & mind.subtasks.keys():
                raise ValueError("Each subtask needs a key of its own")
            async with asyncio.TaskGroup() as group:
                observations = {
                    item.key: group.create_task(self._open_worker(item), name="mind")
                    for item in result.split
                }
            mind.subtasks.update((item.key, item) for item in result.split)
            self._emit("subtasks created",
                       subtasks={item.key: item.goal for item in result.split})
            mind.branch_observations.update(
                (key, job.result()) for key, job in observations.items()
            )

        assignments = dict(mind.assignments)
        busy = {actor for key, actor in assignments.items()
                if key not in mind.branch_results}
        for key, actor in result.assignments:
            if key not in mind.subtasks or key in assignments:
                raise ValueError("R can only assign an unassigned subtask")
            if actor in busy or actor not in mind.active or leader(actor) != "R" or actor == "R":
                raise PermissionError("R must assign an available active lane")
            assignments[key] = actor
            busy.add(actor)

        actions = {proposal.subtask: proposal for proposal in result.branch_actions}
        completions = set(result.complete_branches)
        if len(actions) != len(result.branch_actions) or actions.keys() & completions:
            raise ValueError("A separate pass must observe an action and merge its result")
        taken = dict(mind.claimed)
        for key, proposal in actions.items():
            for name in proposal.action.claims:
                holder = taken.get(name)
                if holder is not None and holder != key:
                    raise PermissionError(
                        f"Two lanes claimed {name}: {holder} and {key}"
                    )
                taken[name] = key
        for key in actions.keys() | completions:
            if key in mind.in_flight:
                raise PermissionError("That lane's hand is still moving")
            actor = assignments.get(key)
            if key in mind.branch_results or actor not in mind.active:
                raise PermissionError("Only an assigned active lane works an open subtask")
            proposal = actions.get(key)
            if proposal is not None and proposal.actor != actor:
                raise PermissionError("An action must come from the lane that holds it")
            if proposal is not None and proposal.veto and "C" not in mind.active:
                raise PermissionError("Only active C vetoes an action")
        mind.assignments = assignments
        for key, actor in result.assignments:
            self._emit("subtask assigned", subtask=key, lane=actor)

        for key, proposal in actions.items():
            self._held[key] = proposal.action.claims
            self._inflight[key] = asyncio.create_task(
                self._act(mind, proposal), name="mind"
            )
        for key in completions:
            mind.branch_results[key] = mind.branch_observations[key]
            self._emit("subtask finished", subtask=key, lane=assignments[key])
            await self._retire_worker(key)

    async def _open_worker(self, item: Subtask) -> Observation:
        """Ask the host for a surface for this subtask.

        A host is free to hand back a surface it has already given out. That
        is how several lanes come to work one surface at once: they share the
        surface and stay out of each other's way by claiming different things
        on it. The surface closes when its last lane lets go.
        """
        if item.key not in self._workers:
            host = self.host
            if host is None:
                raise RuntimeError("No host is connected for this subtask")
            native = host.supports_subagents
            open_worker = host.open_subagent if native else host.open_instance
            stack = AsyncExitStack()
            worker = await stack.enter_async_context(open_worker(item))
            handle = id(worker)
            surface = self._surfaces.get(handle)
            if surface is None:
                self._surfaces[handle] = {
                    "stack": stack, "lanes": {item.key}, "turn": asyncio.Lock(),
                }
                self._emit("host surface opened", subtask=item.key,
                           mode="native subagent" if native else "host instance")
            else:
                await stack.aclose()
                surface["lanes"].add(item.key)
                self._emit("host surface shared", subtask=item.key,
                           lanes=sorted(surface["lanes"]))
            self._workers[item.key] = worker
        return await self._workers[item.key].observe()

    async def _retire_worker(self, key: str) -> None:
        worker = self._workers.pop(key, None)
        self._held.pop(key, None)
        if worker is None:
            return
        surface = self._surfaces.get(id(worker))
        if surface is None:
            return
        surface["lanes"].discard(key)
        if not surface["lanes"]:
            del self._surfaces[id(worker)]
            await surface["stack"].aclose()

    async def _act(self, mind: Mind, proposal: BranchAction) -> None:
        key = proposal.subtask
        if proposal.veto:
            self._emit("action vetoed", subtask=key, lane=proposal.actor)
            mind.events.append(f"C vetoed the proposed action for {key}")
            return
        worker = self._workers[key]
        action = proposal.action
        surface = self._surfaces[id(worker)]
        known = mind.branch_observations[key]
        fresh = all(known.revision(name) == was for name, was in action.seen)
        observed = None
        try:
            if fresh:
                self._emit("action started", subtask=key, lane=proposal.actor,
                           claims=sorted(action.claims), exclusive=action.exclusive)
                if action.exclusive:
                    # One holder at a time. Lanes take turns
                    # rather than fight; the wait is one act long.
                    async with surface["turn"]:
                        observed = await worker.execute(action)
                else:
                    observed = await worker.execute(action)
            if observed is None:
                mind.events.append(f"A claim moved under {key} before it could act")
                observed = await worker.observe()
            mind.branch_observations[key] = observed
            self._emit("observation updated", subtask=key, lane=proposal.actor)
        finally:
            self._held.pop(key, None)
