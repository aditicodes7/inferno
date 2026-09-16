"""R3: iteration-level scheduling.

R2's batch is fixed for its lifetime, so a request that finishes in 8 tokens
sits in the batch for all 128 steps producing nothing (measured: ~18% of all
decode steps wasted, worst request blocked 120 steps). This scheduler re-forms
the batch every iteration instead: finished requests leave immediately and give
back their slot, and waiting requests take it.

It owns state transitions and slot ownership, and nothing else. It never touches
a tensor - that separation is what lets the ugly cases (slot reuse, leaked
slots, starvation) be tested without a model.

POLICY, chosen deliberately and recorded in docs/decisions.md:

  FCFS.       The queue head is admitted first, always. Boring, and it is the
              only policy that provably cannot starve anyone - which makes it
              the right baseline to measure a cleverer policy against later.

  Prefill owns its iteration. A joining request needs hundreds of tokens
              through the model at once; running requests need one each. Those
              are different shapes. Admitting means every running request
              stalls for that iteration. That cost is real and is exactly why
              R3's metric is throughput at a fixed TTFT budget rather than raw
              throughput.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    id: str
    prompt_tokens: list[int]
    arrival: float = 0.0
    max_new_tokens: int = 128

    state: RequestState = RequestState.WAITING
    slot: int | None = None
    output: list[int] = field(default_factory=list)

    # timing, for the TTFT / latency measurements
    admitted_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None

    @property
    def n_prompt(self) -> int:
        return len(self.prompt_tokens)

    @property
    def n_generated(self) -> int:
        return len(self.output)


@dataclass
class Decision:
    """What the engine should do this iteration.

    For a prefill decision the slot is already RESERVED and recorded on the
    request: the engine has to know which slot to write K/V into before it can
    run the forward pass, so assigning it afterwards would be useless.
    """
    kind: str                                  # "prefill" | "decode" | "idle"
    prefill: Request | None = None
    decode: list[Request] = field(default_factory=list)


class Scheduler:
    def __init__(self, n_slots: int) -> None:
        self.n_slots = n_slots
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self._free: list[int] = list(range(n_slots))

    # -- queue ------------------------------------------------------------

    def add(self, request: Request) -> None:
        request.state = RequestState.WAITING
        self.waiting.append(request)

    def done(self) -> bool:
        return not self.waiting and not self.running

    # -- the decision -----------------------------------------------------

    def schedule(self, now: float) -> Decision:
        """Admit if possible, otherwise decode, otherwise idle.

        Admission is checked FIRST so a newly arrived request does not wait for
        the whole running batch to drain - that is the entire point of R3. It
        costs the running batch one stalled iteration, which is the trade R3 is
        built to measure.
        """
        if self._free and self.waiting:
            head = self.waiting[0]
            # FCFS: only the head is considered. Scanning past it for a request
            # that happens to fit is what starves the head forever.
            if head.arrival <= now:
                # Reserve the slot here, not in on_prefilled: the engine needs
                # it to write K/V. Idempotent, so calling schedule() twice
                # without acting does not consume two slots.
                if head.slot is None:
                    head.slot = self._free.pop(0)
                return Decision(kind="prefill", prefill=head)

        if self.running:
            return Decision(kind="decode", decode=list(self.running))
        return Decision(kind="idle")

    # -- transitions ------------------------------------------------------

    def on_prefilled(self, request: Request, token: int, now: float = 0.0,
                     eos_ids: set[int] | None = None) -> None:
        """The admitted request has been prefilled and produced its first token."""
        assert request.state is RequestState.WAITING, \
            f"{request.id} was prefilled from state {request.state}"
        assert request.slot is not None, \
            f"{request.id} was prefilled without a reserved slot"
        self.waiting.remove(request)
        request.state = RequestState.RUNNING
        request.admitted_at = now
        request.first_token_at = now
        request.output.append(token)
        self.running.append(request)
        if self._should_finish(request, token, eos_ids or set()):
            self.finish(request, now)

    def on_decoded(self, tokens: dict[int, int], eos_ids: set[int],
                   now: float = 0.0) -> None:
        """One decode step's results, keyed by SLOT rather than by request.

        Keyed by slot because that is what the engine actually has: a row index
        into the batched forward pass. Mapping it back to a request here keeps
        that translation in one place.
        """
        by_slot = {r.slot: r for r in self.running}
        for slot, token in tokens.items():
            request = by_slot.get(slot)
            if request is None:
                continue                       # finished earlier this iteration
            request.output.append(token)
            if self._should_finish(request, token, eos_ids):
                self.finish(request, now)

    @staticmethod
    def _should_finish(request: Request, token: int, eos_ids: set[int]) -> bool:
        return token in eos_ids or request.n_generated >= request.max_new_tokens

    def finish(self, request: Request, now: float = 0.0) -> None:
        """Release the slot. Only safe once this iteration's reads are done.

        The slot goes back to the FRONT of the free list, so a just-freed slot
        is the next one reused. That keeps the live set compact, which matters
        for the padded key length of the decode batch.
        """
        if request.state is RequestState.FINISHED:
            return
        if request in self.running:
            self.running.remove(request)
        elif request in self.waiting:
            # Reserved a slot but never prefilled (cancelled or shut down).
            # Without this branch the slot leaks silently.
            self.waiting.remove(request)
        if request.slot is not None:
            self._free.insert(0, request.slot)
            request.slot = None
        request.state = RequestState.FINISHED
        request.finished_at = now
        self.finished.append(request)

    # -- introspection, for the benchmark ---------------------------------

    def stats(self) -> dict:
        return {"waiting": len(self.waiting), "running": len(self.running),
                "finished": len(self.finished), "free_slots": len(self._free)}
