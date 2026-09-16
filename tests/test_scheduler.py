"""R3 scheduler logic. No model, no GPU - pure state machine.

Everything here is a bug that produces silent corruption or a hang rather than
an exception, which is why it gets tested in isolation from the engine:

  - a slot reused one iteration too early overwrites a live sequence's KV;
  - a finished request whose slot is never freed leaks capacity until the
    engine stalls with work still waiting;
  - a scheduling policy that prefers cheap requests starves expensive ones
    forever, and a starved request raises nothing at all.

None of those surface as a crash. They surface as wrong output or a hang.
"""

from __future__ import annotations

import pytest

from inferno.scheduler import Request, RequestState, Scheduler


def req(rid: str, n_prompt: int = 10, arrival: float = 0.0,
        max_new: int = 8) -> Request:
    return Request(id=rid, prompt_tokens=list(range(n_prompt)),
                   arrival=arrival, max_new_tokens=max_new)


# --------------------------------------------------------------------------
# admission
# --------------------------------------------------------------------------

def test_requests_start_waiting():
    s = Scheduler(n_slots=2)
    r = req("a")
    s.add(r)
    assert r.state is RequestState.WAITING
    assert r.slot is None


def test_admission_is_bounded_by_slots():
    s = Scheduler(n_slots=2)
    for i in range(5):
        s.add(req(f"r{i}"))
    admitted = []
    for _ in range(5):
        d = s.schedule(now=0.0)
        if d.kind != "prefill":
            break
        admitted.append(d.prefill)
        s.on_prefilled(d.prefill, token=1)
    assert len(admitted) == 2, "admitted more requests than there are slots"
    assert {r.slot for r in admitted} == {0, 1}


def test_unarrived_requests_are_not_admitted():
    s = Scheduler(n_slots=4)
    s.add(req("early", arrival=0.0))
    s.add(req("late", arrival=100.0))
    d = s.schedule(now=0.0)
    assert d.kind == "prefill" and d.prefill.id == "early"
    s.on_prefilled(d.prefill, token=1)
    d = s.schedule(now=0.0)
    assert d.kind == "decode", "a request that has not arrived yet was admitted"
    d = s.schedule(now=100.0)
    assert d.kind == "prefill" and d.prefill.id == "late"


# --------------------------------------------------------------------------
# slots - the silent-corruption cases
# --------------------------------------------------------------------------

def test_slot_is_freed_when_a_request_finishes_and_is_reused():
    s = Scheduler(n_slots=1)
    # max_new=2 so `a` is still RUNNING after prefill: the point of this test
    # is that a LIVE request keeps its slot, and only releases it on finish.
    a, b = req("a", max_new=2), req("b")
    s.add(a); s.add(b)

    d = s.schedule(now=0.0)
    assert d.prefill is a
    s.on_prefilled(a, token=99)
    assert a.state is RequestState.RUNNING and a.slot == 0
    assert s.schedule(now=0.0).kind == "decode", "b admitted while slot 0 is live"

    s.finish(a)
    assert a.state is RequestState.FINISHED
    assert a.slot is None, "finished request still holds its slot"

    d = s.schedule(now=0.0)
    assert d.kind == "prefill" and d.prefill is b
    assert d.prefill.slot == 0, "freed slot was not reused"


def test_a_joining_request_never_takes_a_live_slot():
    s = Scheduler(n_slots=3)
    live = []
    for i in range(2):
        s.add(req(f"live{i}"))
        d = s.schedule(now=0.0)
        s.on_prefilled(d.prefill, token=1)
        live.append(d.prefill)

    s.add(req("joiner"))
    d = s.schedule(now=0.0)
    assert d.kind == "prefill"
    assert d.prefill.slot not in {r.slot for r in live}, \
        "joining request was given a slot held by a running request"


def test_decode_batch_is_exactly_the_running_set():
    s = Scheduler(n_slots=4)
    for i in range(3):
        s.add(req(f"r{i}"))
        s.on_prefilled(s.schedule(now=0.0).prefill, token=1)
    d = s.schedule(now=0.0)
    assert d.kind == "decode"
    assert {r.id for r in d.decode} == {"r0", "r1", "r2"}
    assert all(r.state is RequestState.RUNNING for r in d.decode)


# --------------------------------------------------------------------------
# termination and starvation
# --------------------------------------------------------------------------

def test_request_finishes_at_max_new_tokens():
    s = Scheduler(n_slots=1)
    r = req("a", max_new=3)
    s.add(r)
    s.on_prefilled(s.schedule(now=0.0).prefill, token=1)
    for t in (2, 3):
        d = s.schedule(now=0.0)
        assert d.kind == "decode"
        s.on_decoded({r.slot: t}, eos_ids=set())
    assert r.state is RequestState.FINISHED
    assert len(r.output) == 3, f"expected 3 tokens, got {r.output}"


def test_request_finishes_on_eos():
    s = Scheduler(n_slots=1)
    r = req("a", max_new=100)
    s.add(r)
    s.on_prefilled(s.schedule(now=0.0).prefill, token=1)
    s.on_decoded({r.slot: 42}, eos_ids={42})
    assert r.state is RequestState.FINISHED
    assert r.output == [1, 42], "EOS token must be included in the output"


def test_fcfs_does_not_starve_an_expensive_request():
    """A long request queued first must not be overtaken indefinitely."""
    s = Scheduler(n_slots=1)
    s.add(req("long", n_prompt=500, max_new=50))
    for i in range(20):
        s.add(req(f"cheap{i}", n_prompt=5, max_new=1))
    d = s.schedule(now=0.0)
    assert d.prefill.id == "long", \
        "a later, cheaper request was admitted ahead of the queue head"


def test_scheduler_reports_idle_when_nothing_can_run():
    s = Scheduler(n_slots=1)
    assert s.schedule(now=0.0).kind == "idle"
    s.add(req("future", arrival=5.0))
    assert s.schedule(now=0.0).kind == "idle"
    assert not s.done()
    s.on_prefilled(s.schedule(now=5.0).prefill, token=1)
    s.finish(s.running[0])
    assert s.done()
