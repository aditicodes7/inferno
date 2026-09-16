"""R3 acceptance: output must not depend on WHEN a request arrived.

Same two-tier criterion as R2 (see docs/decisions.md): float32 asserts exact
equality, float16 asserts only that nothing degenerates.

This is a strictly harder assertion than R2's. There, a sequence shared its
batch with a fixed set of neighbours for its whole life. Here its neighbours
change every iteration, it may be admitted into a slot another request just
vacated, and its own decode position is unrelated to anyone else's.
"""

from __future__ import annotations

import json
import os

import pytest

from tests.test_parity import ROOT

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "mps"
MAX_NEW = 32

MIXED = ["short-00", "medium-00", "long-00", "short-01",
         "medium-01", "long-01", "short-02", "short-03"]
N = 8 if os.environ.get("INFERNO_FULL") else 5


@pytest.fixture(scope="session")
def prompt_texts():
    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    by_id = {p["id"]: p for p in spec["prompts"]}
    return [by_id[i]["text"] for i in MIXED[:N]]


@pytest.fixture(scope="session")
def eng32():
    from inferno.continuous_engine import ContinuousEngine
    return ContinuousEngine(MODEL, device=DEVICE, dtype="float32", attn="eager")


@pytest.fixture(scope="session")
def alone32(eng32, prompt_texts):
    """Ground truth: each prompt served entirely on its own."""
    return [eng32.generate(t, max_new_tokens=MAX_NEW) for t in prompt_texts]


ARRIVALS = {
    "all_at_once": lambda n: [0.0] * n,
    "staggered": lambda n: [i * 0.25 for i in range(n)],
    "late_burst": lambda n: [0.0] + [2.0] * (n - 1),
}


@pytest.mark.parametrize("pattern", list(ARRIVALS))
@pytest.mark.parametrize("n_slots", [1, 2, 4])
def test_output_is_independent_of_arrival_time(eng32, prompt_texts, alone32,
                                               pattern, n_slots):
    reqs = eng32.make_requests(prompt_texts, arrivals=ARRIVALS[pattern](len(prompt_texts)),
                               max_new_tokens=MAX_NEW, ids=MIXED[:len(prompt_texts)])
    served, _ = eng32.run(reqs, n_slots=n_slots)

    bad = []
    for r, want in zip(served, alone32):
        if r.output != want:
            i = next((k for k in range(min(len(r.output), len(want)))
                      if r.output[k] != want[k]), min(len(r.output), len(want)))
            bad.append(f"  {r.id}: diverged at token {i} "
                       f"(got {len(r.output)} tok, want {len(want)})")
    assert not bad, (
        f"\nfloat32 continuous batching changed output "
        f"(arrivals={pattern}, n_slots={n_slots}), {len(bad)}/{len(served)} "
        f"prompts.\nThis is a real bug: a slot reused too early, a wrong decode "
        f"position, or a key mask that lets one slot see another's tail.\n"
        + "\n".join(bad))


def test_every_request_is_served_and_slots_are_returned(eng32, prompt_texts):
    """n_slots=1 forces total serialisation: every slot must be recycled."""
    reqs = eng32.make_requests(prompt_texts, arrivals=[0.0] * len(prompt_texts),
                               max_new_tokens=8, ids=MIXED[:len(prompt_texts)])
    served, summary = eng32.run(reqs, n_slots=1)
    assert all(r.state.value == "finished" for r in served)
    assert all(r.slot is None for r in served), "a finished request kept its slot"
    assert all(r.n_generated > 0 for r in served)
    assert summary["prefill_iterations"] == len(served)
