"""R4 acceptance: paging must not change output, at any block size, and must
not leak a block even when memory pressure forces preemption.

Preemption is the case that cannot happen in R0-R3, because everything was
reserved in advance. Here a running request can be evicted mid-generation, have
its blocks returned, and be recomputed from its prompt later. Two things must
hold through that: the recomputed output is identical, and every block comes
back. `PagedEngine.run` raises if any block is outstanding at the end, so a
leak fails the test rather than surfacing as premature OOM much later.
"""

from __future__ import annotations

import json
import os

import pytest

from tests.device import resolve_device
from tests.test_parity import ROOT

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = resolve_device()
MAX_NEW = 24

MIXED = ["short-00", "medium-00", "short-01", "medium-01", "short-02", "long-00"]
N = 6 if os.environ.get("INFERNO_FULL") else 4


@pytest.fixture(scope="session")
def texts():
    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    by_id = {p["id"]: p for p in spec["prompts"]}
    return [by_id[i]["text"] for i in MIXED[:N]]


@pytest.fixture(scope="session")
def eng():
    from inferno.paged_engine import PagedEngine
    return PagedEngine(MODEL, device=DEVICE, dtype="float32", attn="eager")


@pytest.fixture(scope="session")
def alone(eng, texts):
    return [eng.generate(t, max_new_tokens=MAX_NEW) for t in texts]


def _check(served, alone, context):
    bad = []
    for r, want in zip(served, alone):
        if r.output != want:
            n = min(len(r.output), len(want))
            i = next((k for k in range(n) if r.output[k] != want[k]), n)
            bad.append(f"  {r.id}: diverged at token {i} "
                       f"(got {len(r.output)}, want {len(want)})")
    assert not bad, f"\npaged output differs ({context})\n" + "\n".join(bad)


@pytest.mark.parametrize("block_size", [4, 16, 64])
def test_parity_across_block_sizes(eng, texts, alone, block_size):
    """Block size is a memory-layout choice and must not be observable."""
    reqs = eng.make_requests(texts, max_new_tokens=MAX_NEW, ids=MIXED[:len(texts)])
    served, summary = eng.run(reqs, n_blocks=256, block_size=block_size)
    _check(served, alone, f"block_size={block_size}")
    assert summary["blocks_leaked"] == 0


def test_parity_under_memory_pressure_with_preemption(eng, texts, alone):
    """Tight pool: RUNNING requests cannot grow and are evicted mid-generation.

    The configuration matters. Preemption now only ever serves a running
    sequence that cannot grow (docs/bugs.md - evicting on behalf of a waiting
    request livelocked). So the pool must be big enough to admit several
    requests and too small for them all to GROW, which means small blocks and a
    pool sized just above the prompts.
    """
    # short-00 (36 tok) and short-01 (38 tok). At block_size 4 their prompts
    # need 9 + 10 = 19 blocks; each needs <= 16 blocks to run to completion, but
    # both together need 27. A 20-block pool admits both with ONE block spare,
    # so the second time either crosses a block boundary there is nothing left
    # and a running sequence must be evicted - the path under test.
    #
    # 22 blocks was not enough pressure: these prompts hit EOS well before
    # max_new_tokens, so three spare blocks covered all the growth and nothing
    # was ever preempted. The assertion below is what caught that.
    picked = [0, 2]
    sub = [texts[i] for i in picked]
    sub_ids = [MIXED[i] for i in picked]
    reqs = eng.make_requests(sub, max_new_tokens=MAX_NEW, ids=sub_ids)
    served, summary = eng.run(reqs, n_blocks=20, block_size=4)
    assert summary["preemptions"] > 0, (
        "no preemption happened - this test is not exercising the path it "
        "exists for; shrink the pool")
    _check(served, [alone[i] for i in picked],
           f"{summary['preemptions']} preemptions")
    assert summary["blocks_leaked"] == 0, "blocks leaked across preemption"


def test_a_single_sequence_at_a_time_still_returns_every_block(eng, texts, alone):
    reqs = eng.make_requests(texts, max_new_tokens=MAX_NEW, ids=MIXED[:len(texts)])
    served, summary = eng.run(reqs, n_blocks=64, block_size=16, max_concurrent=1)
    _check(served, alone, "max_concurrent=1")
    assert summary["peak_concurrent_sequences"] == 1
    assert summary["blocks_leaked"] == 0


def test_paging_wastes_less_than_a_block_per_sequence(eng, texts):
    """The headline property: internal waste is bounded by block_size."""
    reqs = eng.make_requests(texts, max_new_tokens=MAX_NEW, ids=MIXED[:len(texts)])
    _, summary = eng.run(reqs, n_blocks=256, block_size=16)
    assert summary["block_utilisation_mean"] > 0.75, (
        f"block utilisation {summary['block_utilisation_mean']:.2f} is lower "
        f"than paging should allow")
