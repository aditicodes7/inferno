"""R5 acceptance: prefix caching must not change a single token.

The criterion your plan names is parity INCLUDING for requests that share a
prefix then diverge, because that is where copy-on-write bugs live. A CoW miss
does not crash: the two sequences quietly share a block, one overwrites the
other's K/V, and both produce fluent text that is wrong from the divergence
point on.

The strongest form of the assertion is that enabling the cache changes nothing
at all - same tokens as serving each request entirely alone, and the same tokens
as running with the cache disabled.
"""

from __future__ import annotations

import json
import os

import pytest

from tests.device import resolve_device
from tests.test_parity import ROOT

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = resolve_device()
MAX_NEW = 16


@pytest.fixture(scope="session")
def spec():
    return {p["id"]: p for p in
            json.loads((ROOT / "bench" / "prompts.json").read_text())["prompts"]}


@pytest.fixture(scope="session")
def eng():
    from inferno.prefix_engine import PrefixEngine
    return PrefixEngine(MODEL, device=DEVICE, dtype="float32", attn="eager")


def _alone(eng, texts):
    return [eng.generate(t, max_new_tokens=MAX_NEW) for t in texts]


def _check(served, want, context):
    bad = []
    for r, w in zip(served, want):
        if r.output != w:
            n = min(len(r.output), len(w))
            i = next((k for k in range(n) if r.output[k] != w[k]), n)
            bad.append(f"  {r.id}: diverged at token {i} "
                       f"(got {len(r.output)}, want {len(w)})")
    assert not bad, f"\nprefix caching changed output ({context})\n" + "\n".join(bad)


def test_shared_prefix_then_divergence(eng, spec):
    """The long prompts share one ~2000-char document, then ask different
    questions. Exactly the share-then-diverge shape."""
    ids = [f"long-{i:02d}" for i in range(4)]
    texts = [spec[i]["text"] for i in ids]
    want = _alone(eng, texts)

    reqs = eng.make_requests(texts, max_new_tokens=MAX_NEW, ids=ids)
    served, s = eng.run(reqs, n_blocks=512, block_size=16)
    _check(served, want, f"hit rate {s['hit_rate']:.2f}")
    assert s["hit_rate"] > 0.5, (
        "these prompts share a long document - a near-zero hit rate means the "
        "cache is not matching and this test is not exercising sharing")


def test_cache_on_equals_cache_off(eng, spec):
    ids = [f"long-{i:02d}" for i in range(3)] + ["short-00", "medium-00"]
    texts = [spec[i]["text"] for i in ids]
    reqs_off = eng.make_requests(texts, max_new_tokens=MAX_NEW, ids=ids)
    off, _ = eng.run(reqs_off, n_blocks=512, block_size=16,
                     enable_prefix_cache=False)
    reqs_on = eng.make_requests(texts, max_new_tokens=MAX_NEW, ids=ids)
    on, _ = eng.run(reqs_on, n_blocks=512, block_size=16,
                    enable_prefix_cache=True)
    _check(on, [r.output for r in off], "cache on vs cache off")


def test_divergence_inside_a_block_not_on_a_boundary(eng, spec):
    """Block size 64 makes the shared prefix end mid-block for most prompts,
    which is the case a boundary-aligned test would never reach."""
    ids = [f"long-{i:02d}" for i in range(3)]
    texts = [spec[i]["text"] for i in ids]
    want = _alone(eng, texts)
    reqs = eng.make_requests(texts, max_new_tokens=MAX_NEW, ids=ids)
    served, s = eng.run(reqs, n_blocks=128, block_size=64)
    _check(served, want, "block_size=64, divergence mid-block")


def test_identical_prompts_served_twice(eng, spec):
    """A total hit. At least one token must still be recomputed, because the
    cache holds K/V and never logits."""
    text = spec["long-00"]["text"]
    want = _alone(eng, [text])[0]
    reqs = eng.make_requests([text, text], max_new_tokens=MAX_NEW,
                             ids=["a", "b"])
    served, s = eng.run(reqs, n_blocks=512, block_size=16)
    _check(served, [want, want], "identical prompts")
    second = [p for p in s["per_request"] if p["id"] == "b"][0]
    assert second["computed_tokens"] >= 1
    assert second["cached_tokens"] > 0


def test_a_tight_pool_still_produces_identical_output(eng, spec):
    """Cache reclamation and preemption together, with parity intact."""
    ids = [f"long-{i:02d}" for i in range(3)]
    texts = [spec[i]["text"] for i in ids]
    want = _alone(eng, texts)
    reqs = eng.make_requests(texts, max_new_tokens=MAX_NEW, ids=ids)
    served, s = eng.run(reqs, n_blocks=40, block_size=16, max_concurrent=2)
    _check(served, want, f"tight pool, {s['preemptions']} preemptions")
