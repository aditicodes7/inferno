"""R2 acceptance: parity must hold at EVERY batch size.

This is a strictly stronger assertion than R1's. At batch size 1 we proved a
sequence's output does not depend on what ran *before* it. Here we assert it
does not depend on what it is batched *with* - that padding, masking and
position numbering are all correct, and that no sequence reads another's cache.

Two quite different things can break it:
  1. A real bug - padding leaking into attention, or wrong cache offsets.
     Produces visible corruption.
  2. Legitimate numerics - a batched matmul reduces in a different order than
     an unbatched one. Under greedy decoding a last-bit difference can flip an
     argmax and diverge the whole continuation (see PROJECT_LOG.md B1).

If this fails at batch 16 but holds at batch 2, (2) is live and
bench/diagnose_parity.py is the tool that separates them.

Run:
    .venv/bin/python -m pytest tests/test_batch_parity.py -v          # bs 1,2,4
    INFERNO_FULL=1 .venv/bin/python -m pytest tests/test_batch_parity.py -v
"""

from __future__ import annotations

import json
import os

import pytest

from tests.test_parity import REFERENCE, ROOT, describe_divergence

# Deliberately mixed lengths: ~36-token chat prompts batched alongside
# ~475-token RAG prompts. A batch of uniform-length prompts would barely
# exercise the padding path, which is the whole point of R2.
MIXED = [
    "short-00", "short-01", "short-02", "short-03",
    "medium-00", "medium-01",
    "long-00", "long-01",
    "short-04", "short-05", "short-06", "short-07",
    "medium-02", "medium-03",
    "long-02", "long-03",
]

BATCH_SIZES = [1, 2, 4, 8, 16] if os.environ.get("INFERNO_FULL") else [1, 2, 4]


@pytest.fixture(scope="session")
def reference() -> dict:
    data = json.loads(REFERENCE.read_text())
    s = data["config"]["sampling"]
    assert s["repetition_penalty"] == 1.0 and s["do_sample"] is False
    return data


@pytest.fixture(scope="session")
def expected(reference) -> dict[str, dict]:
    return {r["prompt_id"]: r for r in reference["records"]}


@pytest.fixture(scope="session")
def prompts() -> dict[str, dict]:
    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    return {p["id"]: p for p in spec["prompts"]}


@pytest.fixture(scope="session")
def batch_engine(reference):
    cfg = reference["config"]
    try:
        from inferno.batch_engine import BatchEngine
    except ImportError as e:
        pytest.fail(
            f"Cannot import BatchEngine: {e}\n"
            "R2 is not implemented yet - this failure is the point (rule 4)."
        )
    return BatchEngine(
        model_id=cfg["model"],
        device=cfg["device"],
        dtype=cfg["dtype"].removeprefix("torch."),
        attn=cfg["attn_implementation"],
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batch_parity(batch_engine, prompts, expected, reference, batch_size):
    """Every prompt must produce its reference tokens, at every batch size."""
    n = len(MIXED) if os.environ.get("INFERNO_FULL") else 8
    ids = MIXED[:n]
    max_new = reference["config"]["max_new_tokens"]

    got: dict[str, list[int]] = {}
    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i + batch_size]
        outs = batch_engine.generate_batch(
            [prompts[p]["text"] for p in chunk], max_new_tokens=max_new
        )
        assert len(outs) == len(chunk), "one output sequence per input prompt"
        got.update(zip(chunk, outs))

    failures = []
    for pid in ids:
        want = expected[pid]["token_ids"]
        if got[pid] != want:
            failures.append(f"  {pid}: " + describe_divergence(got[pid], want))
    assert not failures, (
        f"\nBATCH PARITY FAILED at batch_size={batch_size} "
        f"({len(failures)}/{len(ids)} prompts)\n" + "\n".join(failures)
    )
