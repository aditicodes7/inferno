"""R1 acceptance: Inferno must reproduce R0's tokens EXACTLY.

This is the spine of the project. Every rung from here to R5 is accepted by
this same assertion, so it is worth being pedantic about what it compares.

Comparison is against the EAGER reference, never the sdpa one. Measured on
2026-09-15: sdpa and eager disagree on 10 of 50 prompts under greedy decoding
(see docs/decisions.md). Pointing this test at the sdpa file would produce ten
failures with no bug behind them.

Run:
    .venv/bin/python -m pytest tests/test_parity.py -v        # 6-prompt subset
    INFERNO_FULL=1 .venv/bin/python -m pytest tests/test_parity.py -v   # all 50
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.device import find_eager_reference, resolve_device

ROOT = Path(__file__).resolve().parent.parent

# The reference must come from the SAME DEVICE the tests run on. CUDA and MPS
# kernels reduce in different orders, and greedy decoding turns a last-bit
# difference into a different paragraph (PROJECT_LOG.md B1), so borrowing
# another machine's reference reports failures that are not defects.
REFERENCE = find_eager_reference(resolve_device())

# One from each length regime, plus three that R0 showed are fragile: short-15
# diverged between attention backends at generated token 1, long-08 at token 8,
# and short-09 terminated at different lengths. If a cache bug exists, these
# are the prompts most likely to expose it early.
SUBSET = ["short-00", "short-09", "short-15", "medium-01", "long-00", "long-08"]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def reference() -> dict:
    device = resolve_device()
    if REFERENCE is None:
        pytest.skip(
            f"No eager R0 reference for device={device!r}. This is expected the "
            f"first time on new hardware - generate one with:\n"
            f"    python bench/run_baseline.py --device {device} --attn eager")
    data = json.loads(REFERENCE.read_text())
    cfg = data["config"]
    # Guard against the single most expensive mistake available here.
    assert cfg["attn_implementation"] == "eager", (
        f"Reference was produced with attn={cfg['attn_implementation']!r}. "
        "Parity is only meaningful against an eager reference."
    )
    # "greedy" alone is NOT a complete specification of a decoding procedure.
    # The checkpoint ships repetition_penalty=1.1 and HuggingFace applies it in
    # greedy decoding; inheriting it silently cost a full debugging cycle
    # (PROJECT_LOG.md B3). Assert the whole processor stack is neutral.
    s = cfg["sampling"]
    assert s["do_sample"] is False and s["num_beams"] == 1, f"not greedy: {s}"
    assert s["repetition_penalty"] == 1.0, (
        f"Reference was generated with repetition_penalty={s['repetition_penalty']}. "
        "Inferno takes raw argmax, so parity against it is not achievable."
    )
    assert s["no_repeat_ngram_size"] in (0, None), f"ngram blocking active: {s}"
    assert s["renormalize_logits"] is False, f"logits renormalised: {s}"
    return data


@pytest.fixture(scope="session")
def expected(reference) -> dict[str, dict]:
    return {r["prompt_id"]: r for r in reference["records"]}


@pytest.fixture(scope="session")
def prompts() -> dict[str, dict]:
    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    return {p["id"]: p for p in spec["prompts"]}


@pytest.fixture(scope="session")
def engine(reference):
    """Build Inferno configured identically to the reference run.

    Device, dtype and attention implementation are read from the reference
    rather than chosen here. Parity against a reference generated under
    different numerics is not a meaningful assertion.
    """
    cfg = reference["config"]
    try:
        from inferno.engine import InfernoEngine
    except ImportError as e:
        pytest.fail(
            f"Cannot import Inferno: {e}\n"
            "R1 is not implemented yet. This failure is the point - the test "
            "is written before the implementation (rule 4)."
        )

    return InfernoEngine(
        model_id=cfg["model"],
        device=cfg["device"],
        dtype=cfg["dtype"].removeprefix("torch."),
        attn=cfg["attn_implementation"],
    )


# ---------------------------------------------------------------------------
# the assertion
# ---------------------------------------------------------------------------

def _prompt_ids() -> list[str]:
    if os.environ.get("INFERNO_FULL"):
        spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
        return [p["id"] for p in spec["prompts"]]
    return SUBSET


def describe_divergence(got: list[int], want: list[int]) -> str:
    """Say WHERE it diverged, not just THAT it did.

    The index is the single most useful debugging signal available. Divergence
    at token 0 means prefill, weight loading, or the output projection - the
    cache has not been read yet. Divergence at token 1 means the first decode
    step, so the cache was just written and read for the first time.
    Divergence at token 40 after 39 exact matches means something that
    accumulates, not something structurally wrong.
    """
    n = min(len(got), len(want))
    idx = next((i for i in range(n) if got[i] != want[i]), None)
    if idx is None:
        return (f"identical for the first {n} tokens, but lengths differ: "
                f"got {len(got)}, want {len(want)} - check the stop condition")
    where = {
        0: "token 0 - prefill, weight loading, or the output projection. "
           "The cache has not been read yet.",
        1: "token 1 - the first decode step. The cache was written once and "
           "read once.",
    }.get(idx, f"token {idx} - {idx} tokens matched exactly first.")
    return (f"diverged at {where}\n"
            f"  got : ...{got[max(0, idx - 3):idx + 3]}\n"
            f"  want: ...{want[max(0, idx - 3):idx + 3]}\n"
            f"  lengths: got {len(got)}, want {len(want)}")


@pytest.mark.parametrize("prompt_id", _prompt_ids())
def test_token_parity(engine, prompts, expected, reference, prompt_id):
    """Token-identical output to R0. Not similar - identical."""
    ref = expected[prompt_id]
    got = engine.generate(
        prompts[prompt_id]["text"],
        max_new_tokens=reference["config"]["max_new_tokens"],
    )
    assert got == ref["token_ids"], (
        f"\nPARITY FAILED on {prompt_id} "
        f"({ref['n_prompt_tokens']} prompt tokens)\n  "
        + describe_divergence(got, ref["token_ids"])
    )
