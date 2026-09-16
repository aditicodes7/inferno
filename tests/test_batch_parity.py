"""R2 acceptance: batching must not change a sequence's output.

TWO TIERS, because one criterion cannot be both meaningful and achievable here.

  float32 - EXACT, asserted. Batched output must be bit-identical to the same
            engine running the prompt alone, at every batch size. This is what
            actually tests the batching CODE: the mask, the position ids, the
            cache offsets, the left-padding scheme.

  float16 - FINITE, asserted; drift REPORTED, not asserted. Exact parity is
            unachievable in float16 and this is not a defect: a batched matmul
            reduces in a different order than an unbatched one, and greedy
            decoding amplifies a last-bit difference into a different
            continuation. Measured directly (PROJECT_LOG.md B4): long-01 has
            exactly 11 pad slots in the batch-of-2 run that passes and 11 in
            the batch-of-4 run that diverges at token 67 - identical padding,
            different batch size. Demanding exactness here would mean chasing
            a property of floating-point addition forever.

What float16 CAN be held to is that nothing becomes NaN. That failure mode was
real (B4a) and is fixed.

Run:
    .venv/bin/python -m pytest tests/test_batch_parity.py -v          # bs 1,2,4
    INFERNO_FULL=1 .venv/bin/python -m pytest tests/test_batch_parity.py -v
"""

from __future__ import annotations

import os

import pytest

from tests.test_parity import ROOT, prompts  # noqa: F401  (prompts is a fixture)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "mps"
MAX_NEW = 48

# Deliberately mixed lengths: ~36-token chat prompts batched alongside
# ~475-token RAG prompts. Uniform-length prompts barely exercise the padding
# path, which is the whole point of R2.
MIXED = [
    "short-00", "medium-00", "long-00", "short-01",
    "medium-01", "long-01", "short-02", "medium-02",
    "long-02", "short-03", "medium-03", "long-03",
    "short-04", "medium-04", "long-04", "short-05",
]

FULL = bool(os.environ.get("INFERNO_FULL"))
BATCH_SIZES = [1, 2, 4, 8, 16] if FULL else [1, 2, 4]
N_PROMPTS = 16 if FULL else 6


def _engine(dtype: str):
    from inferno.batch_engine import BatchEngine
    return BatchEngine(MODEL, device=DEVICE, dtype=dtype, attn="eager")


@pytest.fixture(scope="session")
def eng_fp32():
    return _engine("float32")


@pytest.fixture(scope="session")
def eng_fp16():
    return _engine("float16")


_ALONE_CACHE: dict = {}


def _alone(engine, prompts, ids):
    """Each prompt through the SAME code path, one at a time.

    Cached per dtype: the unbatched baseline does not depend on batch size, and
    recomputing it for every parametrised case dominates the test's runtime.
    """
    key = str(engine.dtype)
    if key not in _ALONE_CACHE:
        _ALONE_CACHE[key] = {
            p: engine.generate_batch([prompts[p]["text"]], max_new_tokens=MAX_NEW)[0]
            for p in ids}
    return _ALONE_CACHE[key]


def _batched(engine, prompts, ids, batch_size):
    out = {}
    for i in range(0, len(ids), batch_size):
        chunk = ids[i:i + batch_size]
        res = engine.generate_batch([prompts[p]["text"] for p in chunk],
                                    max_new_tokens=MAX_NEW)
        assert len(res) == len(chunk), "one output sequence per input prompt"
        out.update(zip(chunk, res))
    return out


def _divergence(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), None)
    if i is None and len(a) == len(b):
        return None
    return i if i is not None else n


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batching_is_exact_in_fp32(eng_fp32, prompts, batch_size):
    """THE LOGIC GATE. Batch composition must not change any output at all."""
    ids = MIXED[:N_PROMPTS]
    alone = _alone(eng_fp32, prompts, ids)
    together = _batched(eng_fp32, prompts, ids, batch_size)

    bad = [f"  {p}: diverged at token {_divergence(alone[p], together[p])} "
           f"(alone {len(alone[p])} tok, batched {len(together[p])} tok)"
           for p in ids if _divergence(alone[p], together[p]) is not None]
    assert not bad, (
        f"\nfloat32 batching changed output at batch_size={batch_size} "
        f"({len(bad)}/{len(ids)} prompts).\nThis is a real bug in the batching "
        f"code - mask, position ids, or cache offsets.\n" + "\n".join(bad)
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batching_stays_finite_in_fp16(eng_fp16, prompts, batch_size, capsys):
    """float16 may DRIFT, but must never produce NaN or degenerate output.

    The NaN failure was real (B4a): fully-masked padding rows plus an additive
    mask of finfo.min overflowed to -inf, and softmax of an all -inf row is
    NaN. It surfaced as a sequence emitting token id 0 forever.
    """
    ids = MIXED[:N_PROMPTS]
    alone = _alone(eng_fp16, prompts, ids)
    together = _batched(eng_fp16, prompts, ids, batch_size)

    degenerate = [p for p in ids if len(set(together[p][:8])) == 1]
    assert not degenerate, (
        f"\nfloat16 batch_size={batch_size}: degenerate output (a single token "
        f"repeated) from {degenerate}.\nThis is the NaN signature, not drift."
    )

    div = {p: _divergence(alone[p], together[p]) for p in ids}
    exact = [p for p, d in div.items() if d is None]
    drifted = {p: d for p, d in div.items() if d is not None}
    with capsys.disabled():
        print(f"\n    fp16 bs={batch_size}: {len(exact)}/{len(ids)} exact vs unbatched"
              + (f"; drift at {drifted}" if drifted else ""))
