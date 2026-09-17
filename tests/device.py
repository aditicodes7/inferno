"""Which device the tests run on.

Hardcoding "mps" meant every model-backed test file errored out on the first
machine that was not this laptop. The rung tests are not about a device: the
self-consistency ones (batched vs unbatched, paged vs contiguous, cache on vs
off) assert a property that must hold wherever they run.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def resolve_device() -> str:
    """$INFERNO_DEVICE wins, then CUDA, then MPS, then CPU."""
    env = os.environ.get("INFERNO_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def results_dir(device: str) -> Path:
    return ROOT / "results" / ("gpu" if device == "cuda" else "mac")


def find_eager_reference(device: str) -> Path | None:
    """The newest eager R0 reference generated ON THIS DEVICE.

    Parity against a reference from a different device is not a meaningful
    assertion: CUDA and MPS kernels reduce in different orders, and greedy
    decoding turns a last-bit difference into a different paragraph
    (PROJECT_LOG.md B1). So each device needs its own reference rather than
    borrowing one and reporting spurious failures.
    """
    found = sorted(results_dir(device).glob(f"r0_baseline_{device}_eager_*.json"))
    return found[-1] if found else None
