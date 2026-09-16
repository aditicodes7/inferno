"""The KV cache - R1's naive version.

Stores, per layer, the key and value vectors of every position seen so far.
This exists because attention is the only operation in a transformer that
mixes information across positions, and a past position enters that operation
in exactly two roles: as a key to be scored against, and as a value to be
averaged in. Those two tensors are therefore a sufficient summary of the past.
Past queries are never needed again - causality guarantees a position never
attends to anything after itself.

R1 DESIGN CHOICE, deliberately naive: storage is preallocated to
prompt_len + max_new_tokens and never grows. This is exactly the behaviour
that paged attention exists to fix. Two consequences, both of which R4 will
measure:

  1. Every sequence reserves for its worst case. A request that stops after
     8 tokens still holds memory for 128.
  2. The reservation must be contiguous, so free memory that is merely
     fragmented cannot be used.

Keeping that waste visible here is the point. `utilisation()` reports it.
"""

from __future__ import annotations

import torch


class KVCache:
    """Contiguous per-layer KV storage for a single sequence."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 max_len: int, dtype: torch.dtype, device: str,
                 batch_size: int = 1) -> None:
        shape = (batch_size, n_kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device)
                  for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device)
                  for _ in range(n_layers)]
        self.max_len = max_len
        self.length = 0          # positions actually written, across all layers
        self._dtype = dtype
        self._n_layers = n_layers

    def write(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor):
        """Write this layer's new K/V at `start`, return everything up to the end.

        `start` is passed in rather than read from `self.length` because every
        layer writes at the same offset during one forward pass. The length
        advances once per forward, not once per layer - see `advance`.
        """
        end = start + k.shape[2]
        if end > self.max_len:
            raise ValueError(
                f"KV cache overflow: writing to {end} but capacity is "
                f"{self.max_len}. The cache is preallocated and cannot grow."
            )
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n: int) -> None:
        """Called once per forward pass, after every layer has written."""
        self.length += n

    # -- measurement ------------------------------------------------------

    def bytes_allocated(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def bytes_used(self) -> int:
        if self.max_len == 0:
            return 0
        return int(self.bytes_allocated() * self.length / self.max_len)

    def utilisation(self) -> float:
        """Fraction of allocated KV memory that holds a real token.

        This is the number R4 is built to improve.
        """
        return self.length / self.max_len if self.max_len else 0.0
