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


class SlotKVCache:
    """R3: per-slot KV storage with INDEPENDENT lengths.

    R1/R2's `KVCache` carries one `length` for the whole batch. That works only
    because left-padding right-aligns every sequence so they all write at the
    same offset. Continuous batching destroys that: a request admitted at
    iteration 50 has 0 generated tokens while its batch-mates have 50, and they
    are never aligned again. So length becomes per slot.

    A sequence owns a slot for its lifetime. When it finishes the slot is freed
    and reused. Every slot reserves `max_len` whether it needs it or not, so a
    slot holding an 8-token sequence costs exactly as much as one holding 500.
    That waste is now structural rather than incidental, and it is what R4's
    block allocator exists to remove.
    """

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 n_slots: int, max_len: int, dtype: torch.dtype,
                 device: str) -> None:
        shape = (n_slots, n_kv_heads, max_len, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device)
                  for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device)
                  for _ in range(n_layers)]
        self.n_slots = n_slots
        self.max_len = max_len
        self.device = device
        self.lengths = torch.zeros(n_slots, dtype=torch.long, device=device)

    # -- slot lifecycle ---------------------------------------------------

    def reset(self, slot: int) -> None:
        """Release a slot. The stale K/V are never read again because the key
        mask is built from `lengths`, so zeroing the tensors is unnecessary -
        and skipping it keeps slot reuse O(1) instead of O(max_len)."""
        self.lengths[slot] = 0

    # -- writes -----------------------------------------------------------

    def write_prefill(self, layer: int, slot: int, k: torch.Tensor,
                      v: torch.Tensor):
        """Write a whole prompt into one slot. k/v are [1, n_kv, T, head_dim]."""
        t = k.shape[2]
        if t > self.max_len:
            raise ValueError(f"prompt of {t} exceeds slot capacity {self.max_len}")
        self.k[layer][slot, :, :t] = k[0]
        self.v[layer][slot, :, :t] = v[0]
        return self.k[layer][slot:slot + 1, :, :t], self.v[layer][slot:slot + 1, :, :t]

    def write_decode(self, layer: int, slots: torch.Tensor, k: torch.Tensor,
                     v: torch.Tensor, read_len: int):
        """Append one token per slot, each at ITS OWN offset, then read back.

        `slots` is [B]; k/v are [B, n_kv, 1, head_dim]. The scatter uses two
        index tensors so every row lands at a different position - that is the
        whole difference from R2, where one offset served the entire batch.
        """
        pos = self.lengths[slots]
        self.k[layer][slots, :, pos] = k[:, :, 0]
        self.v[layer][slots, :, pos] = v[:, :, 0]
        return self.k[layer][slots][:, :, :read_len], self.v[layer][slots][:, :, :read_len]

    def advance(self, slots: torch.Tensor) -> None:
        self.lengths[slots] += 1

    # -- measurement ------------------------------------------------------

    def bytes_allocated(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def utilisation(self) -> float:
        """Fraction of reserved KV holding a real token, across all slots."""
        return float(self.lengths.sum()) / (self.n_slots * self.max_len)
