"""R4: physical KV storage addressed through block tables.

Layout per layer is [n_blocks, block_size, n_kv_heads, head_dim]. Blocks first
so a block table indexes straight into dimension 0; block_size second so a
gathered run of blocks reshapes into a contiguous token axis without a permute
of the large axis.

THE GATHER IS THE COST OF THIS RUNG, and it is worth being explicit about why.
vLLM's PagedAttention is a custom CUDA kernel that walks the block table inside
the kernel and never materialises a contiguous tensor. We have no CUDA here, so
`gather` indexes the blocks out into a contiguous tensor every decode step -
correct, but it copies the live KV cache once per step. That difference is
expected to be the single largest line item in the final gap analysis, so its
cost is measured rather than asserted (rule 8).
"""

from __future__ import annotations

import torch


class PagedKVCache:
    def __init__(self, n_layers: int, n_blocks: int, block_size: int,
                 n_kv_heads: int, head_dim: int, dtype: torch.dtype,
                 device: str) -> None:
        shape = (n_blocks, block_size, n_kv_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device)
                  for _ in range(n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device)
                  for _ in range(n_layers)]
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.device = device
        self.gather_seconds = 0.0          # measured, see module docstring

    # -- writes -----------------------------------------------------------

    def write_prefill(self, layer: int, table: list[int], n_tokens: int,
                      k: torch.Tensor, v: torch.Tensor, start: int = 0) -> None:
        """Scatter a prompt across its blocks. k/v are [1, n_kv, T, hd].

        `start` skips leading positions already filled by a prefix-cache hit,
        so those blocks are written once by whoever computed them and then only
        ever read.
        """
        pos = torch.arange(start, start + n_tokens, device=self.device)
        tbl = torch.tensor(table, device=self.device)
        blk = tbl[pos // self.block_size]
        off = pos % self.block_size
        self.k[layer][blk, off] = k[0].transpose(0, 1)     # [T, n_kv, hd]
        self.v[layer][blk, off] = v[0].transpose(0, 1)

    def write_decode(self, layer: int, blocks: torch.Tensor,
                     offsets: torch.Tensor, k: torch.Tensor,
                     v: torch.Tensor) -> None:
        """One token per sequence, each at its own (block, offset)."""
        self.k[layer][blocks, offsets] = k[:, :, 0]        # [B, n_kv, hd]
        self.v[layer][blocks, offsets] = v[:, :, 0]

    # -- the gather -------------------------------------------------------

    def gather(self, layer: int, tables: torch.Tensor):
        """Materialise [B, n_kv, n_blocks*block_size, head_dim] from block tables.

        `tables` is [B, max_blocks], right-padded with any block id: padded
        entries sit beyond the sequence length and are masked away by the
        caller. This is the copy that a fused paged-attention kernel avoids.
        """
        b, nb = tables.shape
        k = self.k[layer][tables]                          # [B, NB, bs, n_kv, hd]
        v = self.v[layer][tables]
        k = k.reshape(b, nb * self.block_size, *k.shape[3:]).permute(0, 2, 1, 3)
        v = v.reshape(b, nb * self.block_size, *v.shape[3:]).permute(0, 2, 1, 3)
        return k, v

    # -- measurement ------------------------------------------------------

    def bytes_allocated(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)
