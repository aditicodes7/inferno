"""R4: the block allocator. KV cache as virtual memory.

R0-R3 gave every sequence a contiguous reservation of `max_len`, so an 8-token
sequence cost exactly as much as a 500-token one and max concurrency was fixed
in advance at `memory / max_len`.

This chops the cache into fixed-size blocks and hands them out on demand. A
sequence's KV lives in a scattered list of physical blocks; `tables[seq_id]`
maps logical block index -> physical block number. It is a page table.

Two properties follow from fixed-size blocks, and both matter:

  * Internal waste per sequence is bounded by `block_size - 1` tokens instead
    of `max_len - actual`. With 16-token blocks a sequence wastes at most 15.
  * External fragmentation is impossible - every block is interchangeable, so
    any free block satisfies any request. Variable-size blocks would
    reintroduce exactly the fragmentation problem this exists to remove.

REFERENCE COUNTING is here before anything shares. At R4 every count is 0 or 1
and it looks like ceremony; R5's prefix sharing is the reason, and retrofitting
refcounts into a working allocator is far worse than having them from the start.

This module owns no tensors. The physical KV lives in `PagedKVCache`; the
allocator only decides who owns which block number. That separation is what
lets every silent failure mode - leaks, double frees, use-after-free - be
tested without a model.
"""

from __future__ import annotations


class OutOfBlocks(RuntimeError):
    """Raised instead of partially satisfying an allocation."""


class BlockManager:
    def __init__(self, n_blocks: int, block_size: int) -> None:
        self.n_blocks = n_blocks
        self.block_size = block_size
        # Newest-freed is reused first, which keeps recently touched blocks hot.
        self.free_blocks: list[int] = list(range(n_blocks))
        self.ref_count: list[int] = [0] * n_blocks
        self.tables: dict[str, list[int]] = {}
        self.lengths: dict[str, int] = {}

    # -- introspection ----------------------------------------------------

    @property
    def n_free(self) -> int:
        return len(self.free_blocks)

    def n_blocks_for(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, n_tokens: int) -> bool:
        return self.n_blocks_for(n_tokens) <= self.n_free

    def block_table(self, seq_id: str) -> list[int]:
        return self.tables[seq_id]

    def length(self, seq_id: str) -> int:
        return self.lengths[seq_id]

    def locate(self, seq_id: str, position: int) -> tuple[int, int]:
        """Logical token position -> (physical block, offset within it)."""
        if position < 0 or position >= self.lengths[seq_id]:
            raise IndexError(
                f"position {position} outside sequence {seq_id!r} "
                f"of length {self.lengths[seq_id]}")
        return (self.tables[seq_id][position // self.block_size],
                position % self.block_size)

    # -- allocation -------------------------------------------------------

    def _take(self, n: int) -> list[int]:
        """All or nothing. A partially satisfied allocation would consume
        blocks the caller cannot use and has no way to give back."""
        if n > self.n_free:
            raise OutOfBlocks(
                f"need {n} blocks, {self.n_free} free of {self.n_blocks}")
        taken = [self.free_blocks.pop(0) for _ in range(n)]
        for b in taken:
            self.ref_count[b] += 1
        return taken

    def allocate(self, seq_id: str, n_tokens: int) -> list[int]:
        if seq_id in self.tables:
            raise KeyError(f"{seq_id!r} already has blocks allocated")
        table = self._take(self.n_blocks_for(n_tokens))
        self.tables[seq_id] = table
        self.lengths[seq_id] = n_tokens
        return table

    def append(self, seq_id: str) -> int | None:
        """Grow a sequence by one token. Returns a new block if one was needed.

        A block is allocated only when the last one is exactly full, which is
        what keeps waste bounded rather than proportional to max_len.
        """
        length = self.lengths[seq_id]
        if length % self.block_size == 0:
            block = self._take(1)[0]
            self.tables[seq_id].append(block)
            self.lengths[seq_id] = length + 1
            return block
        self.lengths[seq_id] = length + 1
        return None

    # -- sharing and release ----------------------------------------------

    def share(self, src_id: str, dst_id: str) -> list[int]:
        """Point `dst` at `src`'s blocks. No copy, no new allocation."""
        if dst_id in self.tables:
            raise KeyError(f"{dst_id!r} already has blocks allocated")
        table = list(self.tables[src_id])
        for b in table:
            self.ref_count[b] += 1
        self.tables[dst_id] = table
        self.lengths[dst_id] = self.lengths[src_id]
        return table

    def free(self, seq_id: str) -> None:
        """Drop this sequence's claim. A block returns to the pool at zero.

        `tables.pop` raises KeyError on a second free, which is deliberate: a
        double free would put one physical block on the free list twice, and
        two sequences would later be handed the same memory.
        """
        table = self.tables.pop(seq_id)
        self.lengths.pop(seq_id, None)
        for b in table:
            self.ref_count[b] -= 1
            if self.ref_count[b] == 0:
                self.free_blocks.insert(0, b)
            elif self.ref_count[b] < 0:
                raise RuntimeError(f"refcount for block {b} went negative")

    # -- the numbers R4 exists to produce ---------------------------------

    def utilisation(self) -> float:
        """Fraction of the WHOLE cache holding a real token."""
        return sum(self.lengths.values()) / (self.n_blocks * self.block_size)

    def block_utilisation(self) -> float:
        """Fraction of ALLOCATED blocks holding a real token - i.e. how much of
        what we handed out is actually used. This is the number paging improves."""
        referenced = {b for t in self.tables.values() for b in t}
        if not referenced:
            return 0.0
        return sum(self.lengths.values()) / (len(referenced) * self.block_size)

    def stats(self) -> dict:
        return {"n_blocks": self.n_blocks, "block_size": self.block_size,
                "free": self.n_free, "sequences": len(self.tables),
                "tokens": sum(self.lengths.values()),
                "utilisation": self.utilisation(),
                "block_utilisation": self.block_utilisation()}
