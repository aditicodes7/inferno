"""R5: prefix caching with copy-on-write.

Attention is causal, so position i's K/V depend only on tokens 0..i. Two
sequences that begin with identical tokens have BIT-IDENTICAL KV across that
span - not similar, identical. So the blocks holding it are shared rather than
recomputed, using the reference counting built at R4 for exactly this.

THE KEY IS THE WHOLE PREFIX, NOT THE BLOCK'S CONTENTS. Block 3's KV depends on
blocks 0-2, so two sequences whose block 3 holds the same tokens after different
preceding tokens have entirely different KV there. Keying on contents alone
hands a sequence someone else's activations and produces fluent, wrong output
with no error anywhere.

Keys are exact token tuples rather than a 64-bit digest. A hash collision here
is undetectable at runtime and yields plausible text, and this project's whole
premise is token-identical output. The cost is O(prefix^2 / block_size) memory
in cached keys, which is fine at benchmark scale and would want a
(parent_digest, block_tokens) scheme plus verification in production.

TWO TIERS OF FREE. A block whose refcount hits zero is not discarded if it holds
published KV - it moves to an LRU of cached-but-reclaimable blocks. Without that
tier, prefix caching only ever helps requests that overlap in time, which is not
the workload it exists for.

This module owns no tensors. `append_cow` returns the id of a freshly allocated
block and the ENGINE performs the physical copy; the manager only decides
ownership.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from inferno.block_manager import BlockManager, OutOfBlocks


@dataclass
class Allocation:
    table: list[int]
    n_cached_blocks: int
    n_cached_tokens: int


class PrefixBlockManager(BlockManager):
    def __init__(self, n_blocks: int, block_size: int) -> None:
        super().__init__(n_blocks, block_size)
        self.cached: dict[tuple, int] = {}          # prefix key -> physical block
        self.block_key: dict[int, tuple] = {}       # physical block -> its key
        self.evictable: OrderedDict[int, None] = OrderedDict()   # LRU, refcount 0
        self._requested = 0
        self._hit = 0
        # new block -> the block it was copied FROM, so the engine can move the
        # K/V across. append_cow only reassigns ownership; the bytes still live
        # in the block the other sequence holds.
        self.cow_source: dict[int, int] = {}

    # -- keys -------------------------------------------------------------

    def _key(self, tokens: list[int], k: int) -> tuple:
        """Identity of logical block k: every token up to and including it."""
        return tuple(tokens[: (k + 1) * self.block_size])

    # -- allocation with reclamation --------------------------------------

    def _take(self, n: int) -> list[int]:
        """Reclaim from the cached tier before declaring the pool empty.

        Feasibility is checked BEFORE anything is reclaimed, so a doomed
        allocation does not destroy cache entries on its way to failing.
        """
        if n > len(self.free_blocks) + len(self.evictable):
            raise OutOfBlocks(
                f"need {n} blocks; {len(self.free_blocks)} free + "
                f"{len(self.evictable)} reclaimable of {self.n_blocks}")
        while len(self.free_blocks) < n:
            block, _ = self.evictable.popitem(last=False)     # least recent
            key = self.block_key.pop(block, None)
            if key is not None:
                # The block is about to hold different KV. A surviving cache
                # entry would later be served as a hit and return the wrong
                # activations.
                self.cached.pop(key, None)
            self.free_blocks.append(block)
        return super()._take(n)

    def allocate_with_prefix(self, seq_id: str, tokens: list[int],
                             use_cache: bool = True) -> Allocation:
        """Share every leading FULL block already in the cache; allocate the rest.

        Matching stops at the first miss - a prefix is only a prefix if it is
        unbroken, and a later coincidental match belongs to a different history.
        """
        if seq_id in self.tables:
            raise KeyError(f"{seq_id!r} already has blocks allocated")
        n_tokens = len(tokens)
        n_needed = self.n_blocks_for(n_tokens)
        n_full = n_tokens // self.block_size

        table: list[int] = []
        for k in range(n_full if use_cache else 0):
            block = self.cached.get(self._key(tokens, k))
            if block is None:
                break
            self.evictable.pop(block, None)          # revive from the LRU tier
            self.ref_count[block] += 1
            table.append(block)

        n_hits = len(table)
        table += self._take(n_needed - n_hits)
        self.tables[seq_id] = table
        self.lengths[seq_id] = n_tokens
        self._requested += n_needed
        self._hit += n_hits
        return Allocation(table, n_hits, n_hits * self.block_size)

    def publish(self, seq_id: str, tokens: list[int]) -> int:
        """Register this sequence's FULL blocks as reusable. Call AFTER prefill.

        Only full blocks: a partially filled block's contents still change as
        tokens arrive, so its key is not yet stable. And only after the KV
        actually exists - publishing at allocation time would advertise blocks
        that have not been computed.
        """
        table = self.tables[seq_id]
        published = 0
        for k in range(len(tokens) // self.block_size):
            key = self._key(tokens, k)
            if key not in self.cached:
                self.cached[key] = table[k]
                self.block_key[table[k]] = key
                published += 1
        return published

    # -- sharing and copy-on-write ----------------------------------------

    def share_tail(self, src_id: str, dst_id: str) -> list[int]:
        """Share every block including the PARTIAL tail.

        This is what makes copy-on-write necessary. Sharing only full blocks
        needs no CoW, because a sequence then writes exclusively into blocks it
        allocated itself.
        """
        return self.share(src_id, dst_id)

    def append_cow(self, seq_id: str) -> int | None:
        """Grow by one token, copying first if the target block is shared.

        Returns the new block id when a copy happened, so the engine can move
        the KV across; None when the sequence already owned the block outright.
        """
        length = self.lengths[seq_id]
        if length % self.block_size == 0:
            self.tables[seq_id].append(self._take(1)[0])
            self.lengths[seq_id] = length + 1
            return None                              # fresh block, nothing to copy

        last = self.tables[seq_id][-1]
        copied = None
        if self.ref_count[last] > 1:
            new = self._take(1)[0]
            self.ref_count[last] -= 1
            self.tables[seq_id][-1] = new
            self.cow_source[new] = last
            copied = new
        self.lengths[seq_id] = length + 1
        return copied

    # -- release ----------------------------------------------------------

    def free(self, seq_id: str) -> None:
        """Zero refcount means reusable, not worthless: a published block moves
        to the cached tier instead of straight back to the free list."""
        table = self.tables.pop(seq_id)
        self.lengths.pop(seq_id, None)
        for b in table:
            self.ref_count[b] -= 1
            if self.ref_count[b] == 0:
                if b in self.block_key:
                    self.evictable[b] = None
                else:
                    self.free_blocks.insert(0, b)
            elif self.ref_count[b] < 0:
                raise RuntimeError(f"refcount for block {b} went negative")

    # -- measurement ------------------------------------------------------

    def prefix_stats(self) -> dict:
        return {"blocks_requested": self._requested, "blocks_hit": self._hit,
                "hit_rate": (self._hit / self._requested) if self._requested else 0.0,
                "cached_blocks": len(self.cached),
                "evictable_blocks": len(self.evictable)}
