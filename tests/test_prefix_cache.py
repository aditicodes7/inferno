"""R5 prefix caching. No tensors, no model.

Attention is causal, so position i's K/V depend only on tokens 0..i. Two
sequences that begin with identical tokens therefore have BIT-IDENTICAL KV for
that span, and the blocks holding it can be shared rather than recomputed.

The correctness hazard is the key. A block must be identified by the ENTIRE
token prefix up to and including it, never by its own contents: block 3's KV
depends on blocks 0-2, so two sequences whose block 3 holds the same 16 tokens
after different preceding tokens have completely different KV there. Keying on
contents alone hands a sequence someone else's activations and produces fluent,
wrong output with no error anywhere. `test_identical_block_after_different_prefix_is_not_shared`
is the one that catches it.
"""

from __future__ import annotations

import pytest

from inferno.prefix_cache import PrefixBlockManager


def conserved(bm: PrefixBlockManager) -> bool:
    """Every block is in exactly one of: referenced, free, cached-evictable."""
    referenced = {b for t in bm.tables.values() for b in t}
    free = set(bm.free_blocks)
    evictable = set(bm.evictable)
    return (len(bm.free_blocks) == len(free)
            and not (referenced & free)
            and not (referenced & evictable)
            and not (free & evictable)
            and len(free) + len(evictable) + len(referenced) == bm.n_blocks)


def toks(*runs: tuple[int, int]) -> list[int]:
    """toks((1, 8), (2, 4)) -> eight 1s followed by four 2s."""
    out: list[int] = []
    for value, count in runs:
        out += [value] * count
    return out


# --------------------------------------------------------------------------
# hits and misses
# --------------------------------------------------------------------------

def test_a_cold_cache_has_no_hits():
    bm = PrefixBlockManager(n_blocks=16, block_size=4)
    a = bm.allocate_with_prefix("a", toks((7, 12)))
    assert a.n_cached_blocks == 0
    assert len(a.table) == 3
    assert conserved(bm)


def test_an_identical_prefix_is_shared_not_recomputed():
    bm = PrefixBlockManager(n_blocks=16, block_size=4)
    prompt = toks((7, 12))
    bm.allocate_with_prefix("a", prompt)
    bm.publish("a", prompt)
    before = bm.n_free

    b = bm.allocate_with_prefix("b", prompt)
    assert b.n_cached_blocks == 3, "identical prompt should hit every full block"
    assert b.n_cached_tokens == 12
    assert bm.n_free == before, "a full hit must consume no new blocks"
    assert b.table == bm.tables["a"]
    assert all(bm.ref_count[x] == 2 for x in b.table)
    assert conserved(bm)


def test_sharing_stops_at_the_first_divergent_block():
    bm = PrefixBlockManager(n_blocks=16, block_size=4)
    a_prompt = toks((7, 8), (1, 4))
    bm.allocate_with_prefix("a", a_prompt)
    bm.publish("a", a_prompt)

    b_prompt = toks((7, 8), (2, 4))          # same first 8, then diverges
    b = bm.allocate_with_prefix("b", b_prompt)
    assert b.n_cached_blocks == 2, "should share the two identical blocks only"
    assert b.table[:2] == bm.tables["a"][:2]
    assert b.table[2] != bm.tables["a"][2], "divergent block must be fresh"
    assert conserved(bm)


def test_identical_block_after_different_prefix_is_not_shared():
    """THE correctness test. Same block contents, different history, different KV."""
    bm = PrefixBlockManager(n_blocks=16, block_size=4)
    a_prompt = toks((1, 4), (9, 4))
    bm.allocate_with_prefix("a", a_prompt)
    bm.publish("a", a_prompt)

    b_prompt = toks((2, 4), (9, 4))          # block 1 identical, block 0 is not
    b = bm.allocate_with_prefix("b", b_prompt)
    assert b.n_cached_blocks == 0, (
        "block 1 holds the same tokens but follows a different prefix, so its "
        "KV differs - sharing it would be silent corruption")
    assert set(b.table).isdisjoint(bm.tables["a"])
    assert conserved(bm)


def test_a_partial_trailing_block_is_never_published():
    bm = PrefixBlockManager(n_blocks=16, block_size=4)
    prompt = toks((7, 10))                    # 2 full blocks + 2 tokens
    bm.allocate_with_prefix("a", prompt)
    bm.publish("a", prompt)
    b = bm.allocate_with_prefix("b", prompt)
    assert b.n_cached_blocks == 2, "only the two FULL blocks are stable enough to share"
    assert b.n_cached_tokens == 8


# --------------------------------------------------------------------------
# the LRU tier - what makes hits survive across requests
# --------------------------------------------------------------------------

def test_a_freed_block_stays_cached_and_still_hits():
    """Without this, prefix caching only helps requests that overlap in time."""
    bm = PrefixBlockManager(n_blocks=16, block_size=4)
    prompt = toks((7, 12))
    bm.allocate_with_prefix("a", prompt)
    bm.publish("a", prompt)
    bm.free("a")
    assert bm.n_free + len(bm.evictable) == 16
    assert len(bm.evictable) == 3, "published blocks should be cached, not discarded"

    b = bm.allocate_with_prefix("b", prompt)
    assert b.n_cached_blocks == 3, "a freed-but-cached prefix should still hit"
    assert conserved(bm)


def test_cached_blocks_are_reclaimed_when_the_pool_runs_dry():
    bm = PrefixBlockManager(n_blocks=4, block_size=4)
    p1 = toks((1, 16))
    bm.allocate_with_prefix("a", p1)
    bm.publish("a", p1)
    bm.free("a")
    assert len(bm.evictable) == 4 and bm.n_free == 0

    p2 = toks((2, 16))                        # nothing in common
    bm.allocate_with_prefix("b", p2)
    assert bm.n_free == 0 and len(bm.evictable) == 0, \
        "cached blocks must be reclaimable rather than causing a spurious OOM"
    assert conserved(bm)


def test_reclaiming_a_cached_block_removes_its_cache_entry():
    """A reclaimed block holds someone else's KV now; a stale entry would be
    served as a hit and silently return the wrong activations."""
    bm = PrefixBlockManager(n_blocks=4, block_size=4)
    p1 = toks((1, 16))
    bm.allocate_with_prefix("a", p1)
    bm.publish("a", p1)
    bm.free("a")
    p2 = toks((2, 16))
    bm.allocate_with_prefix("b", p2)          # reclaims all four of p1's blocks
    bm.free("b")                              # ...and release them again

    c = bm.allocate_with_prefix("c", p1)      # p1's entries should be gone
    assert c.n_cached_blocks == 0, "stale cache entry survived reclamation"


# --------------------------------------------------------------------------
# copy-on-write at the divergence point
# --------------------------------------------------------------------------

def test_appending_into_a_shared_partial_block_copies_first():
    bm = PrefixBlockManager(n_blocks=16, block_size=4)
    prompt = toks((7, 6))                     # 1 full block + 2 tokens
    bm.allocate_with_prefix("a", prompt)
    bm.publish("a", prompt)
    bm.share_tail("a", "b")                   # b takes the partial block too

    shared = bm.tables["a"][-1]
    assert bm.ref_count[shared] == 2
    new = bm.append_cow("b")
    assert new != shared, "wrote into a block another sequence still holds"
    assert bm.tables["a"][-1] == shared, "the other sequence must be untouched"
    assert bm.ref_count[shared] == 1
    assert conserved(bm)


def test_append_does_not_copy_a_block_this_sequence_owns_alone():
    bm = PrefixBlockManager(n_blocks=16, block_size=4)
    prompt = toks((7, 6))
    bm.allocate_with_prefix("a", prompt)
    own = bm.tables["a"][-1]
    assert bm.ref_count[own] == 1
    assert bm.append_cow("a") is None, "copied a block nobody else was holding"
    assert bm.tables["a"][-1] == own
    assert conserved(bm)


def test_hit_rate_is_reported():
    bm = PrefixBlockManager(n_blocks=64, block_size=4)
    doc = toks((5, 16))
    for i in range(4):
        prompt = doc + toks((i + 100, 4))
        bm.allocate_with_prefix(f"q{i}", prompt)
        bm.publish(f"q{i}", prompt)
    s = bm.prefix_stats()
    assert s["blocks_requested"] == 20
    assert s["blocks_hit"] == 12, "4 of 5 blocks shared on each of the last 3"
    assert s["hit_rate"] == pytest.approx(12 / 20)
