"""R4 allocator. No tensors, no model - pure bookkeeping.

An allocator bug leaves output perfectly correct right up until the moment it
does not, so this is tested separately from parity. The three failure modes
predicted for this rung are all silent:

  - a block leaked on early termination or preemption looks like premature OOM,
    arriving much later and nowhere near the cause;
  - a refcount off by one either frees a block a live sequence still reads
    (silent corruption) or never frees it at all (a leak);
  - a double free puts one physical block on the free list twice, so two
    sequences are later handed the same memory and quietly overwrite each
    other.

The invariant that catches most of this is conservation: free + referenced must
always equal capacity. It is asserted after every operation here.
"""

from __future__ import annotations

import pytest

from inferno.block_manager import BlockManager, OutOfBlocks


def conserved(bm: BlockManager) -> bool:
    """No block is lost, and none is on the free list while still referenced."""
    referenced = {b for t in bm.tables.values() for b in t}
    free = set(bm.free_blocks)
    return (len(bm.free_blocks) == len(free)                  # no duplicates
            and not (referenced & free)                       # no use-after-free
            and len(free) + len(referenced) == bm.n_blocks)


# --------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------

def test_allocation_rounds_up_to_whole_blocks():
    bm = BlockManager(n_blocks=8, block_size=16)
    bm.allocate("a", n_tokens=17)
    assert len(bm.tables["a"]) == 2, "17 tokens needs 2 blocks of 16"
    assert bm.n_free == 6
    assert conserved(bm)


def test_a_full_block_boundary_does_not_over_allocate():
    bm = BlockManager(n_blocks=8, block_size=16)
    bm.allocate("a", n_tokens=32)
    assert len(bm.tables["a"]) == 2, "32 tokens is exactly 2 blocks, not 3"
    assert conserved(bm)


def test_internal_waste_is_bounded_by_block_size():
    """The entire point of paging: waste per sequence < block_size, not max_len."""
    bm = BlockManager(n_blocks=64, block_size=16)
    bm.allocate("tiny", n_tokens=1)
    reserved = len(bm.tables["tiny"]) * bm.block_size
    assert reserved - 1 < bm.block_size


def test_append_allocates_a_new_block_only_when_the_current_one_fills():
    bm = BlockManager(n_blocks=8, block_size=4)
    bm.allocate("a", n_tokens=4)
    assert len(bm.tables["a"]) == 1
    for _ in range(3):
        bm.append("a")
    assert len(bm.tables["a"]) == 2, "should still be inside the second block"
    bm.append("a")
    assert len(bm.tables["a"]) == 2
    bm.append("a")
    assert len(bm.tables["a"]) == 3
    assert conserved(bm)


def test_out_of_blocks_raises_rather_than_corrupting():
    bm = BlockManager(n_blocks=2, block_size=4)
    bm.allocate("a", n_tokens=8)
    assert bm.n_free == 0
    with pytest.raises(OutOfBlocks):
        bm.allocate("b", n_tokens=1)
    assert conserved(bm), "a failed allocation must leave the pool untouched"


def test_a_partially_satisfiable_allocation_leaves_nothing_behind():
    """Asking for 3 blocks with 2 free must not consume the 2."""
    bm = BlockManager(n_blocks=2, block_size=4)
    with pytest.raises(OutOfBlocks):
        bm.allocate("big", n_tokens=12)
    assert bm.n_free == 2
    assert "big" not in bm.tables
    assert conserved(bm)


# --------------------------------------------------------------------------
# free / reallocate - leaks and double frees
# --------------------------------------------------------------------------

def test_free_returns_every_block():
    bm = BlockManager(n_blocks=8, block_size=4)
    bm.allocate("a", n_tokens=16)
    assert bm.n_free == 4
    bm.free("a")
    assert bm.n_free == 8, "blocks leaked on free"
    assert conserved(bm)


def test_allocate_free_reallocate_many_times_leaks_nothing():
    bm = BlockManager(n_blocks=16, block_size=8)
    for i in range(200):
        sid = f"s{i}"
        bm.allocate(sid, n_tokens=(i % 5) * 8 + 1)
        for _ in range(i % 11):
            bm.append(sid)
        bm.free(sid)
        assert conserved(bm), f"conservation broken at iteration {i}"
    assert bm.n_free == 16, "blocks leaked across allocate/free cycles"


def test_double_free_is_refused():
    bm = BlockManager(n_blocks=4, block_size=4)
    bm.allocate("a", n_tokens=4)
    bm.free("a")
    with pytest.raises(KeyError):
        bm.free("a")
    assert bm.n_free == 4, "double free put a block on the free list twice"
    assert conserved(bm)


def test_freeing_mid_generation_returns_everything():
    """Early termination is where blocks leak."""
    bm = BlockManager(n_blocks=32, block_size=4)
    bm.allocate("a", n_tokens=10)
    for _ in range(7):
        bm.append("a")
    bm.free("a")
    assert bm.n_free == 32
    assert conserved(bm)


# --------------------------------------------------------------------------
# reference counting - the R5 groundwork, tested now
# --------------------------------------------------------------------------

def test_shared_blocks_are_not_freed_while_another_sequence_holds_them():
    bm = BlockManager(n_blocks=8, block_size=4)
    bm.allocate("parent", n_tokens=8)
    before = bm.n_free
    bm.share("parent", "child")
    assert bm.n_free == before, "sharing must not consume new blocks"
    assert bm.tables["child"] == bm.tables["parent"]
    assert all(bm.ref_count[b] == 2 for b in bm.tables["parent"])

    bm.free("parent")
    assert bm.n_free == before, "a block still referenced by child was freed"
    assert all(bm.ref_count[b] == 1 for b in bm.tables["child"])

    bm.free("child")
    assert bm.n_free == 8
    assert conserved(bm)


def test_refcount_reaches_zero_exactly_once_under_repeated_sharing():
    bm = BlockManager(n_blocks=8, block_size=4)
    bm.allocate("a", n_tokens=4)
    for i in range(5):
        bm.share("a", f"c{i}")
    blk = bm.tables["a"][0]
    assert bm.ref_count[blk] == 6
    bm.free("a")
    for i in range(5):
        assert bm.ref_count[blk] == 5 - i
        bm.free(f"c{i}")
    assert bm.ref_count[blk] == 0
    assert bm.n_free == 8
    assert conserved(bm)


# --------------------------------------------------------------------------
# the numbers R4 exists to produce
# --------------------------------------------------------------------------

def test_utilisation_reports_real_occupancy():
    bm = BlockManager(n_blocks=10, block_size=10)
    bm.allocate("a", n_tokens=5)
    assert bm.utilisation() == pytest.approx(5 / 100)
    assert bm.block_utilisation() == pytest.approx(0.5)


def test_position_of_a_token_maps_through_the_block_table():
    bm = BlockManager(n_blocks=8, block_size=4)
    bm.allocate("a", n_tokens=10)
    table = bm.tables["a"]
    assert bm.locate("a", 0) == (table[0], 0)
    assert bm.locate("a", 3) == (table[0], 3)
    assert bm.locate("a", 4) == (table[1], 0)
    assert bm.locate("a", 9) == (table[2], 1)
    with pytest.raises(IndexError):
        bm.locate("a", 10)
