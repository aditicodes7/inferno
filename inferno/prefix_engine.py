"""R5: paged serving with prefix caching.

Identical to R4 except at admission. Before prefilling, the allocator is asked
whether any leading FULL blocks of this prompt are already in the cache. On a
hit the blocks are shared - refcount incremented, nothing copied - and prefill
computes only the uncached suffix.

That is where the TTFT saving comes from, and it is real rather than
bookkeeping: those blocks are never recomputed. Their K/V are read by the gather
exactly like any other block, because attention is causal and a shared prefix's
K/V are bit-identical between the sequences that share it.

At least one token is always recomputed even on a total hit: the cache holds
K/V, not logits, and the next token needs a forward pass at the final position.
"""

from __future__ import annotations

import time

import torch

from inferno.block_manager import OutOfBlocks
from inferno.paged_cache import PagedKVCache
from inferno.paged_engine import PagedEngine
from inferno.prefix_cache import PrefixBlockManager
from inferno.scheduler import Decision, Request, Scheduler


class PrefixEngine(PagedEngine):

    @torch.inference_mode()
    def run(self, requests: list[Request], n_blocks: int, block_size: int = 16,
            max_concurrent: int = 64, enable_prefix_cache: bool = True):
        sched = Scheduler(n_slots=max_concurrent)
        for r in requests:
            sched.add(r)
        bm = PrefixBlockManager(n_blocks=n_blocks, block_size=block_size)
        cache = PagedKVCache(
            n_layers=self.cfg.n_layers, n_blocks=n_blocks, block_size=block_size,
            n_kv_heads=self.cfg.n_kv_heads, head_dim=self.cfg.head_dim,
            dtype=self.dtype, device=self.device)

        worst = max(r.n_prompt + r.max_new_tokens for r in requests)
        if bm.n_blocks_for(worst) > n_blocks:
            raise ValueError(
                f"a request needs up to {bm.n_blocks_for(worst)} blocks "
                f"({worst} tokens at block_size {block_size}) but the pool has "
                f"only {n_blocks}")

        t0 = time.perf_counter()
        n_prefill = n_decode = n_preempt = 0
        peak_concurrent = 0
        prefill_tokens_computed = prefill_tokens_total = 0
        ttfts: list[float] = []
        per_request: list[dict] = []

        while not sched.done():
            now = time.perf_counter() - t0
            d: Decision = sched.schedule(now)
            if d.kind == "idle":
                time.sleep(0.001)
                continue
            done_before = len(sched.finished)

            if d.kind == "prefill":
                r = d.prefill
                if bm.can_allocate(r.n_prompt) or bm.n_free + len(bm.evictable) >= \
                        bm.n_blocks_for(r.n_prompt):
                    alloc = bm.allocate_with_prefix(
                        r.id, r.prompt_tokens, use_cache=enable_prefix_cache)

                    # The cache holds K/V, never logits, so the final position
                    # must always be computed to get the next token.
                    start = min(alloc.n_cached_tokens, r.n_prompt - 1)
                    suffix = torch.tensor([r.prompt_tokens[start:]],
                                          device=self.device)
                    t_pf = time.perf_counter()
                    logits = self.model.forward_paged_prefill(
                        suffix, cache, bm.tables[r.id], n_tokens=r.n_prompt,
                        start=start)
                    dt = time.perf_counter() - t_pf
                    ttfts.append(dt)
                    per_request.append({
                        "id": r.id, "n_prompt": r.n_prompt,
                        "cached_tokens": start, "computed_tokens": r.n_prompt - start,
                        "prefill_s": dt, "hit": start > 0})
                    prefill_tokens_computed += r.n_prompt - start
                    prefill_tokens_total += r.n_prompt
                    if enable_prefix_cache:
                        bm.publish(r.id, r.prompt_tokens)
                    sched.on_prefilled(r, int(logits[0, -1].argmax()),
                                       now=time.perf_counter() - t0,
                                       eos_ids=self.eos_ids)
                    n_prefill += 1
                elif sched.running:
                    d = Decision(kind="decode", decode=list(sched.running))
                else:
                    raise OutOfBlocks(f"{r.id} cannot be admitted")

            if d.kind == "decode":
                batch = list(d.decode)
                needed = sum(1 for r in batch
                             if bm.length(r.id) % block_size == 0)
                while needed > bm.n_free + len(bm.evictable) and len(batch) > 1:
                    victim = batch.pop()
                    if bm.length(victim.id) % block_size == 0:
                        needed -= 1
                    bm.free(victim.id)
                    sched.preempt(victim)
                    n_preempt += 1
                if not batch:
                    continue

                for r in batch:
                    copied = bm.append_cow(r.id)
                    if copied is not None:
                        # Copy-on-write: this sequence shared its partial tail
                        # block with another and is about to write into it.
                        _cow_copy(cache, copied, bm)

                lengths = [bm.length(r.id) for r in batch]
                max_blocks = max(len(bm.block_table(r.id)) for r in batch)
                tables = torch.tensor(
                    [bm.block_table(r.id) + [bm.block_table(r.id)[-1]]
                     * (max_blocks - len(bm.block_table(r.id))) for r in batch],
                    device=self.device)
                located = [bm.locate(r.id, bm.length(r.id) - 1) for r in batch]
                blocks = torch.tensor([b for b, _ in located], device=self.device)
                offsets = torch.tensor([o for _, o in located], device=self.device)
                last = torch.tensor([[r.output[-1]] for r in batch], device=self.device)
                lens_t = torch.tensor(lengths, device=self.device)
                logits = self.model.forward_paged_decode(
                    last, cache, tables, blocks, offsets, lens_t,
                    (lens_t - 1)[:, None])
                toks = logits[:, -1].argmax(-1).tolist()
                sched.on_decoded({r.slot: t for r, t in zip(batch, toks)},
                                 eos_ids=self.eos_ids,
                                 now=time.perf_counter() - t0)
                n_decode += 1

            for r in sched.finished[done_before:]:
                if r.id in bm.tables:
                    bm.free(r.id)
            peak_concurrent = max(peak_concurrent, len(sched.running))

        stats = bm.prefix_stats()
        summary = {
            "wall_clock_s": time.perf_counter() - t0,
            "prefill_iterations": n_prefill, "decode_iterations": n_decode,
            "preemptions": n_preempt, "peak_concurrent_sequences": peak_concurrent,
            "n_blocks": n_blocks, "block_size": block_size,
            "prefix_cache_enabled": enable_prefix_cache,
            "prefill_tokens_computed": prefill_tokens_computed,
            "prefill_tokens_total": prefill_tokens_total,
            "prefill_tokens_saved_frac": (
                1 - prefill_tokens_computed / prefill_tokens_total
                if prefill_tokens_total else 0.0),
            "ttft_mean_s": sum(ttfts) / len(ttfts) if ttfts else 0.0,
            "tokens_generated": sum(r.n_generated for r in requests),
            "per_request": per_request,
            **stats,
        }
        return requests, summary


def _cow_copy(cache: PagedKVCache, new_block: int, bm) -> None:
    """Physically duplicate a block the allocator just copy-on-wrote.

    `append_cow` only reassigns ownership; the K/V still live in the block the
    other sequence holds. Without this copy the new block contains whatever was
    there before, and the sequence silently attends to garbage for those
    positions.
    """
    old = bm.cow_source.pop(new_block, None)
    if old is None:
        return
    for layer in range(len(cache.k)):
        cache.k[layer][new_block] = cache.k[layer][old]
        cache.v[layer][new_block] = cache.v[layer][old]
