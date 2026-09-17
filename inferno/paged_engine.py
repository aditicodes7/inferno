"""R4: continuous batching on a paged KV cache, with preemption.

R3 admitted a request into a slot that reserved `max_len` up front, so max
concurrency was `memory / max_len` decided in advance. Here a request is
admitted with blocks for its prompt only and grows one block at a time, so
concurrency is bounded by what sequences ACTUALLY use.

The price is that memory can run out mid-generation, which cannot happen when
everything is reserved in advance. The answer is preemption: evict a running
request, free its blocks, put it back at the front of the queue and recompute
it later. Its generated tokens are discarded - real wasted work, counted in the
summary so it stays visible.

POLICY: PREEMPTION ONLY EVER SERVES A RUNNING SEQUENCE THAT CANNOT GROW.
A waiting request that does not fit simply waits. Never evicting on behalf of
an admission is not a tuning choice - it is what makes progress guaranteed.
Allowing it produced a livelock (docs/bugs.md, 2026-09-16): `preempt()` returns
the victim to the FRONT of the queue so it is not starved, and FCFS admission
only considers the HEAD, so the victim was immediately re-admitted and
immediately re-evicted. Two individually correct rules, composing into a cycle
with zero forward progress and no error. Only running sequences can trigger
eviction now, and a running sequence that grows is making progress by
definition, so the cycle cannot form.

Victim selection is LIFO: the most recently admitted running request goes
first. It has the least work to lose, and evicting the oldest would repeatedly
punish the request that has already waited longest.
"""

from __future__ import annotations

import time

import torch

from inferno.block_manager import BlockManager, OutOfBlocks
from inferno.engine import InfernoEngine
from inferno.paged_cache import PagedKVCache
from inferno.scheduler import Decision, Request, Scheduler


class PagedEngine(InfernoEngine):

    def make_requests(self, prompts, arrivals=None, max_new_tokens=128, ids=None):
        arrivals = arrivals if arrivals is not None else [0.0] * len(prompts)
        ids = ids if ids is not None else [str(i) for i in range(len(prompts))]
        return [Request(id=rid, prompt_tokens=self.encode(p)[0].tolist(),
                        arrival=a, max_new_tokens=max_new_tokens)
                for rid, p, a in zip(ids, prompts, arrivals)]

    @staticmethod
    def _victim(sched: Scheduler, exclude: Request | None) -> Request | None:
        for r in reversed(sched.running):          # LIFO
            if r is not exclude:
                return r
        return None

    @torch.inference_mode()
    def run(self, requests: list[Request], n_blocks: int, block_size: int = 16,
            max_concurrent: int = 64):
        sched = Scheduler(n_slots=max_concurrent)
        for r in requests:
            sched.add(r)
        bm = BlockManager(n_blocks=n_blocks, block_size=block_size)
        cache = PagedKVCache(
            n_layers=self.cfg.n_layers, n_blocks=n_blocks, block_size=block_size,
            n_kv_heads=self.cfg.n_kv_heads, head_dim=self.cfg.head_dim,
            dtype=self.dtype, device=self.device)

        # Feasibility check, NOT a reservation. Paging allocates on demand, but
        # a sequence whose WORST CASE cannot fit in the entire pool is
        # unservable: it will be admitted, grow, find nothing left to evict,
        # and fail partway through. Catching that here turns a confusing
        # mid-run OutOfBlocks into a clear message before any work is done.
        worst = max(r.n_prompt + r.max_new_tokens for r in requests)
        if bm.n_blocks_for(worst) > n_blocks:
            raise ValueError(
                f"a request needs up to {bm.n_blocks_for(worst)} blocks "
                f"({worst} tokens at block_size {block_size}) but the pool has "
                f"only {n_blocks}. Paging allocates on demand, but the pool "
                f"must still be able to hold one sequence's worst case.")

        t0 = time.perf_counter()
        n_prefill = n_decode = n_preempt = 0
        peak_concurrent = 0
        util_samples: list[float] = []

        while not sched.done():
            now = time.perf_counter() - t0
            d: Decision = sched.schedule(now)

            if d.kind == "idle":
                time.sleep(0.001)
                continue

            done_before = len(sched.finished)

            if d.kind == "prefill":
                r = d.prefill
                if bm.can_allocate(r.n_prompt):
                    table = bm.allocate(r.id, r.n_prompt)
                    ids = torch.tensor([r.prompt_tokens], device=self.device)
                    logits = self.model.forward_paged_prefill(ids, cache, table)
                    sched.on_prefilled(r, int(logits[0, -1].argmax()),
                                       now=time.perf_counter() - t0,
                                       eos_ids=self.eos_ids)
                    n_prefill += 1
                elif sched.running:
                    # Decline the admission and let the running set drain a
                    # step. We never evict on behalf of a waiting request.
                    d = Decision(kind="decode", decode=list(sched.running))
                else:
                    raise OutOfBlocks(
                        f"{r.id} needs {bm.n_blocks_for(r.n_prompt)} blocks, "
                        f"{bm.n_free} free, and nothing is running to drain")

            if d.kind == "decode":
                batch = list(d.decode)
                # Make room for one token per sequence BEFORE writing any of
                # them, so a mid-batch failure cannot leave the cache half
                # updated. `sched.running` is in admission order, so the tail
                # is the most recently admitted - the LIFO victim.
                needed = sum(1 for r in batch
                             if bm.length(r.id) % block_size == 0)
                while needed > bm.n_free and len(batch) > 1:
                    victim = batch.pop()
                    if bm.length(victim.id) % block_size == 0:
                        needed -= 1
                    bm.free(victim.id)
                    sched.preempt(victim)
                    n_preempt += 1
                if needed > bm.n_free:
                    raise OutOfBlocks(
                        f"{batch[0].id} cannot grow: it holds the whole "
                        f"{n_blocks}-block pool and there is nothing left to evict")

                for r in batch:
                    bm.append(r.id)
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
            util_samples.append(bm.block_utilisation())

        leaked = bm.n_blocks - bm.n_free
        summary = {
            "wall_clock_s": time.perf_counter() - t0,
            "prefill_iterations": n_prefill, "decode_iterations": n_decode,
            "preemptions": n_preempt,
            "peak_concurrent_sequences": peak_concurrent,
            "n_blocks": n_blocks, "block_size": block_size,
            "kv_bytes_allocated": cache.bytes_allocated(),
            "block_utilisation_mean": (sum(util_samples) / len(util_samples)
                                       if util_samples else 0.0),
            "tokens_generated": sum(r.n_generated for r in requests),
            "blocks_leaked": leaked,
        }
        if leaked:
            raise RuntimeError(f"{leaked} blocks leaked after all requests finished")
        return requests, summary
