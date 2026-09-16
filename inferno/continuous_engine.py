"""R3: the loop that drives the scheduler.

One iteration = one forward pass = one scheduling decision. Compare R2, where
one decision covered up to 128 forward passes and a request that finished early
sat in the batch until the slowest one caught up.

The engine owns tensors and the scheduler owns state; the only thing crossing
between them is a slot index. That boundary is what makes the scheduler
testable without a model (`tests/test_scheduler.py`).
"""

from __future__ import annotations

import time

import torch

from inferno.cache import SlotKVCache
from inferno.engine import InfernoEngine
from inferno.scheduler import Decision, Request, Scheduler


class ContinuousEngine(InfernoEngine):

    def make_requests(self, prompts: list[str], arrivals: list[float] | None = None,
                      max_new_tokens: int = 128, ids: list[str] | None = None
                      ) -> list[Request]:
        arrivals = arrivals if arrivals is not None else [0.0] * len(prompts)
        ids = ids if ids is not None else [str(i) for i in range(len(prompts))]
        return [Request(id=rid, prompt_tokens=self.encode(p)[0].tolist(),
                        arrival=a, max_new_tokens=max_new_tokens)
                for rid, p, a in zip(ids, prompts, arrivals)]

    @torch.inference_mode()
    def run(self, requests: list[Request], n_slots: int = 8,
            max_len: int | None = None, collect_trace: bool = False):
        """Serve every request. Returns them in the order given."""
        if max_len is None:
            max_len = max(r.n_prompt + r.max_new_tokens for r in requests)

        sched = Scheduler(n_slots=n_slots)
        for r in requests:
            sched.add(r)

        cache = SlotKVCache(
            n_layers=self.cfg.n_layers, n_kv_heads=self.cfg.n_kv_heads,
            head_dim=self.cfg.head_dim, n_slots=n_slots, max_len=max_len,
            dtype=self.dtype, device=self.device,
        )

        trace, t0 = [], time.perf_counter()
        n_prefill = n_decode = 0

        while not sched.done():
            now = time.perf_counter() - t0
            d: Decision = sched.schedule(now)

            if d.kind == "idle":
                # Nothing runnable: every remaining request is in the future.
                time.sleep(0.001)
                continue

            done_before = len(sched.finished)

            if d.kind == "prefill":
                r = d.prefill
                cache.reset(r.slot)
                ids = torch.tensor([r.prompt_tokens], device=self.device)
                logits = self.model.forward_prefill_slot(ids, cache, r.slot)
                cache.lengths[r.slot] = r.n_prompt
                slot_of = {r.id: r.slot}
                sched.on_prefilled(r, int(logits[0, -1].argmax()),
                                   now=time.perf_counter() - t0,
                                   eos_ids=self.eos_ids)
                n_prefill += 1
            else:
                batch = d.decode
                slots = torch.tensor([r.slot for r in batch], device=self.device)
                last = torch.tensor([[r.output[-1]] for r in batch],
                                    device=self.device)
                lens = cache.lengths[slots]
                read_len = int(lens.max()) + 1
                # Slot i holds real keys at 0..lens[i] inclusive once this
                # token is written; everything past that is another slot's
                # tail and must not be visible.
                key_mask = (torch.arange(read_len, device=self.device)[None, :]
                            <= lens[:, None]).long()
                logits = self.model.forward_slots(
                    last, cache, slots, lens[:, None], key_mask, read_len)
                cache.advance(slots)
                toks = logits[:, -1].argmax(-1).tolist()
                slot_of = {r.id: r.slot for r in batch}
                sched.on_decoded({r.slot: t for r, t in zip(batch, toks)},
                                 eos_ids=self.eos_ids,
                                 now=time.perf_counter() - t0)
                n_decode += 1

            # Release the cache slot of anything that finished THIS iteration -
            # after the forward pass that read it, never before.
            for r in sched.finished[done_before:]:
                cache.reset(slot_of[r.id])

            if collect_trace:
                st = sched.stats()
                st.update(kind=d.kind, t=time.perf_counter() - t0,
                          kv_utilisation=cache.utilisation())
                trace.append(st)

        summary = {
            "wall_clock_s": time.perf_counter() - t0,
            "prefill_iterations": n_prefill,
            "decode_iterations": n_decode,
            "n_slots": n_slots,
            "slot_max_len": max_len,
            "kv_bytes_allocated": cache.bytes_allocated(),
            "tokens_generated": sum(r.n_generated for r in requests),
        }
        return (requests, summary, trace) if collect_trace else (requests, summary)
