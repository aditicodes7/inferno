# Inferno

An LLM inference engine written from scratch, to understand how production
serving systems work from the inside. It implements the core ideas behind
vLLM — KV caching, continuous batching, paged attention, prefix caching — each
built only after the previous one produced a measurement and a correctness
proof.

> **This project is not trying to beat vLLM.** It is trying to understand it.
> The honest account of the remaining gap is the point, not the throughput
> numbers.

---

## The idea underneath everything

**Prefill is compute-bound and runs once. Decode is memory-bandwidth-bound and
runs hundreds of times per request.** Nearly every design decision in a serving
system follows from that asymmetry, and every rung below is a different
consequence of it.

---

## Results

Six rungs, each accepted only on a number *and* a proof. **The workloads are not
identical across rungs** — each rung is measured against the design it replaces,
under the workload that exposes what it changed — so read down the "measured
against" column rather than comparing throughput figures across rows.

| Rung | What changed | Headline result | Measured against | Proof |
|---|---|---|---|---|
| **R0** | HuggingFace `generate()` | 36.29 tok/s, TTFT p50 54.1 ms | — (the floor) | reruns within 2.8%; token-identical on repeat *and* reversed order |
| **R1** | Hand-written forward pass + KV cache | 29.21 tok/s (+9.1% vs HF eager, −19.5% vs HF sdpa) | HuggingFace, same attention backend | 50/50 token-identical |
| **R2** | Static batching, left-padded | **~18% of decode steps wasted**; worst request blocked 120 steps | R1 | fp32 exact at batch 1/2/4/8/16 |
| **R3** | Continuous batching | **5.3× arrival rate** at p95 TTFT < 500 ms (0.3 → 1.6 req/s) | static batching, same Poisson arrivals | identical across 3 arrival patterns × 3 slot counts |
| **R4** | Paged KV cache | **8× concurrent sequences** at fixed 48 MB (7 → 56); utilisation 15.8% → 86.0% | contiguous reservation, same memory | parity at block size 4/16/64; zero blocks leaked |
| **R5** | Prefix caching | **4.89× TTFT on a cache hit** (1730.8 → 354.2 ms); 85.3% of prefill never computed | the same engine with the cache off | parity including share-then-diverge |

**Hardware: MacBook, Apple Silicon, MPS backend, float16, Qwen2.5-0.5B-Instruct.
These are not GPU numbers.** See [What's honest about these numbers](#whats-honest-about-these-numbers).

📊 [Dashboard](https://claude.ai/artifact/RpzegYya34hiJMEjD7XWiX) ·
📓 [Full project log](PROJECT_LOG.md) ·
🐛 [Bug log](docs/bugs.md) ·
⚖️ [Decisions](docs/decisions.md)

---

## The correctness spine

Every rung from R1 onward is accepted on the same assertion: **token-identical
output under greedy decoding.** Not similar — identical. That single test is
what makes the performance numbers mean anything.

It is a two-tier criterion, and the reason is measured rather than assumed:

- **float32 asserts exact equality.** Batched output must be bit-identical to
  unbatched, at every batch size. This is the gate on the code.
- **float16 asserts only that nothing becomes NaN.** Exact parity is
  unachievable there, and *not because anything is broken*: differently shaped
  matmuls reduce in different orders, and greedy decoding amplifies a last-bit
  difference into a different paragraph. One prompt diverges at token 67 purely
  because its batch had four rows instead of two — identical padding, identical
  code path.

```bash
./run_tests.sh      # 63 tests, 8 files, one process per file
```

---

## Architecture

```
inferno/
  model.py           hand-written Qwen2 forward pass - embedding, 24 decoder
                     blocks, RoPE, GQA, masks, tied output projection
  cache.py           KVCache (R1/R2, one shared length) and SlotKVCache (R3,
                     independent per-slot lengths)
  engine.py          weight loading, chat template, greedy decode loop
  batch_engine.py    R2 - left-padded static batching
  scheduler.py       R3 - waiting/running/finished, FCFS, owns no tensors
  continuous_engine.py  R3 - iteration-level scheduling
  block_manager.py   R4 - page table, free list, reference counting
  paged_cache.py     R4 - physical blocks and the gather
  paged_engine.py    R4 - admission, preemption, recompute
  prefix_cache.py    R5 - prefix keys, LRU cached tier, copy-on-write
  prefix_engine.py   R5 - prefill skips cached blocks entirely
```

`scheduler.py`, `block_manager.py` and `prefix_cache.py` deliberately own **no
tensors**. Every failure mode in those three is silent — a slot reused one
iteration early is corruption, a leaked block is a hang, a starved request
raises nothing — and keeping them tensor-free means all of it is unit-testable
in 0.05 s without a model. That separation caught two bugs before a GPU was
ever involved.

---

## Running it

```bash
python3.11 -m venv .venv
./.venv/bin/pip install torch transformers pytest accelerate

./run_tests.sh                                    # correctness
./.venv/bin/python bench/run_baseline.py          # R0
./.venv/bin/python bench/run_inferno.py           # R1
./.venv/bin/python bench/run_batched.py           # R2
./.venv/bin/python bench/run_continuous.py        # R3
./.venv/bin/python bench/run_paged.py             # R4
./.venv/bin/python bench/run_prefix.py            # R5
```

Every script takes `--device cuda` and writes to `results/gpu/` automatically.

**Use `./run_tests.sh`, not `pytest tests/`.** Each test file holds its engine
in a session-scoped fixture, so one pytest session keeps roughly five model
copies resident and swaps itself to a standstill — 25 minutes at 48 MB RSS and
11% CPU, observed. Separate processes free each engine before the next loads.

---

## What's honest about these numbers

- **No GPU run has happened.** `results/gpu/` is empty. Every figure above is
  Mac/MPS.
- **Three findings are suspect until re-measured on real hardware:** throughput
  peaks at *exactly* batch 16 and falls back at 20 (reproducible across three
  runs, unexplained); static batching buys only ~8% from batch 1 to 8, against
  a theory that predicts much more; and paged attention came out ~8% *faster*
  than contiguous at equal concurrency, which contradicts the prediction.
- **The gather is unmeasured against a fused kernel.** Both the paged and
  contiguous caches materialise a contiguous tensor, so every comparison here
  measures one gather against another. vLLM's PagedAttention walks the block
  table inside the kernel and never materialises at all. This is the largest
  expected line item in the gap analysis and it is currently unquantified.
- **Inferno's prefill is 1.8–4.6× slower than HuggingFace's eager path.** Four
  candidate causes, none profiled.

---

## Three bugs worth reading

The [bug log](docs/bugs.md) has ten with full mechanisms. These three are the
ones that taught the most:

**B3 — "greedy" is not a complete specification of a decoding procedure.**
R1 parity failed 49/50. Every suspicion pointed at the hand-written attention,
the mask, RoPE, the cache. The forward pass was correct from the first run:
teacher-forced agreement was 371/371. The checkpoint ships
`generation_config.json` with `repetition_penalty: 1.1`, and HuggingFace applies
it **during greedy decoding** — it is not gated on `do_sample`. A baseline is
only reproducible if the full logits-processor stack is recorded, not just the
sampling mode.

**B4 — masking does not protect a real query from a NaN.**
The additive attention mask used `torch.finfo(dtype).min`. float16's most
negative finite value is −65504, so adding any score past about −16 rounds off
the end of the range to `-inf`; scores reach ±225 by layer 8. A fully-masked
row — which exists only because of left padding — becomes all `-inf`, and
softmax gives NaN. It then reached *real* tokens, because pad K/V live in the
same cache and a masked weight is exactly 0 — but **0 × NaN = NaN**.

**B7 — two individually correct rules composing into a livelock.**
`preempt()` returned an evicted request to the front of the queue so it would
not starve. FCFS admission only ever considers the head. Together: the victim
became the head, was immediately re-admitted, and immediately forced the next
head to evict it again — forever, with no error and no output. It is *not*
starvation; both requests were repeatedly **admitted**. They simply never held
memory long enough to emit a token. Reviewing either rule alone would never
have found it.

---

## What is not done

- GPU runs for every rung
- vLLM installed and run on identical hardware and workload
- **The gap analysis** — the most important deliverable, and it depends on both
  of the above

The expected finding is already visible from two independent directions: at R4,
paging beat contiguous at equal concurrency because *both* gather; at R5, a
cache hit computes 8.7% of a prompt's tokens but still takes 23% of a cold
prefill's time. The gather does not shrink. Quantifying that against a fused
kernel is what remains.
