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

**The headline, on a Tesla T4:**

| | HuggingFace (sdpa) | **Inferno** | vLLM 0.29 |
|---|---|---|---|
| single-stream decode | 31.76 tok/s | **38.88 tok/s** (+22.4%) | 174.52 tok/s (**4.49×**) |
| batched, 50 prompts | — | **643.82 tok/s** (bs=32) | 2547.38 tok/s (**3.96×**) |
| TTFT p50 | 36.6 ms | **29.1 ms** | 20.7 ms |

Inferno beats HuggingFace by 22% and loses to vLLM by ~4×. **[Why, in detail →
docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md)** — short version: the gap is *not*
the paged-attention gather, which was the hypothesis held throughout this
project. R1, which has no block table and no gather at all, is already 4.49×
behind. The gap is CUDA graph capture, `torch.compile` and fused kernels —
things neither Inferno nor HuggingFace does, which is why those two sit within
22% of each other while vLLM is 4.5× above both.

**And the techniques themselves work.** On the same shared-document workload,
Inferno's prefix cache gives **+20.1% throughput at an 83.8% hit rate**, against
vLLM's **+19.3% at 86.1%**. The ideas transfer; the execution layer does not.

### The rungs

Each accepted only on a number *and* a proof. **The workloads are not identical
across rungs** — each is measured against the design it replaces, under the
workload that exposes what it changed — so read the "measured against" column
rather than comparing throughput across rows.

| Rung | What changed | Result (Tesla T4) | Measured against | Proof |
|---|---|---|---|---|
| **R0** | HuggingFace `generate()` | 31.76 tok/s, TTFT p50 36.6 ms | — (the floor) | reruns within 2.8%; token-identical on repeat *and* reversed order |
| **R1** | Hand-written forward pass + KV cache | 38.88 tok/s (**+22.4%** vs HF) | HuggingFace, same attention backend | 50/50 token-identical |
| **R2** | Static batching, left-padded | **17.9% of decode steps wasted**; worst request blocked 120 steps | R1 | fp32 exact at batch 1/2/4/8/16 |
| **R3** | Continuous batching | **16× arrival rate** at p95 TTFT < 500 ms (0.25 → 4.0 req/s) | static batching, same Poisson arrivals | identical across 3 arrival patterns × 3 slot counts |
| **R4** | Paged KV cache | **9.1× concurrent sequences** at fixed 48 MB (7 → 64); utilisation 17.3% → 97.2% | contiguous reservation, same memory | parity at block size 4/16/64; zero blocks leaked across 24–35 preemptions |
| **R5** | Prefix caching | **83.8% hit rate**, 85.3% of prefill never computed, +20.1% throughput | the same engine with the cache off | parity including share-then-diverge |

All 68 tests pass on **both** Apple Silicon (MPS) and CUDA.

📊 [Dashboard](https://claude.ai/artifact/RpzegYya34hiJMEjD7XWiX) ·
📉 [Gap analysis](docs/GAP_ANALYSIS.md) ·
📓 [Full project log](PROJECT_LOG.md) ·
🐛 [Bug log](docs/bugs.md) ·
⚖️ [Decisions](docs/decisions.md)

### Four conclusions that moving to GPU destroyed

Every rung was first measured on Apple Silicon. Four findings did not survive
CUDA, and that is the most useful result in the project:

| Concluded on MPS | On CUDA |
|---|---|
| Throughput peaks at *exactly* batch 16 and falls back at 20 — reproducible three times | **Dead.** Monotonic 1→32. An MPS kernel artifact. |
| "The textbook batching argument does not hold here" — batching bought ~8% | **Dead.** 5.5× from batch 1→8. It holds; MPS was the problem. |
| Block size 16 is clearly best | **Dead.** Flat across 4/16/64 on CUDA, and 16 is the *slowest*. |
| Prefix caching cuts TTFT 4.89× | **Overstated.** 1.44× — prefill costs a T4 only ~42 ms, so there is little to save. |

All four were measured carefully, reproduced, and written down with confidence.
They were properties of one platform mistaken for properties of the technique,
and no amount of care on that platform could have caught it.

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

- **One GPU, one model, one session.** Tesla T4, 0.5B parameters. A 7B model on
  an A100 would move every ratio here.
- **The 4× gap is named but not decomposed.** The gap analysis identifies four
  causes and ranks them by argument. No profiler was run, and that ranking is
  reasoned rather than measured.
- **Token agreement with vLLM is 42/50, not 50/50.** Expected — different kernels
  reduce in different orders and greedy decoding amplifies the last bit. But it
  means a small part of the timing difference could be different work rather
  than the same work done faster. Unquantified.
- **vLLM was not tuned.** Defaults, no `--max-num-seqs` tuning, no quantization.
  A tuned vLLM would be faster still.
- **torch version matters more than expected.** vLLM's install downgraded torch
  2.10 → 2.13, and Inferno lost 7.6% on that change alone. Every comparison
  figure here was re-measured afterwards; the earlier ones would have overstated
  Inferno.

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

- **The 4× is not decomposed.** Profiling with Nsight or `torch.profiler` would
  apportion it between graph capture, compilation and the attention kernel.
  Currently those are ranked by argument, not measurement.
- **Chunked prefill.** R3 gives prefill its own iteration, which stalls every
  running request. vLLM mixes prefill into the decode batch.
- **The obvious optimisations the analysis points at** — CUDA graph capture of
  the decode step, `torch.compile` on the decoder block, a fused attention
  kernel. The gap analysis argues these are where the 4× lives; none is
  implemented.

None of these are gaps in the deliverable. They are what a version 2 would do,
and the analysis says which ones would actually pay.
