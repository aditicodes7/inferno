# Gap analysis: Inferno vs vLLM

**Hardware:** Tesla T4 (compute capability 7.5), Kaggle. float16,
`Qwen2.5-0.5B-Instruct`, greedy decoding, `repetition_penalty=1.0`, the same 50
fixed prompts throughout. All figures below are from a single session on
2026-09-17; raw data in `results/gpu/SUMMARY.json`.

---

## 1. The number

| | HuggingFace (sdpa) | **Inferno** | vLLM 0.29 |
|---|---|---|---|
| single-stream decode | 31.76 tok/s | **38.88 tok/s** | 174.52 tok/s |
| batched throughput, 50 prompts | — | **643.82 tok/s** (bs=32) | 2547.38 tok/s |
| TTFT p50, single stream | 36.6 ms | **29.1 ms** | 20.7 ms |

**Inferno beats HuggingFace by 22%. vLLM beats Inferno by 4.5× single-stream and
4.0× batched.**

The two regimes agreeing to within 12% matters: they are different workloads,
different batch shapes and different bottlenecks, and they produce essentially
the same ratio. Whatever causes the gap is not specific to one operating point.

**Controlling for library version.** vLLM's install downgraded torch from
2.10.0+cu128 to 2.13.0+cu130. Every Inferno figure above was re-measured under
2.13 after that. It was not a cosmetic difference — Inferno's single-stream
decode fell from 42.09 to 38.88 tok/s (−7.6%) on torch 2.13 alone. Comparing the
pre-install numbers against vLLM would have overstated Inferno by that margin.

---

## 2. Where the gap is *not*: the gather

**Throughout this project the working hypothesis was that the dominant cost was
the gather** — Inferno materialises a contiguous K/V tensor from the block table
on every layer of every decode step, where vLLM's PagedAttention kernel walks the
block table inside the kernel and never materialises at all. It was logged as the
largest expected line item (Q9) and stated twice in `PROJECT_LOG.md`.

**The data says that hypothesis is wrong.**

R1 is the plain contiguous-cache engine. It has **no block table and no gather** —
`cache.write()` returns a tensor slice, which is a view, not a copy. R1 is
**4.49× behind vLLM**. The gap is fully present before paging is introduced at
all, so paging cannot be what causes it.

Two further measurements point the same way:

- At equal concurrency (512 MB, 64 sequences) the **paged** engine ran at
  854.41 tok/s against the **contiguous** engine's 836.80 — paging was 2%
  *faster*, not slower. The gather costs essentially nothing at this scale.
- A prefix-cache hit computes 8.5% of a prompt's tokens but takes 70.0% of a cold
  prefill's time on GPU (29.0 ms vs 41.7 ms), which does show the gather not
  shrinking with the cache — but that is a 1.4× effect inside prefill, not a 4.5×
  effect on the whole engine.

The honest conclusion is that a hypothesis held for the length of the project
survived until it met hardware that could test it, and then failed.

---

## 3. Where the gap actually is

Inferno (38.88) and HuggingFace (31.76) are within 22% of each other. vLLM is
4.5× *both*. That grouping is the clue: the difference is not something Inferno
does badly relative to a reference implementation — it is something **neither
Inferno nor HuggingFace does at all.**

From vLLM's own startup log, the things it does that neither does:

**CUDA graph capture.** `Capturing CUDA graphs (PIECEWISE): 51/51` and
`(FULL): 35/35`. A decode step in Inferno launches roughly 24 layers × ~15
operations ≈ **360 kernel launches, dispatched from Python, every single step**.
vLLM replays one captured graph. At 0.5B parameters the actual arithmetic per
step is trivial, so launch overhead is not a tax on the work — it very plausibly
*is* most of the work. This is the leading candidate.

**Ahead-of-time compilation.** `torch.compile took 17.47 s`, inductor backend,
with fusion of norms and elementwise chains. Inferno runs eager PyTorch: every
RMSNorm, every SiLU, every residual add is a separate kernel reading and writing
HBM.

**A fused attention kernel.** The log shows `TRITON_ATTN` selected (FlashAttention-2
requires compute capability ≥ 8.0, so even vLLM falls back on a T4). Attention is
one kernel rather than matmul → mask-add → softmax → matmul.

**Continuous batching with chunked prefill**, which Inferno has at R3 but without
the chunked-prefill half (Q7).

**What this analysis cannot do is apportion the 4.5× between those four.** That
needs a profiler — Nsight Systems or `torch.profiler` — and a decomposition run
that has not been done. Ranking them by argument: graph capture first, because
the model is small enough that launch overhead should dominate; compilation
second; the attention kernel third. That ordering is reasoned, not measured, and
should be labelled as such.

---

## 4. What Inferno got right

The algorithms are correct and produce the expected effects at the expected
magnitudes. Compared against vLLM doing the same thing on the same workload:

| technique | vLLM | Inferno |
|---|---|---|
| prefix caching, throughput gain | +19.3% (2547 → 3038 tok/s) | **+20.1%** (282.5 → 339.3 tok/s) |
| prefix cache hit rate, shared-document workload | 86.1% (reported by vLLM) | **83.8%** |

**Inferno's prefix cache is as effective as vLLM's.** Same workload, same
document, hit rates within 3 points and throughput gains within 1 point. The
mechanism — hashing the whole token prefix per block, sharing by reference count,
copy-on-write at divergence — is implemented correctly.

The same holds for the other two techniques, measured against the design each
replaces rather than against vLLM:

- **Continuous batching:** static batching sustains 0.25 req/s under a 500 ms p95
  TTFT budget; continuous sustains 4.0 req/s. **16× the arrival rate.** Static's
  p95 TTFT at 1 req/s is 1.04 s against continuous's 0.046 s.
- **Paged attention:** at a fixed 48 MB KV budget, contiguous reservation fits 7
  concurrent sequences at 17.3% utilisation; paging fits **64 at 97.2%**. 9.1×
  concurrency, 2.3× throughput, zero blocks leaked across 24–35 preemptions.

So: **the ideas transfer, the execution layer does not.** The 4× is not in the
scheduling, the memory management or the cache design. It is underneath all of
them, in how the arithmetic reaches the GPU.

---

## 5. What moving to GPU overturned

Four conclusions drawn on Apple Silicon did not survive, and this is the most
useful outcome of the whole exercise.

| MPS conclusion | On CUDA |
|---|---|
| Throughput peaks at exactly batch 16, then falls back (B5) — reproducible 3× | **Dead.** Monotonic 1→32: 38.9 → 214 → 644 tok/s. An MPS kernel artifact. |
| "The textbook batching argument does not hold here" — batching bought ~8% from bs 1→8 | **Dead.** 5.5× from 1→8, 16.6× at 32. The argument holds; MPS was the problem. |
| Block size 16 is clearly best (146 vs 115/139 tok/s) | **Dead.** Throughput nearly flat across 4/16/64, and 16 is the *slowest*. bs=4 gives 97.2% utilisation *and* full concurrency. |
| Prefix caching cuts TTFT 4.89× | **Overstated.** 1.44× on GPU. Prefilling 485 tokens costs a T4 ~42 ms; there is little to save. The *value* of prefix caching is proportional to how expensive prefill was. |

What survived: head-of-line blocking at **17.9% of decode steps** on both
platforms (a property of the scheduling policy, not the hardware); continuous
batching's advantage, which grew from 5.3× to 16×; paging's concurrency
advantage, 8× → 9.1×; and the prefix cache hit rate, 83.8% on both.

**The general lesson is about method, not about MPS.** Every one of those four
conclusions was measured carefully, reproduced multiple times, and written down
with confidence. They were still wrong, because they were properties of one
platform being mistaken for properties of the technique. Nothing in the
methodology could have caught that — only different hardware could.

---

## 6. What would close the gap, and what would not

**Would help, in expected order of effect:**

1. **CUDA graph capture of the decode step.** Removes ~360 Python-dispatched
   kernel launches per step. At this model size, likely the single biggest win.
2. **`torch.compile` on the decoder block.** Fuses norm/activation/residual
   chains that currently each round-trip through HBM.
3. **A fused attention kernel** (Triton). Avoids materialising the score matrix.
4. **Chunked prefill** (Q7), which removes the dedicated prefill iteration that
   currently stalls every running request.

**Would not help much:**

- **Optimising the gather.** Section 2 shows the gap is fully present at R1,
  which has no gather.
- **A better scheduling policy.** Section 4 shows the scheduling already
  delivers the expected effects.
- **A bigger block table, different block size.** Flat across 4/16/64 on CUDA.

---

## 7. Honest limits of this analysis

- **One GPU, one model, one session.** Tesla T4, 0.5B parameters. A 7B model on
  an A100 would shift every ratio here, probably in Inferno's favour on the
  launch-overhead argument and against it on everything else.
- **The 4.5× is not decomposed.** Section 3 names four causes and ranks them by
  argument. No profiler was run.
- **Token agreement with vLLM is 42/50, not 50/50** — expected, since different
  kernels reduce in different orders and greedy decoding amplifies the last bit
  (B1). It means the two engines are not computing bit-identical results, so a
  small part of any timing difference could be different work rather than the
  same work done faster. Unquantified.
- **vLLM was not tuned.** Default settings, no `--max-num-seqs` tuning, no
  quantization. A tuned vLLM would be faster still.
- **The batched comparison is not perfectly matched.** Inferno's bs=32 static
  batch against vLLM's continuous batching over 50 prompts — vLLM's scheduler is
  doing strictly more than Inferno's is in that row.
