# Inferno — Project Log

Master log. Append-only: entries are never rewritten or removed, only added to.
Maintained automatically as part of finishing each chunk of work.

Phases map to the rung plan: R0 baseline, R1 own attention + KV
cache, R2 static batching, R3 continuous batching, R4 paged KV cache,
R5 prefix caching.

**Status: COMPLETE. All six rungs built, measured on Apple Silicon AND on a Tesla T4, compared against vLLM 0.29, gap analysed in [docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md).**

---

## 1. Decisions

### 2026-09-15 — Process rules 6, 2, 3 dropped
Dropped the explain-and-check quiz loop, the one-file-per-turn limit, and the
150-line file cap. Kept rules 5 (hypotheses before fixes), 8-10 (no unmeasured
claims, label the hardware) and 11 (no building ahead). The dropped three were
pacing rules and the pacing was wrong; progress was ceremony-bound before any
number existed.

### 2026-09-15 — R0 applies the Instruct chat template
Chosen: wrap every prompt via `apply_chat_template(add_generation_prompt=True)`.
Rejected: raw prompt text. Qwen2.5-0.5B-**Instruct** is post-trained to expect
the template; raw text gives off-distribution continuations and an
unrepresentative baseline. Costs ~20 prompt tokens/request, recorded per record.

### 2026-09-15 — Throughput is decode-only, prefill excluded by construction
Chosen: a `StoppingCriteria` that observes but never stops, timestamping the
first generated token. `decode_tok_s = (n_new - 1) / (t_end - t_first)`.
Rejected: `n_new / total_time` (folds prefill into decode, inflates the number,
and the inflation scales with prompt length). Note `n_new - 1`: the first token
came out of prefill, so it is not a decode step.
Why it matters: the entire project rests on prefill and decode being different
regimes. A metric that averages them cannot show that.

### 2026-09-15 — R0 stores generated token IDs, not just text
Decoded text hides differences the tokenizer round-trip smooths over
(whitespace, special tokens, byte-level merges). Comparing IDs is the only
assertion that means what we want it to mean.

### 2026-09-15 — Two R0 runs: `sdpa` for speed, `eager` for parity
Fused SDPA accumulates in a different order than the eager path. Eager is the
cleaner parity reference, but it is slower, and using it as the speed baseline
would understate HuggingFace — flattering Inferno for a reason unrelated to
Inferno (rule 9). So: two runs, two files, both recorded.
**This was later confirmed by measurement — see Bugs §B1.**

### 2026-09-15 — dtype: float32 on CPU, float16 on MPS
float16 on CPU is emulated in PyTorch and pathologically slow. Consequence:
CPU and MPS runs are **not** comparable and their parity references are not
interchangeable. dtype is written into every results file.

### 2026-09-15 — Results split by hardware directory (`results/mac`, `results/gpu`)
Rejected a flat directory with a hardware field inside each file. Rule 10 says
never present a Mac number as a GPU number; a field is easy to overlook when
assembling the final table, a directory boundary is not.

### 2026-09-15 — R0 is batch size 1, sequential
R0 is the floor and the floor is what a naive implementation does. Batching is
R2 (rule 11).

### 2026-09-15 — The reported R0 baseline is a warm run
`r0_baseline_mps_sdpa_20260915_145811.json` is the baseline of record. The
first-ever run on a machine is discarded — its tail was contaminated by
one-time compilation cost (Bugs §B2). Generalises to the GPU runs: discard the
first run on any new machine, a cold CUDA context pays kernel autotuning the
same way.

### 2026-09-16 — Parity test pins the engine interface, not the cache interface
`tests/test_parity.py` pins exactly one contract:
`InfernoEngine(model_id, device, dtype, attn).generate(text, max_new_tokens)
-> list[int]`. It deliberately asserts nothing about `KVCache`, because the
cache is what R4 replaces wholesale — a parity test that must be rewritten
between rungs is not a parity test. Device/dtype/attn are read from the
reference file rather than chosen, since parity against a reference generated
under different numerics asserts nothing.

### 2026-09-16 — R1 scope: the whole forward pass is hand-written
Chosen: embedding, 24 decoder blocks, final norm, tied output projection, all
by hand. HF supplies the tokenizer and the state dict, nothing else.
Rejected: (a) keeping `Qwen2MLP`/`Qwen2RMSNorm` and writing only attention —
saves ~20 lines while keeping a numerics boundary we cannot debug across, the
worst of both; (b) keeping the HF model and replacing only the generation loop
— would make the final gap analysis an argument about a scheduler wrapped
around someone else's model.
Deciding argument: **R4 is surgery inside the attention function** (gathering
across non-contiguous blocks instead of indexing a contiguous tensor), so it
must be a function we own. Writing it at R1 is paying for R4 early, at a
discount.

### 2026-09-16 — R1 cache is deliberately naive (preallocated to the worst case)
`KVCache` preallocates `prompt_len + max_new_tokens` and never grows. This is
precisely the behaviour paged attention exists to fix: every sequence reserves
for its worst case, and the reservation must be contiguous. Keeping the waste
visible is the point; `KVCache.utilisation()` reports it, and that is the
number R4 improves.

### 2026-09-16 — R0 regenerated as true greedy; decoding policy stated explicitly
Chosen (option b): `bench/run_baseline.py` now **constructs** a
`GenerationConfig` rather than inheriting the checkpoint's, with
`repetition_penalty=1.0`, `no_repeat_ngram_size=0`, `renormalize_logits=False`.
Both references regenerated. Inferno's raw argmax is unchanged.
Rejected (option a): implementing `repetition_penalty=1.1` in Inferno to match.
That passes parity immediately (verified 12/12) but carries a sampling policy
inherited by accident from a checkpoint file through five more rungs and into
the vLLM comparison.
Why: the reference files are cheap to regenerate now and expensive to distrust
later. the project spec and the results schema both claim greedy; now that is true.
Old references moved to `results/mac/superseded/` rather than deleted.
Results now record the **full logits-processor stack** under `config.sampling`,
not just a `greedy: true` flag — and `tests/test_parity.py` asserts every field
of it, so this class of mismatch cannot recur silently.

### 2026-09-16 — A results dashboard now; the live serving view deferred to R3
Chosen: `docs/dashboard.html`, a static dashboard built from the JSON files in
`results/`, published as an Artifact. It shows the rung ladder, R0/R1
comparisons, the KV utilisation distribution, the bug log and open questions.
Rejected for now: a **live serving dashboard** — queue states, per-iteration
batch composition, KV blocks allocated/freed, prefix cache hits. That is the
view that makes continuous batching and paging legible in a way no table does,
but it needs a scheduler to exist, so it is an R3 decision. Deciding now would
change nothing about today's work.
**Hard constraint on it:** the dashboard renders only measured numbers. R2 shows
its actual pass/fail state, and R3/R4/R5 render as explicitly empty rather than
plausible. A dashboard carrying invented R4 numbers is exactly what rules 9 and
10 exist to prevent, and it would poison the artifact this project is for.
The hardware caveat (Mac/MPS, not GPU) is a banner at the top, not a footnote.

Published: https://aditicodes7.github.io/inferno/

### 2026-09-16 — R3 policy: FCFS, prefill owns its iteration, slot-based cache
**Chosen:** (a) **FCFS** — only the queue head is ever considered for
admission. (b) **Prefill gets its own iteration**; admitting a request stalls
every running request for one step. (c) **Slot-based cache** with per-slot
lengths.

**Rejected:** (a) scanning past the head for a request that happens to fit —
that is exactly what starves the head forever, and a starved request raises
nothing; (b) chunked prefill, which mixes a joining request's prompt into the
decode batch — the right answer eventually, and a different rung's problem;
(c) one `KVCache` per sequence, which is simpler but gives up batched attention
entirely.

**Why (c) specifically:** R1/R2's cache carries one `length` for the whole
batch, which works *only* because left-padding right-aligns every sequence.
Continuous batching destroys that — a request admitted at iteration 50 has 0
generated tokens while its batch-mates have 50, and they are never aligned
again. Length has to become per-slot. Every slot then reserves `max_len`
regardless of what it holds, so an 8-token sequence costs the same as a
500-token one. **That waste is now structural rather than incidental, which is
precisely the thing R4's block allocator removes.**

**Cost of (b), measured:** at low arrival rates static batching actually beats
continuous on TTFT p50 (0.034 s vs 0.102 s at 0.3 req/s), because a static batch
prefills inside the batch while continuous spends a whole dedicated iteration on
it. The dedicated prefill iteration is not free and the numbers say so.

### 2026-09-16 — R4 policy: preemption serves running sequences only; block size 16
**Chosen:** a request is evicted only so a RUNNING sequence can grow. A waiting
request that does not fit waits. Block size 16, chosen from a measured sweep.
**Rejected:** evicting on behalf of a waiting request (livelocked — §B7);
victim to the back of the queue (reintroduces starvation); an anti-thrash rule
(adds state and a knob, and only bounds the thrashing).
**Why:** the livelock needed a *waiting* request to be able to evict a
*running* one. Remove that and only a running sequence can trigger eviction —
and a running sequence that grows is making progress by definition. Structural,
not tuned.

Block size came from the table, not from copying vLLM (which also uses 16):

| block size | block utilisation | throughput |
|---|---|---|
| 4 | 95.7% | 115.26 tok/s |
| **16** | **82.8%** | **146.37 tok/s** |
| 64 | 65.3% | 138.74 tok/s |

### 2026-09-17 — R5: exact token keys, full-block sharing + CoW on the tail, LRU cached tier
**Chosen:** (a) a block is keyed by the EXACT token tuple of the whole prefix up
to and including it; (b) full blocks are shared and the partial tail is shared
too, with copy-on-write on first write; (c) a block whose refcount hits zero
moves to an LRU of cached-but-reclaimable blocks rather than back to the free
list.

**Rejected:** (a) a 64-bit digest — compact, but a collision is undetectable at
runtime and yields fluent wrong output, and token-identical output is this
project's entire premise; (b) full-block-only sharing, which needs no CoW at all
because a sequence then only ever writes into blocks it allocated itself — the
simpler design, and it gives up the tail; (c) freeing immediately, which limits
prefix caching to requests that overlap *in time*.

**Why (c) matters most:** it is the single biggest lever on the hit rate. With
immediate free, the fifteen RAG prompts only share when concurrent. With the LRU
tier they share regardless of arrival order, which is what a real system needs.

**Cost of (a):** O(prefix² / block_size) memory in keys. Fine at benchmark
scale; production would want (parent_digest, block_tokens) plus verification.

---

## 2. Build Log

### Phase 0 (R0) — baseline — 2026-09-15 — **COMPLETE, ACCEPTED**

Built:
- `bench/prompts.json` — the fixed 50. 20 short (~35 chars), 15 medium (~200),
  15 long (~2160). The long set is 15 different questions appended to **one
  shared ~2000-char document**, so R5's prefix-caching workload is built into
  the benchmark from the start rather than bolted on later.
- `bench/run_baseline.py` — HF `generate()`, greedy, batch 1, 3 warm-ups
  discarded, per-prompt records including full token IDs.
- `conftest.py`, directory layout, `.gitignore`, git init.

Worked first try: the harness end-to-end, TTFT instrumentation, results
serialisation.

Did **not** work first try:
- Environment. System Python was 3.14.7 with no torch; had to build a 3.11 venv.
- `transformers` 5.17.0 renamed `torch_dtype` → `dtype` in `from_pretrained`.
- Two background runs were killed by session ends before completing; the model
  download had to be restarted. Long benchmark runs now go in the foreground.

### Phase 1 (R1) — own attention + KV cache — 2026-09-16 — **IN PROGRESS**

Built (test first, per rule 4):
- `tests/test_parity.py` — written before any implementation, confirmed failing
  with `No module named 'inferno.engine'` across all 6 parametrised cases.
  Reports **where** divergence occurs, not just that it did: token 0 implicates
  prefill/weight-loading/output projection, token 1 the first decode step, token
  40 something that accumulates.
- `inferno/cache.py` (81 lines) — preallocated per-layer KV storage.
- `inferno/model.py` (220 lines) — hand-written Qwen2 forward pass.
- `inferno/engine.py` (138 lines) — weight loading, chat template, prefill +
  greedy decode loop.

Worked first try: weight loading (including the Qwen2 quirks — biases on
q/k/v but not o_proj, `tie_word_embeddings=True` so there is no
`lm_head.weight`), `rope_theta` retrieval from the nested `rope_parameters`
config, the prefill/decode shared code path, GQA expansion, no crashes, fluent
output.

Did **not** work: **parity, 1/50 passing.** Root-caused to the checkpoint's
`repetition_penalty` (Bugs §B3) — not to any of the four failure modes
predicted for R1 (cache off-by-one, decode position IDs, mask shape with a
non-empty cache, RoPE on the wrong side of the cache write). **None of those
occurred.** The hand-written forward pass was numerically correct on its first
run; teacher-forced top-1 agreement was 371/371 on mps/fp16 and 32/32 on
cpu/fp32.

After regenerating R0 under true greedy: **parity 50/50, full prompt set,
195s.** R1 accepted.

Also built: `docs/dashboard.html` (results dashboard, published as an Artifact;
categorical palette validated for CVD separation in both light and dark themes),
`bench/diagnose_parity.py` (teacher-forced agreement harness — the
tool that localised B3) and `bench/run_inferno.py` (R1 measurement, metric
definitions identical to `run_baseline.py`).

### Phase 2 (R2) — static batching — 2026-09-16 — **COMPLETE, ACCEPTED**

Built: `inferno/batch_engine.py` (left-padding, batched prefill + decode),
padding-aware causal mask and pad-derived position ids in `inferno/model.py`,
`tests/test_batch_parity.py`, `bench/run_batched.py`.

Worked first try: left-padding and the single shared `cache.length` it enables
(right-alignment means every sequence writes at the same offset); position ids
from `pad_mask.cumsum(-1) - 1`; batched prefill and decode; **the batching logic
itself, which float32 showed was correct from the first run**.

Did **not** work first try: a `NameError` from my own patch dropping the
`rope()` call (loud, instant, cheap); then B4 — which consumed a wrong
hypothesis, a wrong probe, and a wrong fix before the instrumented trace found
it. Details in §B4.

Acceptance: `INFERNO_FULL=1 pytest tests/test_batch_parity.py` — **10 passed**.
float32 exact at batch sizes 1/2/4/8/16 across 16 mixed-length prompts;
float16 79/80 exact with one drift (`long-04` at bs=4, token 37).

### Phase 3 (R3) — continuous batching — 2026-09-16 — **COMPLETE, ACCEPTED**

Built: `inferno/scheduler.py` (154 lines, no tensors at all),
`SlotKVCache` in `inferno/cache.py`, `forward_slots` /
`forward_prefill_slot` in `inferno/model.py`, `inferno/continuous_engine.py`,
`tests/test_scheduler.py` (10 tests), `tests/test_continuous_parity.py`,
`bench/run_continuous.py` (the arrival-driven harness the final deliverables
also need).

Worked first try: the per-slot scatter (`self.k[layer][slots, :, pos]` with two
index tensors, so every row lands at its own offset — the whole difference from
R2, where one offset served the batch); the key mask built from per-slot
lengths; refactoring `_layer` to take a KV-writer callable so one 24-layer loop
serves both cache types **without regressing R1 or R2 parity** (12 passed);
R3 parity on the first run — 10 passed across 3 arrival patterns × 3 slot counts.

Did **not** work first try: **the scheduler assigned slots in the wrong place**
(§B6). Caught by a unit test, before any GPU time was spent.

Acceptance: `pytest tests/test_scheduler.py tests/test_continuous_parity.py` —
**20 passed**. Output is identical whether a request arrives at t=0, staggered,
or in a late burst, at 1, 2 and 4 slots.

### Phase 4 (R4) — paged KV cache — 2026-09-16 — **COMPLETE, ACCEPTED**

Built: `inferno/block_manager.py` (page table + refcounting, no tensors),
`inferno/paged_cache.py` (physical blocks + the gather), `forward_paged_prefill`
/ `forward_paged_decode` in `inferno/model.py`, `inferno/paged_engine.py`
(admission, preemption, recompute), `Scheduler.preempt`,
`tests/test_block_manager.py` (14 tests), `tests/test_paged_parity.py`,
`bench/run_paged.py`.

Worked first try: **the allocator — all 14 tests passed on the first run**,
including the conservation invariant (`free + referenced == capacity`) checked
after every operation, refcount-to-zero-exactly-once under repeated sharing, and
a 200-cycle allocate/append/free loop with no leak. Also first try: the block
table scatter and gather, and paged parity — identical output on the first
end-to-end run.

Did **not** work: **the preemption livelock (§B7)**, which is the best bug in
the project so far. Then two measurement mistakes of my own: a test pool sized
from `max_new_tokens` when the prompts hit EOS long before (so nothing was ever
preempted, caught only because the test asserts `preemptions > 0`), and a
utilisation metric that reported 126%.

Acceptance: `pytest tests/test_block_manager.py tests/test_paged_parity.py` —
**20 passed**. Parity at block sizes 4/16/64, parity under memory pressure with
preemption actually occurring, parity at `max_concurrent=1`, and **zero blocks
leaked in every case** (the engine raises if any block is outstanding at the
end, so a leak fails the run rather than surfacing later as premature OOM).

### Phase 5 (R5) — prefix caching — 2026-09-17 — **COMPLETE, ACCEPTED**

Built: `inferno/prefix_cache.py` (`PrefixBlockManager` — prefix keys, the LRU
cached tier, copy-on-write), `inferno/prefix_engine.py`, a `start` offset on the
paged prefill path so cached blocks are genuinely never recomputed,
`tests/test_prefix_cache.py` (11 tests), `tests/test_prefix_parity.py`,
`bench/run_prefix.py`, and `run_tests.sh`.

Worked first try: prefix matching and sharing, the LRU reclamation tier, CoW
ownership transfer, and end-to-end parity — the first A/B run gave identical
output with a 59% hit rate.

Did **not** work first try: two of my own tests (one allocated four blocks from
an exhausted pool; one mis-stated an expected hit count), and the full test suite
itself (§B9). No engine bug in this rung.

Acceptance: `pytest tests/test_prefix_cache.py tests/test_prefix_parity.py` —
**16 passed**. Full regression suite via `./run_tests.sh`: **63 tests across 8
files, all green** (scheduler 10, block_manager 14, prefix_cache 11, parity 6,
paged_parity 6, prefix_parity 5, continuous_parity 10, batch_parity 6). Parity holds for share-then-diverge, for cache-on vs cache-off,
for **divergence inside a block rather than on a boundary** (block_size 64 — the
only configuration that actually exercises copy-on-write), for two identical
prompts, and under a tight pool where reclamation and preemption interact.

---

## 3. Bugs & Debugging

### B1 — sdpa and eager produce different tokens from identical inputs
**2026-09-15. Root cause understood. Not a defect — a property.**

*What broke:* Comparing the two R0 runs, 10 of 50 prompts produced **different
token sequences** under identical model, device, dtype and greedy settings.
Earliest divergence at generated token 1 (`short-15`). Three prompts produced
different *lengths* because one path hit EOS and the other did not.

*Mechanism:* Fused SDPA kernels accumulate the attention score matmul in a
different order than the eager path. Floating-point addition is not
associative, so the logits differ in the last bits. Under greedy decoding this
is normally invisible — argmax is robust to tiny perturbations — until two
candidate tokens are nearly tied. Then one bit flips the argmax, the two
sequences are now conditioned on **different prefixes**, and they never
re-converge. A 1-ulp difference becomes a completely different paragraph.

*What it taught us:* "Token-identical output" is only a coherent assertion
against a reference produced with the *same attention implementation*. Testing
R1 against the sdpa file would have produced 10 failures with no bug behind
them — days of hunting for a defect that does not exist. Greedy decoding is a
chaotic amplifier of numerical noise, not a stabiliser.

### B2 — R0 TTFT p95 not reproducible: 297ms vs 88ms
**2026-09-15. Resolved (methodology change, no code change).**

*Symptom:* First full R0 run reported TTFT p95 = 297.3ms with five requests
above 250ms. TTFT was **not monotonic in prompt length**: a 64-token prompt
took 320ms while a 73-token prompt took 21ms and a 485-token prompt took 301ms.

*Hypotheses:* (1) shape-triggered kernel compilation on MPS; (2) insufficient
warm-up; (3) host noise (thermal, background load).

*Test:* re-run identically (isolates noise), then re-run with `--reverse` so
every prompt occupies a different position (separates position-in-run from
prompt-identity). Warm-up pinned to a fixed prompt so ordering was the only
changed variable.

*Result:*

| | run1 fwd (first ever) | run2 fwd | run3 reversed |
|---|---|---|---|
| decode tok/s | 34.77 | 35.75 (+2.8%) | 35.63 (+2.5%) |
| TTFT p50 | 61.9 ms | 61.1 ms (−1.3%) | 64.3 ms (+4.0%) |
| TTFT p95 | 297.3 ms | 87.8 ms (**−70.5%**) | 87.4 ms (−70.6%) |
| requests > 150 ms | 5 | **0** | **0** |

*Root cause:* Hypothesis 3 is dead — the effect reproduces perfectly in its
absence, twice. Hypothesis 2 is dead as stated — run 2 used identical warm-ups
and had zero outliers. Hypothesis 1 survived **in refined form**: the outliers
followed neither the prompt (`medium-06`: 320 → 73 → 86 ms) nor the position
(positions 20/21/26/35/36 in run1; those same prompts at 13/14/23/28/29 in run3
with no outliers). They appeared **only in the first run ever executed on this
machine**. macOS keeps a Metal shader/pipeline cache on disk, so the first
process to hit a shape pays the compile and every later process reuses it.

*Honest limit:* the data proves *"first execution on this machine only,
independent of prompt and of position."* Persistent shader caching is the
leading explanation consistent with that; it was **not directly proven**. A
direct test (clear the Metal cache, re-run) was not done.

*What it taught us:* **the first benchmark run on any new machine is
untrustworthy for tail latency.** Directly relevant to the rented-GPU runs — a
cold CUDA context pays kernel autotuning the same way.

### B3 — R1 parity failure: 1/50, divergence scattered from token 0 to 34
**2026-09-16. RESOLVED.** Fix: option (b) — R0 regenerated as true greedy; Inferno unchanged.

*Symptom:* Hand-written engine produces fluent, sensible text and never
crashes, but only `short-00` (8 generated tokens, the shortest) matches the
reference exactly. Full distribution of first-divergence index across 50
prompts:

```
min 0   median 8   max 34
diverged at token 0: 4 prompts    diverged at token 1: 4 prompts
```

Lengths diverge in both directions — `short-14` produced 109 tokens where the
reference produced 83; `short-03` produced 62 where the reference produced 120.

*Hypotheses:* (1) a logic bug that is usually harmless — an off-by-one in the
prefill mask, a position edge case; (2) precision / accumulation-order
differences between Inferno's kernels and HF's; (3) a mismatch **outside** the
model — chat template, EOS set, or the sampling rule.

*Tests that distinguished them:*

**A. Teacher-forced agreement (mps/fp16).** Feed both implementations the same
prefix at every step — the reference's own tokens — and compare argmax on raw
logits. This separates "our logits are wrong" from "we diverged once and then
drifted."

```
short-17   max |logit diff| 0.084  (scale ~38, rel 2.2e-03)   top-1  51/51 agree
long-01    max |logit diff| 0.070  (scale ~31, rel 2.3e-03)   top-1 128/128 agree
short-09   max |logit diff| 0.066  (scale ~31, rel 2.2e-03)   top-1  64/64 agree
medium-06  max |logit diff| 0.059  (scale ~31, rel 1.9e-03)   top-1 128/128 agree
```

371/371 steps agreed. Relative error ~2e-3 is about 2x float16 epsilon —
ordinary accumulated rounding over 24 layers, nothing structural.

**B. Free-running parity in float32 on CPU** (deterministic, no MPS kernel
variance): still 1/5. **C:** Inferno is deterministic on MPS across repeated
runs, and its output does not depend on cache preallocation capacity — both
ruled out. **D. Teacher-forced on CPU/fp32:** 32/32 agreement, max logit diff
1e-4.

*The contradiction that cracked it:* teacher-forced step 0 and free-running
token 0 are **the same computation with the same input** — there is no prior
token to have diverged on. Yet teacher-forced agreed at step 0 while
free-running diverged at token 0 on the identical config. Two things that must
be equal were not, so the difference could not be in the forward pass at all.
It had to be in what happened to the logits *after* the model returned them.

*Root cause:* `Qwen2.5-0.5B-Instruct` ships a `generation_config.json`
containing **`repetition_penalty: 1.1`**. HuggingFace's `generate()` applies
`RepetitionPenaltyLogitsProcessor` **in greedy decoding** — it is not gated on
`do_sample`, so passing `do_sample=False` does not disable it. Every logit for
a token already present in the sequence is divided by 1.1 (or multiplied, if
negative) before argmax. Inferno took argmax over raw logits.

*Confirmation:* same engine, same weights, 12 prompts, only the penalty
changed — **1/12 exact → 12/12 exact**.

*Why the failure pattern looked the way it did:* the penalty only bites once a
token repeats, so early tokens usually match and divergence appears a few
tokens in (median 8). It bit at token 0 for 4 prompts whose *prompt* already
contained the token the model wanted next. And it changed generation lengths in
both directions by shifting when EOS won.

*What it taught us — the part that matters for the writeup:*
1. **"Greedy" is not a complete specification of a decoding procedure.** The
   R0 results files record `"greedy": true`, and that claim was *wrong* —
   those runs had a repetition penalty applied. A baseline is only reproducible
   if the full logits-processor stack is recorded, not just the sampling mode.
2. **The bug was never in the 220 lines of hand-written transformer.** Every
   suspicion pointed at attention, the mask, RoPE, or the cache. The forward
   pass was correct from the first run; the defect was one line of sampling
   policy inherited silently from a JSON file in the checkpoint.
3. **Teacher forcing is the diagnostic that localises this class of bug.**
   Free-running comparison can only say "these diverge." Teacher forcing
   answers "are the logits right?" separately from "does the loop agree?", and
   the contradiction between the two answers is what identified the layer the
   bug lived in.

### B4 — R2 batch parity fails at batch_size 4, passes at 1 and 2
**2026-09-16. OPEN.** Full hypotheses and the distinguishing experiment are in
`docs/bugs.md`.

*Two distinct failure shapes in one batch*, which is itself the most useful
clue — they are unlikely to share a cause:

```
medium-00, medium-01   diverge at token 0, emit token id 0 repeatedly
                       got [0, 0, 0]   want [95456, 0, 6771]
long-01                diverges at token 67 after 67 EXACT matches,
                       plausible continuation
```

*What the batch-size pattern says:* `MIXED[:8]` chunked by 4 puts the ~80-token
medium prompts in the same batch as the ~480-token long prompts, so the medium
sequences carry ~400 pad slots each. At batch 2 the chunks pair similar lengths
and there is almost no padding. **The failure tracks the padding ratio, not the
batch size.** R1 parity still passes 6/6, so the unbatched path is untouched.

*Root cause, confirmed in part (2026-09-16):* **float16 range, triggered by
fully-masked padding rows.** Measured on the `medium-00` + `long-00` pair — 87
real tokens against 485, so 398 pad slots on the medium sequence:

```
query rows that can see NO key at all : 398   (all padding rows)
mask contains -inf                    : False
(-5) + finfo.min in fp16              : -65504.0, NOT -inf
first layer with ANY NaN              : 1
first layer with NaN in REAL rows     : 2
same batch in float32                 : clean at every layer
```

**The hypothesised first step was wrong, and the correction is the interesting
part.** Softmax over a fully-masked row does *not* produce NaN — every entry is
`finfo.min`, so after the float32 softmax the row is a finite **uniform**
distribution (verified: sums to 1.0). And `finfo.min` does not overflow to
`-inf` when a score is added to it; fp16 saturates at −65504. Both halves of the
guess were wrong while the conclusion — "fully-masked rows are the trigger" —
was right, which is exactly why the test was worth running instead of reasoning
it out.

What is confirmed:
- NaN originates in **padding** rows at layer 1. Pad-row values at layer 0 are
  small and finite (max |x| = 1.5), and no `inf` appears in any layer *output* —
  so the overflow happens in an intermediate **inside** layer 1 and is consumed
  into a NaN before reaching the output. Which intermediate is not yet pinned.
- It reaches **real** tokens one layer later, and the path is the one worth
  remembering: pad K/V are written into the shared cache, and a masked weight is
  exactly 0 after the float32 softmax — but **0 × NaN = NaN** in the value
  matmul. *Masking does not protect a real query from a NaN pad value.*
- It is float16-specific; float32 is clean end to end.
- Fully-masked rows exist **because of left padding** — a leading pad query row
  has no real key at or before it. Structural to the padding scheme, not an edge
  case.

*Separation test (2026-09-16):* compare each prompt **batched-of-4** against the
same engine running it **alone** — identical code path both ways, so only batch
composition changes.

```
float16   medium-00  diverge@0   (all-zeros = NaN)
          medium-01  diverge@0   (all-zeros = NaN)
          long-00    IDENTICAL   (485 tok, ZERO padding)
          long-01    diverge@67
float32   all four   IDENTICAL
```

**The batching logic is correct.** In float32, batched output is bit-identical to
unbatched for every prompt. The mask, the position ids, the cache offsets and the
left-padding scheme are all right. There is no batching bug — which is not what
the failing test looked like it was saying.

There are **two** float16 effects and they are unrelated:

- **(a) NaN from fully-masked padding rows.** Catastrophic, scales with padding.
  The medium prompts carry ~400 pad slots and die at token 0; `long-00` carries
  **zero** padding and is untouched.
- **(b) Near-tie argmax flips from batch-shape-dependent reduction order.**
  Subtle, and nothing to do with padding. The decisive evidence: `long-01` has
  exactly 11 pad slots in the batch-of-2 run that **passed** and exactly 11 in
  the batch-of-4 run that diverged at token 67. Padding identical, batch size
  different. This is B1 again, one level up.

**(b) is not fixable.** It is the same float-associativity property proven in B1,
and it means *"parity at every batch size" may be unachievable in float16 between
differently-shaped matmuls.* That is a criterion decision, not a defect — see Q2.

---

**ROOT CAUSE OF (a), CONFIRMED 2026-09-16.** Instrumenting every tensor inside
the layer put the first nonfinite value in **softmax, layer 8, on padding rows**:

```
L0  scores= 940.0   softmax=1.0   x=0.0
L7  scores=  21.4   softmax=0.0   x=0.0
L8  scores= 224.9   softmax=NONFINITE:193030   ->  x=NaN
```

The additive mask used `torch.finfo(dtype).min`. float16's most negative finite
value is **−65504**, so adding any score beyond about −16 rounds past the end of
the range to **−inf**. Scores reach ±225 by layer 8. A fully-masked row then
becomes *all* −inf, and softmax computes `exp(-inf − (−inf))` = **NaN**.

```
score    -5 + finfo.min  ->  -65504.0     (rounds back, finite)
score   -50 + finfo.min  ->  -inf
score  -225 + finfo.min  ->  -inf
score  -225 + min/2      ->  -32992.0     (finite, and exp() still underflows to 0)
```

**Fix:** `mask_fill_value(dtype) = finfo.min / 2`. A masked position must
contribute *exactly zero* weight, which requires a value that is very negative —
not maximally negative. `finfo.min` is the obvious choice and is precisely the
one value that cannot absorb an addition.

**Two mistakes worth keeping, because both are instructive:**

1. **The probe that misled me.** I tested `(-5) + finfo.min`, saw `-65504.0`, and
   concluded "does not overflow" — and wrote that into the log as a confirmed
   fact. −5 is within half a ULP of the endpoint so it rounds back; −50 does not.
   *A probe value chosen without thinking about the real magnitude of the
   quantity produced a confident wrong conclusion*, and that conclusion then
   ruled out the true cause for an entire debugging cycle.
2. **The wrong fix.** Zeroing the residual stream at padding positions is
   provably output-neutral and sounded principled. It addressed *activation
   growth* when the overflow was in the *mask addition*. It moved the first NaN
   from layer 1 to layer 8 and made `long-01` strictly worse (divergence at 67 →
   at 0). **A fix that moves a symptom without removing it is evidence the
   mechanism is still wrong** — the right response was to revert it and go
   measure, not to patch on top of it.

*Build note:* one self-inflicted error before this — the patch that added the
padded-mask branch dropped the `cos, sin = self.rope(...)` call, giving a clean
`NameError` on the first run. Mentioned only because it is the contrast case:
a structural mistake fails loudly and instantly, which is the cheap kind.

### B5 — throughput vs batch size is non-monotonic, with an isolated peak at 16
**2026-09-16. OPEN — reproducible, mechanism unconfirmed.**

*Symptom:* static batching buys almost nothing from bs=1 to bs=8 (28.30 → 30.53
tok/s, +8%), then bs=16 triples it, then bs=20 falls back.

```
bs    tok/s    wall_s   batches   steps   ms/step
 1    28.30     185.7        50    5257      35.3
 2    30.16     174.3        25    2811      62.0
 4    30.60     171.8        13    1538     111.7
 8    30.53     172.2         7     896     192.2
12    37.06     141.6         5     640     221.3
16    97.06      54.2         4     512     105.9   <-- isolated peak
20    68.36      76.9         3     384     200.3
```

*Reproducible:* bs=16 measured at 96.96, 97.06 and 99.42 tok/s across three
independent runs (±2.5%). Token counts are the same at every batch size (5257 vs
5263), so this is not an artifact of generating less.

*What it is not:* not batch composition. Weighting each batch by its padded
width, bs=16 does *more* total attention work than bs=12 (2.22M vs 2.40M
sequence-step-width units is only a 7% spread) while taking 2.6× less wall time.
Not a monotone cliff either — bs=20 is slow again, so it is a peak at 16, not a
threshold above it.

*Hypotheses:* (1) MPS matmul kernel selection at a 16-wide tile boundary — 16 is
a natural SIMD width, and per-step cost halving while work doubles is what a
better kernel looks like; (2) a memory-layout or alignment effect that happens
to be satisfied at exactly 16; (3) something about how the decode step's small
matrices are dispatched that changes shape at 16.

*Distinguishing test, not yet run:* sweep 13,14,15,16,17,18 with fixed-length
prompts so composition is held constant. If the dip is exactly at 16 and nowhere
else, it is kernel selection.

*Why this matters beyond the anomaly:* **the textbook batching argument does not
hold here.** Per-step cost from 1→8 grows ~1.75× per doubling, i.e. nearly
linearly with batch size, so batching recovers almost nothing. That argument
assumes decode is weight-bandwidth-bound — true for an 8B model on an A100,
evidently not for a 0.5B model on MPS, where the weights are small enough to sit
in cache and the fixed per-step cost is not being amortised. This is exactly the
kind of honest negative result the gap analysis is for, and it is a strong reason
to re-run the batch-size sweep on rented GPU hardware before drawing conclusions.

### B6 — the scheduler reserved slots too late to be usable
**2026-09-16. Found by a unit test before the engine existed. Fixed.**

*Symptom:* `test_slot_is_freed_when_a_request_finishes_and_is_reused` failed on
`assert d.prefill.slot == 0` — the slot was `None` on a request the scheduler
had just decided to prefill.

*Root cause:* `schedule()` returned "prefill this request" and `on_prefilled()`
assigned the slot afterwards. But **the engine needs the slot to write K/V into
before it can run the forward pass.** As written, the scheduler was unusable by
the thing it existed to drive — the API was wrong, not the state machine.

*Fix:* reserve the slot inside `schedule()` and carry it on the decision, made
idempotent so calling `schedule()` twice without acting cannot consume two
slots. Plus a branch in `finish()` for a request that reserved a slot and was
never prefilled, which would otherwise leak it silently.

*Why it matters beyond the fix:* this is the first bug in the project caught by
a **unit test rather than a benchmark**, and it cost seconds instead of the
minutes-long parity runs everything else has needed. The scheduler was written
to own no tensors specifically so its ugly cases — slot reuse, leaked slots,
starvation — could be tested without a model. That separation paid for itself
immediately. All three of the R3 failure modes predicted at the start of the
project are silent: a slot reused one iteration early is corruption, a leaked
slot is a hang, a starved request raises nothing.

### B7 — preemption livelock: two requests evict each other forever
**2026-09-16. Fixed.** Full trace in `docs/bugs.md`.

*Symptom:* the R4 test suite produced no output and did not terminate. No error,
no crash, no progress. Killed at 600 s.

*Distinguishing test:* replay the **admission policy alone** — BlockManager plus
Scheduler, no model, no tensors. Under a second, versus the ten minutes the full
suite had already burned.

```
short-00 needs 3 blocks, medium-00 needs 6, pool has 8

iter 0  admit short-00                   free=5
iter 1  admit medium-00, evict short-00  free=2
iter 2  admit short-00, evict medium-00  free=5
iter 3  admit medium-00, evict short-00  free=2   ... forever
```

*Root cause:* **two individually correct rules composing into a cycle.**
`preempt()` returns the victim to the FRONT of the queue so it is not starved by
everything behind it; FCFS admission only considers the HEAD. Together the victim
becomes the head, is immediately re-admitted, and immediately forces the next head
to evict it again.

*Why it is worth keeping:* this is **not starvation**, the failure mode predicted
for the scheduler at the start of the project. Both requests were repeatedly
*admitted*. They simply never held memory long enough to produce a token —
forward progress exactly zero while the system looked perfectly busy. Reviewing
either rule in isolation would never have found it; neither is wrong alone.

*Fix:* preemption now only ever serves a running sequence that cannot grow. The
cycle required a waiting request to be able to evict a running one; removing that
makes progress structural rather than tuned.

### B8 — the R4 benchmark reported 126% cache utilisation
**2026-09-16. Fixed.** The metric divided total tokens across all served
requests by `n_slots × max_len`, which is correct only when every request holds
a slot simultaneously. On the capacity run — 56 requests through 7 slots — it
over-counted roughly 8×. Corrected to occupancy-while-resident: **R3's real
utilisation is 15.8%, not 39.0%.** The error flattered the baseline, so R4's
advantage is *larger* than first reported.

**Third error this session of the same shape** — a constant or formula that is
right for the case in mind, applied to a case where it is not. See also the −5
mask probe in §B4, and a test pool sized from `max_new_tokens` when the prompts
stop at EOS long before. What caught this one is that the number was
**impossible** rather than merely wrong; at 84% it would have gone into the log
unchallenged. Prefer metrics that can be visibly out of range.

### B9 — the test suite swapped itself to a standstill
**2026-09-17. Fixed, and it is a constraint rather than a defect.**

*Symptom:* `pytest tests/` ran 25 minutes with no output and no progress.

*Diagnosis:* the process was at **48 MB RSS and ~11% CPU** with system swap at
**6.97 GB of 8 GB**. Not hung — paged out.

*Root cause:* every test file holds its engine in a **session-scoped** fixture,
and one pytest session keeps them all alive at once: fp16 for `test_parity`,
fp32 **and** fp16 for `test_batch_parity`, fp32 for `test_continuous_parity`,
`test_paged_parity` and `test_prefix_parity`. That is roughly five model copies
resident on a machine that holds about two. Each file passes comfortably alone.

*Fix:* `run_tests.sh` runs each file in its own pytest process, so each engine
is freed before the next loads. The three tensor-free files run first, so a logic
regression surfaces in 0.05 s instead of after several minutes of model loading.

*Why it is worth keeping:* I had told the user this run would take 5–8 minutes
and attributed the wait to model loading. It would never have finished. **A
process that is swapping looks exactly like a process that is working**, and the
distinguishing evidence — RSS far below the working set, CPU far below 100% —
is not visible from the test output at all. This will matter again on rented GPU
instances, which often have less RAM than this laptop.

### B10 — the test runner could report a failing file as green
**2026-09-17. Fixed.** Found while reading output, not from a failure.

*Symptom:* `run_tests.sh` printed a bare `.` for `test_batch_parity.py` where a
`6 passed` summary should have been.

*Root cause:* the script extracted pytest's summary with `tail -2 | head -1`.
That file prints progress lines from inside tests (`capsys.disabled()`), which
interleave with the dots, so the captured line was not the summary. The script
then grepped **that captured line** for `failed|error` to decide pass/fail —
so a file that genuinely failed would have been reported green.

*Fix:* match pytest's summary line explicitly against the whole output, treat a
missing summary as a failure, and print an explicit `ALL GREEN` /
`FAILURES PRESENT` verdict. Also added `test_prefix_parity.py`, which had been
left out of the list.

*Why it is worth keeping:* same class as the other slips this session — logic
correct for the case in mind, applied where it does not hold. But the direction
of failure is what matters. The 126% utilisation metric (§B8) failed *loudly*
and caught itself; this one failed **silently toward green**, which in a test
runner is strictly worse. A suite that cannot fail is worth exactly as much as
no suite. Verified afterwards by running the file directly: 6 passed.

---

## 4. Benchmark Results

### R0 — HuggingFace baseline
**Hardware: MacBook, Apple Silicon, MPS backend. NOT a GPU number.**
Config: `Qwen/Qwen2.5-0.5B-Instruct`, float16, greedy, batch size 1,
`max_new_tokens=128`, chat template applied, 3 warm-ups discarded, 50 prompts.

| metric | sdpa (speed baseline) | eager (parity reference) |
|---|---|---|
| decode throughput | **35.75 tok/s** | 26.15 tok/s |
| TTFT p50 | 61.1 ms | 56.1 ms |
| TTFT p95 | 87.8 ms | 203.5 ms |
| peak RSS | 2072 MB | 1247 MB |

By category (sdpa, warm run): short 37.21 · medium 36.79 · **long 34.77 tok/s**

Raw: [`results/mac/r0_baseline_mps_sdpa_20260915_145811.json`](results/mac/r0_baseline_mps_sdpa_20260915_145811.json) ·
[`results/mac/r0_baseline_mps_sdpa_rev_20260915_150101.json`](results/mac/r0_baseline_mps_sdpa_rev_20260915_150101.json) ·
parity reference [`results/mac/r0_baseline_mps_eager_20260915_131830.json`](results/mac/r0_baseline_mps_eager_20260915_131830.json)

*Interpretation:* Long prompts decode ~7% slower than short ones (34.77 vs
37.21 tok/s) with identical model and settings — the only difference is 473
context tokens versus 38. That is the KV cache getting more expensive to read
on every single decode step, and it is the R4 motivation appearing unprompted
in the baseline.

*Caveat:* the eager run's **timings** carry first-run contamination (it was the
first eager process on this machine — see B2). It is used only for token IDs.
Do not quote its latency.

### R0 acceptance — PASSED
- Token-identical across a repeat run: **50/50**.
- Token-identical across a **reversed-order** run: **50/50**. Order
  independence at batch 1 is now measured, not assumed — which matters because
  R2 batching can legitimately break it, and this run is the evidence that the
  cause would be the batching code.
- decode tok/s reproduces within **2.8%** (band ~5%); TTFT p50 within **4.0%**.
- TTFT p95 reproduces within 0.5% *between warm runs*; cold run excluded (B2).

### R0 — re-baselined under TRUE greedy (repetition_penalty=1.0)
**Hardware: MacBook, Apple Silicon, MPS. NOT a GPU number.**
Supersedes the numbers above, which were generated with an inherited
`repetition_penalty=1.1`. Old files kept in `results/mac/superseded/`.

| metric | sdpa (speed baseline) | eager (parity reference) |
|---|---|---|
| decode throughput | **36.29 tok/s** | 26.77 tok/s |
| TTFT p50 | 54.1 ms | 51.0 ms |
| TTFT p95 | 73.0 ms | 78.1 ms |
| peak RSS | 1998 MB | 2422 MB |

Raw: [`r0_baseline_mps_sdpa_20260916_114721.json`](results/mac/r0_baseline_mps_sdpa_20260916_114721.json) ·
[`r0_baseline_mps_eager_20260916_115052.json`](results/mac/r0_baseline_mps_eager_20260916_115052.json)

*Interpretation:* changing only the decoding policy moved decode throughput
from 35.75 to 36.29 tok/s (+1.5%) and cut TTFT p95 from 87.8 to 73.0 ms. The
throughput shift is near the ~5% reproducibility band and is not evidence of
anything; the point of the re-run was correctness, not speed.

### R1 — Inferno, hand-written forward pass + KV cache — **ACCEPTED**
**Hardware: MacBook, Apple Silicon, MPS, float16. NOT a GPU number.**
Config: same 50 prompts, eager attention, greedy, `repetition_penalty=1.0`,
batch size 1, `max_new_tokens=128`, 3 warm-ups discarded.

| metric | R0 eager (same attn path) | **R1 Inferno** | R0 sdpa (fastest HF) |
|---|---|---|---|
| decode throughput | 26.77 tok/s | **29.21 tok/s** (+9.1%) | 36.29 tok/s (−19.5%) |
| TTFT p50 | 51.0 ms | **93.7 ms** (1.84× slower) | 54.1 ms |
| TTFT p95 | 78.1 ms | **358.8 ms** (4.59× slower) | 73.0 ms |

By category (Inferno): short 32.24 · medium 30.02 · **long 26.96 tok/s**

Raw: [`r1_inferno_mps_20260916_115813.json`](results/mac/r1_inferno_mps_20260916_115813.json)

*Correctness proof:* **50/50 token-identical** to the eager reference
(`INFERNO_FULL=1 pytest tests/test_parity.py` — 50 passed in 195s).

*Interpretation:* against the **same attention implementation**, Inferno's
decode is 9% faster than HuggingFace — plausibly because the decode loop has
far less per-step Python overhead (no logits-processor stack, no stopping-criteria
machinery, no cache abstraction), and at 0.5B on MPS per-step dispatch overhead
is a real fraction of step time. That is a hypothesis, not a measurement.
**Prefill is much worse: 1.84× slower at p50 and 4.59× at p95**, and the p95 gap
is dominated by the 475-token RAG prompts. This is the first genuine performance
gap of the project and it is in exactly the regime the project claims to care
about.

*KV cache, verified linear:* allocated bytes / (prompt + max_new) = **12,288
bytes per token for all 50 prompts** — a single value, exactly matching the
predicted `2 × 24 layers × 2 KV heads × 64 head_dim × 2 bytes`.

*KV occupancy:* mean utilisation **85.8%** (173.2 MB used of 187.8 MB
allocated). **This number flatters the design and should not be quoted alone:**
35 of 50 prompts ran the full 128 tokens, so there was little early termination
to waste reservation on. The prompts that did stop early show the real cost —
`short-00` generated 8 tokens of a reserved 128 and held **26.2% utilisation,
1.49 MB wasted on one request**. A workload with realistic length variance
would show far more waste. That gap is what R4 exists to close.

### R2 — static batching, left-padded — **ACCEPTED**
**Hardware: MacBook, Apple Silicon, MPS, float16. NOT a GPU number.**
Config: all 50 prompts, eager attention, greedy, batch composed **in list
order and deliberately not sorted by length**, `max_new_tokens=128`.

| batch | tok/s | wall s | latency p50 | latency p95 | blocked steps (mean) | max | steps wasted |
|---|---|---|---|---|---|---|---|
| 1 | 28.30 | 185.7 | 4.24 s | 5.09 s | 0.0 | 0 | **0.0%** |
| 2 | 30.16 | 174.3 | 7.13 s | 9.34 s | 7.3 | 95 | 6.5% |
| 4 | 30.60 | 171.8 | 13.54 s | 17.74 s | 12.8 | 115 | 10.8% |
| 8 | 30.53 | 172.2 | 24.66 s | 32.11 s | 22.9 | 120 | 17.9% |
| 12 | 37.06 | 141.6 | 27.87 s | 39.20 s | 23.0 | 120 | 18.0% |
| 16 | **97.06** | 54.2 | 10.65 s | 25.39 s | 22.7 | 120 | 17.8% |
| 20 | 68.36 | 76.9 | 32.38 s | 33.77 s | 22.9 | 120 | 17.9% |

Raw: [`r2_batched_mps_bs1-2-4_…`](results/mac/) · [`…bs8-16…`](results/mac/) ·
[`…bs12-20…`](results/mac/) · [`…bs16…`](results/mac/)

*Correctness proof:* float32 batched output **bit-identical** to unbatched at
every batch size, 16 mixed-length prompts. float16 79/80 exact (see Q2 for why
float16 is not held to exactness).

*Interpretation — head-of-line blocking, measured.* `blocked_steps` is the
number of decode steps a request sat in the batch **after producing its last
token**, because a static batch runs until every member finishes. At batch 8 and
above, **~18% of all decode steps are spent computing tokens that get thrown
away**, and the worst-hit request waits **120 steps** — it finished in 8 and was
delivered after 128. Latency p95 climbs 5.09 s → 32.11 s from batch 1 to 8 while
throughput is flat.

That is the entire argument for R3 in one row: the cost is structural, not a
tuning problem. A batch is fixed for its lifetime, so a request that needed 8
tokens waits on one that needed 128. No batch size fixes it — the numbers get
worse as batch size grows.

*Throughput is a separate and stranger story* — see B5. Batching buys ~8% from
bs=1 to bs=8, with an unexplained reproducible peak at 16.

### R3 — continuous batching vs static, under real arrivals — **ACCEPTED**
**Hardware: MacBook, Apple Silicon, MPS, float16. NOT a GPU number.**
Poisson arrivals, 12–16 requests, `max_new_tokens=40`, 8 slots vs static
batch size 8, budget **p95 TTFT < 500 ms**. The static baseline is R2's policy
made arrival-aware: take whatever has arrived, run that batch to completion,
then look again.

| rate (req/s) | continuous tok/s | cont. p95 TTFT | | static tok/s | static p95 TTFT | |
|---|---|---|---|---|---|---|
| 0.2 | 6.53 | 0.160 s | OK | 6.53 | 0.041 s | OK |
| 0.3 | 9.69 | 0.095 s | OK | 9.69 | 0.162 s | OK |
| 0.4 | 14.35 | 0.137 s | OK | 14.26 | **0.501 s** | MISS |
| 0.8 | 27.12 | 0.100 s | OK | 26.43 | 1.258 s | MISS |
| 1.6 | 38.76 | **0.433 s** | OK | 35.79 | 3.598 s | MISS |
| 2.0 | 40.35 | 1.863 s | MISS | 38.47 | 3.406 s | MISS |
| 3.2 | 43.15 | 3.740 s | MISS | 41.10 | 6.314 s | MISS |

**Headline — sustainable arrival rate at p95 TTFT < 500 ms:**

```
static batching      0.3 req/s
continuous batching  1.6 req/s      5.3x
```

Bursty arrivals (18 requests, three tight bursts): continuous **44.60 tok/s**
vs static 37.77 (**+18%**), TTFT p50 **0.279 s vs 2.923 s** (10.5× better).
Bursts are where the gap is widest, which is what you would expect — a burst is
precisely a pile of requests arriving while a batch is already running.

Raw: [`r3_continuous_mps_poisson_…`](results/mac/) · [`r3_continuous_mps_burst_…`](results/mac/)

*Correctness proof:* float32 output identical to serving each request entirely
alone, across arrival patterns {all-at-once, staggered, late burst} × slot
counts {1, 2, 4}. 20 tests passed.

*Interpretation.* Raw throughput barely moves (35.79 → 38.76 tok/s at rate 1.6,
+8%). **The win is almost entirely in latency**, and that is the correct shape
for this change: continuous batching does not make the GPU faster, it stops
requests waiting on strangers. Static batching's TTFT collapses as soon as
arrivals overlap a running batch, because a request arriving one iteration late
waits for the entire batch to drain — at rate 1.6 that is 3.6 s at p95 against
continuous's 0.43 s.

*The R2 blocking tail is gone by construction, not by tuning.* R2 wasted ~18% of
all decode steps computing tokens that were discarded, with the worst request
blocked 120 steps. In R3 a decode iteration contains exactly the running set, and
every token computed for a running request is kept, so wasted steps are **0 by
construction**. That is a property of the code, not a measurement.

*Honest cost.* At rates the system is not saturated at (0.2–0.3 req/s), static
has *better* TTFT p50 — 0.034 s against 0.102 s — because continuous spends a
whole dedicated iteration on prefill while static prefills inside its batch.
Continuous batching is not free; it is a trade that only pays once arrivals
overlap.

### R4 — paged KV cache vs contiguous, at a FIXED memory budget — **ACCEPTED**
**Hardware: MacBook, Apple Silicon, MPS, float16. NOT a GPU number.**
48 MB KV budget for both designs, 56 requests of short/medium chat traffic,
`max_new_tokens=40`. `max_len` is 525 — sized for the longest prompt the system
might see (485) plus generation, which is what contiguous reservation requires
and is precisely the case it handles worst.

| | concurrent | tok/s | wall | utilisation | preemptions |
|---|---|---|---|---|---|
| R3 contiguous | 7 | 39.19 | 51.1 s | 15.8% | 0 |
| **R4 paged, bs=16** | **56** | **110.65** | **18.1 s** | **86.0%** | 8 |

**Headline: 8× the concurrent sequences at identical memory.** Utilisation 5.4×,
throughput 2.8×, wall clock 2.8× shorter.

Block size sweep (48 MB, 20 requests): bs=4 → 95.7% util / 115.26 tok/s;
**bs=16 → 82.8% / 146.37**; bs=64 → 65.3% / 138.74.

Raw: [`r4_paged_mps_20260916_201758.json`](results/mac/) (capacity) ·
[`…201457.json`](results/mac/) (equal concurrency) · [`…201410.json`](results/mac/) (block sweep)

*Interpretation.* R3's concurrency is `budget / (max_len × bytes_per_token)` —
fixed in advance, and independent of how long sequences actually are. This
workload's sequences average ~77 tokens against a 525-token reservation, so **84%
of the reserved cache never holds a token.** Paging allocates on demand, so the
same bytes hold 8× as many sequences and the throughput follows from the extra
concurrency, not from paging being faster per step.

*A prediction of mine that the measurement contradicted.* I expected paging to
cost throughput, because the gather materialises a contiguous tensor every decode
step where vLLM's fused kernel does not. **At equal concurrency (140 MB, 20
sequences both ways) paged is ~8% FASTER: 122.09 vs 112.98 tok/s.** The reason is
that R3's slot cache *also* gathers — `k[layer][slots][:, :, :read_len]` is
advanced indexing, which copies. So this measures one gather against another, and
**the cost of the gather relative to a fused kernel remains unmeasured** — see Q9.
It would be easy and wrong to report the 8% as "paging is free."

### R5 — prefix caching on a shared-document workload — **ACCEPTED**
**Hardware: MacBook, Apple Silicon, MPS, float16. NOT a GPU number.**
15 long prompts sharing one ~2000-character document (prompt lengths 466–485
tokens), `max_new_tokens=24`, block size 16, 768 blocks. A/B on the same engine
with only the cache toggled.

| prefix cache | hit rate | prefill tokens | saved | cold TTFT | hit TTFT | wall | tok/s |
|---|---|---|---|---|---|---|---|
| off | 0.0% | 7092 / 7092 | 0.0% | 1730.8 ms | — | 56.7 s | 6.34 |
| **on** | **83.8%** | **1044 / 7092** | **85.3%** | 1392.4 ms | **354.2 ms** | **19.0 s** | **18.95** |

**TTFT on a cache hit: 1730.8 → 354.2 ms (4.89×). Wall clock 2.99×.**

Raw: [`r5_prefix_mps_rag_20260917_114907.json`](results/mac/)

*Correctness proof:* 16 tests. Output identical to serving each request alone,
and identical to running with the cache disabled — including share-then-diverge
and divergence inside a block.

*Interpretation.* 85.3% of all prefill tokens are never computed, because the
document's K/V are bit-identical across every request that shares it. This is the
only rung whose win is in *avoided work* rather than better scheduling or better
packing, which is why it is the largest single speedup measured.

*The limit, measured.* A cache hit computes ~8.7% of the prompt's tokens but
still takes ~23% of a cold prefill's time. The saving lands on the projections,
the MLP and the attention *math* — but the **gather is unchanged**: every layer
still materialises the whole block table regardless of how much of it was cached.
Same structural ceiling as Q9, now visible from a second direction.

### GPU — Tesla T4, 2026-09-17 — **THE RUN THAT OVERTURNED FOUR CONCLUSIONS**
**Hardware: Tesla T4 (cc 7.5), Kaggle, float16.** Raw: `results/gpu/SUMMARY.json`.
Full analysis: [docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md).

**All 68 tests passed on CUDA**, hardware the engine was never developed on.

| | HuggingFace sdpa | **Inferno** | vLLM 0.29 |
|---|---|---|---|
| single-stream decode | 31.76 tok/s | **38.88** (+22.4%) | 174.52 (**4.49×**) |
| batched, 50 prompts | — | **643.82** (bs=32) | 2547.38 (**3.96×**) |
| TTFT p50 | 36.6 ms | **29.1 ms** | 20.7 ms |

*Controlled:* vLLM's install downgraded torch 2.10.0+cu128 → 2.13.0+cu130, so
R0/R1/R2 were re-measured under 2.13. Not cosmetic — Inferno fell 42.09 → 38.88
tok/s (−7.6%) on the torch change alone. The pre-install numbers would have
overstated Inferno by that margin.

**Four MPS conclusions died:**

| Concluded on MPS | On CUDA |
|---|---|
| Throughput peaks at batch 16, falls back at 20 (B5), reproducible 3× | **Dead.** Monotonic 1→32: 38.9 → 214 → 644 tok/s. MPS kernel artifact. |
| "The textbook batching argument does not hold here" (~8% from bs 1→8) | **Dead.** 5.5× from 1→8, 16.6× at 32. The argument holds. |
| Block size 16 clearly best (146 vs 115/139) | **Dead.** Flat across 4/16/64 on CUDA, and 16 is the *slowest*. |
| Prefix caching cuts TTFT 4.89× | **Overstated.** 1.44× on GPU — prefill costs a T4 only ~42 ms, so there is little to save. |

**What survived:** head-of-line blocking at **17.9% of decode steps on both
platforms** (a property of the policy, not the hardware); continuous batching's
advantage, which *grew* from 5.3× to **16×** (static sustains 0.25 req/s under a
500 ms p95 budget, continuous 4.0); paging's concurrency advantage, 8× → **9.1×**
(7 → 64 sequences at 48 MB, utilisation 17.3% → 97.2%, zero blocks leaked across
24–35 preemptions); and the prefix cache hit rate, **83.8% on both**.

**And the project's central hypothesis was wrong.** Throughout, the gather was
logged as the largest expected line item in the gap (Q9). R1 — the plain
contiguous engine, **no block table, no gather** — is already 4.49× behind vLLM.
The gap is fully present before paging exists, so paging cannot cause it. At
equal concurrency the paged engine was 2% *faster* than the contiguous one. The
real gap is the execution layer: CUDA graph capture, `torch.compile`, and a fused
attention kernel — things **neither Inferno nor HuggingFace does**, which is why
those two sit within 22% of each other and vLLM is 4.5× above both.

**Inferno's prefix cache is as effective as vLLM's:** +20.1% throughput against
vLLM's +19.3%, hit rate 83.8% against 86.1%, same workload. The ideas transfer.
The execution layer does not.

---

## 5. Open Questions

### Q1 — Why does R1 parity fail? **[RESOLVED 2026-09-16 — see Bugs §B3]**
Resolved to hypothesis 3: a mismatch outside the model. The model's shipped
`generation_config.json` carries `repetition_penalty: 1.1`, which HuggingFace
applies during greedy decoding. Hypotheses 1 and 2 are both dead — teacher-forced
top-1 agreement was 371/371 on mps/fp16 and 32/32 on cpu/fp32, so the
hand-written forward pass is numerically correct.

*Superseded text, kept for the record:*

### Q1 (original) — Why does R1 parity fail, and is exact parity achievable on MPS/fp16?
Three hypotheses, not yet distinguished:

1. **A logic bug that is usually harmless** — e.g. an off-by-one in the prefill
   causal mask, or a position-id edge case. *Predicts:* prefill logits differ
   from HF by far more than float16 noise, and the difference localises to one
   layer or one operation.
2. **Precision / accumulation-order differences.** Inferno's tensor shapes and
   op sequence differ from HF's, so MPS may select different kernels with
   different accumulation order — the exact mechanism already proven real in
   B1. *Predicts:* prefill logits agree to within float16 noise, top-1 agrees
   almost always, and divergence happens only where the top-1/top-2 margin is
   tiny. Under this hypothesis **exact parity on MPS/fp16 may be unachievable
   between two different implementations**, and the acceptance criterion itself
   needs revisiting.
3. **A mismatch outside the model** — chat template, EOS token set, or the
   stopping rule. *Predicts:* divergence at token 0 or a systematic length
   offset, not mid-sequence divergence.

*Distinguishing experiment:* compare prefill logits directly against HF for a
handful of prompts — max absolute difference, top-1 agreement rate, and the
top1-minus-top2 margin at the first divergent step. Then re-run parity in
**float32 on CPU**, where accumulation is far more stable. If parity holds in
fp32/CPU and fails in fp16/MPS, the cause is hypothesis 2 and the project has a
methodology decision to make, not a bug to fix.

### Q5 — Match HF's penalty, or regenerate R0 as true greedy? **[OPEN, blocking R1]**
Two legitimate resolutions to B3, and they lead to different projects:

**(a) Implement `repetition_penalty=1.1` in Inferno**, matching HF exactly.
Parity passes immediately (verified 12/12). But the baseline of record keeps a
sampling policy inherited by accident from a file in the checkpoint, and every
future comparison — including vLLM — must replicate it.

**(b) Regenerate the R0 reference with `repetition_penalty=1.0`**, i.e. actual
greedy decoding, and keep Inferno's argmax as-is. Costs two re-runs (~7 min) and
invalidates the current reference files, but makes "greedy" mean greedy, which
is what the project spec and the results files both claim.

Note that the R0 records currently assert `"greedy": true`, which is not
accurate as written. Either way the results schema should record the full
logits-processor stack, not just the sampling mode.

### Q2 — Should the parity criterion be hardware- and dtype-specific? **[RESOLVED 2026-09-16]**
**Resolved: two-tier.** float32 asserts exact batched-vs-unbatched equality at
every batch size — the gate on the batching code. float16 asserts only
finiteness; drift is reported, not asserted. Verified: float32 exact at bs
1/2/4/8/16, float16 79/80. Propagates to R3, R4, R5 and the vLLM comparison.

*Original analysis, kept for the record:*
in float32 batched output is bit-identical to unbatched for all four test
prompts; in float16 `long-01` diverges at token 67 purely because the batch had
four rows instead of two. Identical padding, identical code path.

So **"parity at every batch size" is not achievable in float16**, not because of
a defect but because differently-shaped matmuls reduce in different orders and
greedy decoding amplifies the last bit. Options, all with real costs:
1. **Correctness gate in float32, performance in float16.** Parity tests run
   fp32/CPU; throughput numbers stay fp16/MPS. Honest, but the two configurations
   are then never validated against each other, and it roughly doubles test time.
2. **Per-batch-size references.** Generate an R0 reference at each batch size.
   Cheap to run, but it quietly redefines what the test proves — it can no longer
   catch a bug that is stable across batch sizes.
3. **Replace exact token matching** with logits closeness plus a top-1 agreement
   rate over teacher-forced steps. Strictly more informative, and it is the
   diagnostic that actually found B3 — but it gives up the one-line assertion
   that makes the project legible.
This decision propagates to R3, R4, R5 and to the vLLM comparison, where CUDA
kernels will reduce differently again.
If Q1 resolves to hypothesis 2, "token-identical to R0" cannot survive a move
to rented GPUs either, since CUDA kernels will accumulate differently again.
Options to weigh: keep exact parity but pin it to fp32/CPU as the correctness
gate while fp16/MPS and GPU runs are performance-only; or replace exact token
matching with a logits-closeness criterion plus a top-1 agreement rate. This
decision affects every rung from here to R5.

### Q6 — Why is Inferno's prefill 1.8–4.6× slower than HF's eager path? **[OPEN, R1 follow-up]**
The decode loop beats HF; prefill loses badly, worst on the 475-token prompts.
Candidate causes, none measured: (1) `build_causal_mask` allocates a fresh
`q_len × kv_len` tensor on every forward instead of building once and slicing;
(2) `repeat_kv` materialises a 7× copy of K and V — at 475 positions that is a
real allocation, and HF's eager path may avoid or reuse it; (3) `cos`/`sin` are
recomputed per forward rather than cached; (4) fp32 softmax over a
`14 × 475 × 475` tensor. Rule 8 applies: profile before touching any of it.
Not on the R1 critical path — parity is accepted and the number is recorded.

### Q7 — Does the dedicated prefill iteration need replacing with chunked prefill? **[OPEN, R4/R5 follow-up]**
Measured cost: at low arrival rates static beats continuous on TTFT p50
(0.034 s vs 0.102 s) purely because prefill owns a whole iteration and stalls
every running request. Chunked prefill — splitting a prompt across several
iterations and mixing it into the decode batch — is what production engines do
instead. It interacts directly with R4's block allocator, so it is worth
revisiting there rather than retrofitting now.

### Q9 — What does the gather cost against a fused kernel? **[RESOLVED 2026-09-17 — and the answer was "not much"]**
**The hypothesis was wrong.** R1 has no block table and no gather, and is already
4.49× behind vLLM; at equal concurrency the paged engine was 2% *faster* than the
contiguous one. The gather is not the dominant cost. See §3 of
[docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) — the gap is CUDA graph capture,
compilation and fused kernels, none of which Inferno or HuggingFace does.

*Original framing, kept because it was stated confidently and repeatedly, and was
wrong:*
The equal-concurrency comparison above measures R4's gather against R3's gather,
because both materialise. It says nothing about the thing that matters for the
vLLM comparison: a fused paged-attention kernel that walks the block table inside
the kernel and never materialises at all. Two ways to get the number: build a
contiguous-attention baseline that does no gather (R2's batched cache is close),
or measure it directly against vLLM on GPU. The second is the real answer and it
is part of the final phase anyway. **This is likely the single largest line item
in the gap analysis, and it is currently unquantified.**

### Q10 — Does preemption need a watermark? **[OPEN, low priority]**
8 preemptions across 56 requests at 48 MB — low enough not to matter here. The
livelock fix guarantees progress but does not prevent thrashing: a preempted
request goes to the front of the queue and can be re-admitted as soon as its own
freed blocks make room. Under heavier pressure a reserve threshold (do not
re-admit below N free blocks) is the standard answer. Not needed at these rates,
so not built.

### Q3 — Was B2 actually Metal shader caching? **[OPEN, low priority]**
Never directly proven. Clearing the Metal cache and re-running would settle it.

### Q4 — Does the eager parity reference need regenerating for timings? **[RESOLVED 2026-09-16]**
Moot: both references were regenerated for the greedy fix, warm, on a machine
whose shader cache was long since populated. The eager TTFT p95 dropped from
203.5 ms to 78.1 ms, consistent with B2.
