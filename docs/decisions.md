# Decisions

Append-only. Each entry: what was chosen, what was rejected, why.

---

## 2026-09-15 — Process rules 6, 2 and 3 dropped

**Chosen:** No explain-and-check quiz loop (rule 6). Write as many files per
turn as a component needs (rule 2). No 150-line file cap (rule 3).

**Rejected:** Keeping them. They were making progress feel like ceremony
before a single number existed.

**Why:** The rules that actually protect the project are 5 (hypotheses before
fixes), 8-10 (no unmeasured claims, label the hardware) and 11 (no building
ahead). Those stay. The dropped three were pacing rules, and the pacing was
wrong.

---

## 2026-09-15 — R0 applies the Instruct chat template

**Chosen:** Wrap every prompt in Qwen's chat template via
`apply_chat_template(..., add_generation_prompt=True)`.

**Rejected:** Feeding raw prompt text to the model.

**Why:** `Qwen2.5-0.5B-Instruct` is post-trained to expect the template. Raw
text produces off-distribution continuations, which would make the baseline
unrepresentative of a real serving path. The cost is ~20 extra prompt tokens
per request, which is recorded in `n_prompt_tokens`. Templating happens at the
tokenizer level, so it is orthogonal to R1's reimplementation of the model.

---

## 2026-09-15 — Throughput is decode-only; prefill is excluded by construction

**Chosen:** Timestamp the first generated token with a `StoppingCriteria` that
observes but never stops. `ttft = t_first - t_start`;
`decode_tok_s = (n_new - 1) / (t_end - t_first)`.

**Rejected:** (a) `n_new / total_time`, which folds prefill into decode and
inflates the number, especially on long prompts. (b) Two separate `generate()`
calls (one with `max_new_tokens=1`) and subtracting — correct, but pays an
extra prefill per prompt and assumes the two prefills cost the same.

**Why:** The whole project rests on prefill and decode being different regimes.
A metric that averages them together cannot show that, and every later rung's
comparison would be contaminated by prompt length.

**Note:** `n_new - 1` in the numerator, not `n_new` — the first token was
produced by prefill, so it is not a decode step.

---

## 2026-09-15 — R0 stores generated token IDs, not just text

**Chosen:** Every record carries the full `token_ids` list.

**Rejected:** Storing decoded text only.

**Why:** From R1 onward the acceptance criterion is token-identical output.
Decoded text hides differences that the tokenizer round-trip smooths over
(whitespace handling, special tokens, byte-level merges). Comparing IDs is the
only assertion that actually means what we want it to mean.

---

## 2026-09-15 — Two R0 runs: `sdpa` for speed, `eager` for parity

**Chosen:** `--attn` flag. Default `sdpa` produces the headline speed number.
A second run with `--attn eager` produces the parity reference.

**Rejected:** A single `eager` run used for both.

**Why:** Fused SDPA kernels accumulate in a different order than the eager
path, so the two can disagree on a near-tie argmax and diverge. Eager is the
cleaner parity reference. But eager is also slower, and using it as the speed
baseline would understate HuggingFace — making Inferno look better for a
reason that has nothing to do with Inferno. Rule 9. So: two runs, two files,
both recorded.

---

## 2026-09-15 — dtype: float32 on CPU, float16 on MPS

**Chosen:** Device-dependent default, always recorded in the output file.

**Rejected:** float16 everywhere.

**Why:** float16 on CPU is emulated in PyTorch and pathologically slow, which
would produce a meaningless baseline. The consequence is that CPU and MPS runs
are **not** comparable to each other and their parity references are not
interchangeable. That is why dtype is written into every results file.

---

## 2026-09-15 — Results are split by hardware directory

**Chosen:** `results/mac/` and `results/gpu/`, chosen automatically from the
resolved device.

**Rejected:** One flat directory with a hardware field inside each file.

**Why:** Rule 10 says never present a Mac number as a GPU number. A field
inside a file is easy to overlook when assembling the final table; a directory
boundary is not.

---

## 2026-09-15 — R0 is batch size 1, one request at a time

**Chosen:** Sequential requests, no batching.

**Rejected:** Batching the 50 prompts for a faster baseline run.

**Why:** R0 is the floor, and the floor is what a naive implementation does.
Batching is R2. Rule 11.

---

## 2026-09-15 — sdpa/eager divergence CONFIRMED by measurement (addendum)

The two-run decision above was made on theory. It is now measured.

On the same 50 prompts, same device, same dtype, same greedy settings, sdpa
and eager produce **divergent token sequences on 10 of 50 prompts**. Earliest
divergence was at generated token 1 (`short-15`); several diverged only after
50-80 tokens, and three produced different *lengths* because one path hit EOS
and the other did not.

Both outputs are "correct". Neither is buggy. The difference is floating-point
accumulation order inside the attention kernel, amplified by greedy decoding:
once argmax flips on a near-tie, the two sequences are conditioned on different
prefixes and never re-converge.

**Consequence for every rung from R1 onward:** parity is only meaningful
against a reference produced with the *same attention implementation*. The
eager run is that reference. Comparing R1 output against the sdpa file would
produce 10 spurious failures that no amount of debugging could fix.

Parity reference: `results/mac/r0_baseline_mps_eager_20260915_131830.json`
Speed baseline:   `results/mac/r0_baseline_mps_sdpa_20260915_120143.json`

---

## 2026-09-15 — The reported R0 baseline is a warm run, and R0 is accepted

**Chosen:** `r0_baseline_mps_sdpa_20260915_145811.json` (run 2) is the R0
speed baseline of record. The first-ever run on a machine is discarded.

**Rejected:** Reporting run 1 (the first full run), which gave TTFT p95 of
297 ms.

**Why:** Run 1's tail was contaminated by one-time compilation cost that never
recurs in any later process — measured, see `bugs.md`. Reporting it would
overstate the baseline's tail latency, which would make every later rung look
better than it is for a reason unrelated to anything built here. Rule 9.

**Generalisation for the GPU runs:** discard the first run on any new machine
before recording numbers. A cold CUDA context pays kernel autotuning the same
way MPS pays shader compilation.

**R0 acceptance — PASSED:**
- Token-identical across a repeat run: **50/50**.
- Token-identical across a reversed-order run: **50/50**. Greedy decoding is
  order-independent, as it must be at batch size 1 — this also establishes the
  baseline for R2, where batching *can* change results and that will be a bug.
- decode tok/s reproduces within **2.8%** (band: ~5%).
- TTFT p50 reproduces within **4.0%** (band: ~5%).
- TTFT p95 reproduces within **0.5%** *between warm runs*; the cold run is
  excluded for the reason above.

---

## 2026-09-16 — Parity test pins the engine interface, not the cache interface

**Chosen:** `tests/test_parity.py` specifies exactly one contract:
`InfernoEngine(model_id, device, dtype, attn).generate(text, max_new_tokens)
-> list[int]`. Device, dtype and attention implementation are read out of the
reference file rather than chosen by the test.

**Rejected:** (a) Also asserting on a `KVCache` API. (b) Letting the test pick
its own device/dtype.

**Why:** (a) The cache is the one component R4 replaces wholesale with a paged
allocator. A test that pins its API now would have to be rewritten then, and a
parity test that changes between rungs is not a parity test. (b) Parity
against a reference generated under different numerics asserts nothing — fp32
CPU and fp16 MPS can legitimately produce different tokens, so the test must
inherit the reference's configuration rather than pick one.

**Also chosen:** the failing import is caught inside a session fixture rather
than at module level, so pytest still collects all parametrised cases. A
collection crash reports one error; this reports six failing prompts, which is
the shape the output will have during real debugging.

**Also chosen:** the subset is not random. `short-15` (diverged between
attention backends at generated token 1), `long-08` (token 8) and `short-09`
(different termination lengths) are prompts R0 already demonstrated are
numerically fragile. If a cache bug exists, these surface it fastest.

## 2026-09-16 — Scope of R1: the whole forward pass is hand-written

**Chosen:** Embedding, 24 decoder blocks, final norm and the tied output
projection are all written by hand. Weights are loaded from the HF checkpoint
by name. HF supplies the tokenizer and the state dict, nothing else.

**Rejected:** (a) Keeping `Qwen2MLP`/`Qwen2RMSNorm` and writing only
attention. (b) Keeping the HF model and replacing only the generation loop.

**Why:** R4 replaces how attention reaches its keys and values — gathering
across non-contiguous blocks instead of indexing one contiguous tensor. That
is surgery inside the attention function, so it must be a function we own.
(a) saves roughly twenty lines while keeping a numerics boundary we cannot
debug across: worst of both. (b) would make the final gap analysis an argument
about a scheduler wrapped around someone else's model.

**Cost accepted:** more surface for weight-loading mistakes (Qwen2 has biases
on q/k/v but not o_proj; `tie_word_embeddings=True` means there is no
`lm_head.weight` in the checkpoint). Those fail loudly and early, which is the
cheap kind of bug.

---

## 2026-09-16 — R0 regenerated as true greedy (resolution of the R1 parity bug)

**Chosen:** `bench/run_baseline.py` constructs its own `GenerationConfig`
(`repetition_penalty=1.0`, `no_repeat_ngram_size=0`, `renormalize_logits=False`)
instead of inheriting the checkpoint's. Both R0 references regenerated.
Inferno's raw argmax is unchanged. Old files moved to
`results/mac/superseded/`.

**Rejected:** implementing `repetition_penalty=1.1` inside Inferno to match
HuggingFace. Verified to work (1/12 -> 12/12 parity), but it carries a sampling
policy inherited by accident from a JSON file in the checkpoint through five
more rungs and into the vLLM comparison.

**Why:** the reference files are cheap to regenerate now and expensive to
distrust later. Both the project spec and the results schema claim greedy decoding;
after this change that claim is true.

**Consequence for the schema:** results now record the full logits-processor
stack under `config.sampling`, and `tests/test_parity.py` asserts every field
of it. "greedy: true" was not a sufficient specification of a decoding
procedure, and treating it as one cost a full debugging cycle.

---

## 2026-09-16 — Results dashboard now, live serving dashboard deferred to R3

**Chosen:** `docs/dashboard.html` - a static dashboard generated from the JSON
files in `results/`, published as an Artifact. Rung ladder, R0/R1 comparisons,
KV utilisation distribution, bug log, open questions.

**Rejected (for now):** a live serving dashboard showing queue states,
per-iteration batch composition, KV block allocation and prefix cache hits.

**Why:** the live view is the one that makes continuous batching and paged
attention legible to a viewer, but it needs a scheduler to exist. It is an R3
decision, and taking it now would not change any work done today.

**Constraint that shapes the whole page:** only measured numbers are rendered.
R2 shows its real pass/fail state; R3, R4 and R5 render as explicitly empty.
Invented placeholder numbers would be precisely the failure rules 9 and 10 exist
to prevent. The Mac/MPS hardware caveat is a banner, not a footnote.

---

## 2026-09-16 — Parity criterion becomes two-tier (resolves Q2)

**Chosen:** float32 asserts EXACT batched-vs-unbatched equality at every batch
size - that is the gate on the batching code. float16 asserts only that nothing
becomes NaN or degenerate; drift is reported, not asserted.

**Rejected:** (a) per-batch-size reference files - cheap, but the test could no
longer catch a bug that is stable across batch sizes; (b) replacing exact token
matching with logits-closeness everywhere - more informative but gives up the
one-line assertion that makes the project legible.

**Why:** measured, not assumed. In float32 batched output is bit-identical to
unbatched for all 16 test prompts at batch sizes 1/2/4/8/16. In float16 it is
not, and the cause is not a defect: `long-01` has exactly 11 pad slots in the
batch-of-2 run that passes and 11 in the batch-of-4 run that diverges at token
67. Identical padding, different batch size - differently shaped matmuls reduce
in different orders, and greedy decoding amplifies the last bit. Demanding
exactness in float16 would mean chasing a property of floating-point addition.

**What float16 CAN be held to** is finiteness. That failure mode was real
(B4a) and is fixed.

**Propagates to:** R3, R4, R5 and the vLLM comparison, where CUDA kernels will
reduce differently again.

---

## 2026-09-16 — Additive attention mask uses finfo.min / 2, not finfo.min

**Chosen:** `mask_fill_value(dtype) = torch.finfo(dtype).min / 2`.

**Rejected:** `torch.finfo(dtype).min`, which is what HuggingFace uses and what
the first implementation copied.

**Why:** in float16 the most negative finite value is -65504, so adding any
attention score beyond about -16 rounds past the end of the range to -inf.
Scores reach +/-225 by layer 8 of this model. A fully-masked row - which exists
only because of left padding - then becomes all -inf, and softmax yields NaN.
Halving leaves ~32752 of headroom while `exp(-32752 - max)` still underflows to
exactly 0 in the float32 softmax, so masked positions contribute nothing, which
is the entire requirement. See `docs/bugs.md` B4 for the full trace.

---

## 2026-09-16 — R3: FCFS, prefill owns its iteration, slot-based cache

**Chosen:** FCFS admission (only the queue head is considered); prefill runs in
its own iteration, stalling running requests for one step; a slot-based KV cache
with independent per-slot lengths.

**Rejected:** scanning past the queue head for a request that fits (starves the
head, and a starved request raises nothing); chunked prefill (the right answer
eventually, but it interacts with R4's allocator - see Q7); one cache object per
sequence (simpler, but gives up batched attention).

**Why the cache had to change:** R1/R2 carry one `length` for the whole batch,
which works only because left-padding right-aligns every sequence. A request
admitted at iteration 50 has 0 generated tokens while its batch-mates have 50 -
they are never aligned again. Per-slot lengths are forced. The consequence is
that every slot reserves `max_len` regardless of occupancy, making the waste
structural. That is exactly what R4 removes.

**Measured cost of prefill-owns-its-iteration:** at 0.3 req/s static batching
beats continuous on TTFT p50, 0.034 s vs 0.102 s. The dedicated prefill
iteration is not free and the numbers say so.

---

## 2026-09-16 — The scheduler owns no tensors

**Chosen:** `inferno/scheduler.py` handles state transitions and slot ownership
only. The engine owns every tensor. The sole thing crossing the boundary is a
slot index.

**Rejected:** a scheduler that also manages the KV cache directly.

**Why:** all three R3 failure modes are silent - a slot reused one iteration
early is corruption, a leaked slot is a hang, a starved request raises nothing.
None of them produce an exception, and none of them need a model to test. The
separation let `tests/test_scheduler.py` run in 0.01s and catch B6 before the
engine existed at all. Every other bug in this project so far was found by a
multi-minute benchmark run.

---

## 2026-09-16 — R4: preemption serves running sequences only

**Chosen:** a request is evicted only to let a RUNNING sequence grow. A waiting
request that does not fit waits; the engine declines the admission and lets the
running set drain a step.

**Rejected:** evicting on behalf of a waiting request (livelocked - see
`docs/bugs.md`); sending preempted requests to the back of the queue
(reintroduces starvation); an anti-thrash rule (adds state and a tuning knob,
and only bounds the thrashing).

**Why:** the livelock needed a waiting request to be able to evict a running
one. Removing that makes progress structural rather than tuned - a running
sequence that grows is making progress by definition.

---

## 2026-09-16 — Block size 16 (studied, not inherited)

**Chosen:** 16 as the default, measured against 4 and 64 at a fixed 48 MB
budget.

| block size | block utilisation | throughput |
|---|---|---|
| 4 | 95.7% | 115.26 tok/s |
| **16** | **82.8%** | **146.37 tok/s** |
| 64 | 65.3% | 138.74 tok/s |

**Why:** exactly the predicted trade - small blocks waste least but multiply
per-block bookkeeping and gather work; large blocks gather cheaply but strand up
to `block_size - 1` tokens per sequence. 16 wins on throughput here. It is also
vLLM's default, which is reassuring but was not the reason: the number came from
this table.

---

## 2026-09-16 — A feasibility check at startup, which is not a reservation

**Chosen:** `PagedEngine.run` refuses to start if any request's worst case
(`n_prompt + max_new_tokens`) cannot fit in the entire block pool.

**Rejected:** discovering it mid-run, which is what happened first - a confusing
`OutOfBlocks` partway through generation, far from the cause.

**Why:** paging still allocates on demand; this reserves nothing. But a sequence
whose worst case exceeds the whole pool is unservable - it will be admitted,
grow, find nothing left to evict, and fail. Better to say so before any work is
done.

---

## 2026-09-17 — R5: exact token keys, tail sharing with CoW, an LRU cached tier

**Chosen:** a block is keyed by the exact token tuple of the entire prefix up to
and including it; full blocks AND the partial tail are shared, with
copy-on-write on first write; a block at refcount zero moves to an LRU of
cached-but-reclaimable blocks rather than to the free list.

**Rejected:** a 64-bit digest (a collision is undetectable at runtime and gives
fluent wrong output); full-block-only sharing (needs no CoW at all, and gives up
the tail); freeing immediately (limits prefix caching to requests that overlap in
time).

**Why the key is the whole prefix, not the block's contents:** block 3's K/V
depend on blocks 0-2. Two sequences whose block 3 holds identical tokens after
different preceding tokens have entirely different K/V there. Keying on contents
alone is silent corruption. `test_identical_block_after_different_prefix_is_not_shared`
is the guard.

**Why the LRU tier:** it is the single biggest lever on hit rate. Without it the
fifteen RAG prompts only share when concurrent; with it they share regardless of
arrival order.

---

## 2026-09-17 — Tests run one file per process (`run_tests.sh`)

**Chosen:** each test file gets its own pytest process.

**Rejected:** `pytest tests/`, which is the obvious thing and does not work here.

**Why:** every file holds its engine in a session-scoped fixture, so one session
keeps ~5 model copies resident. On this machine that swaps to a standstill - 25
minutes at 48MB RSS and 11% CPU with swap at 6.9GB of 8GB. Separate processes
free each engine before the next loads. The three tensor-free files run first so
a logic regression surfaces in 0.05s rather than after minutes of model loading.
