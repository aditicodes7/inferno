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
distrust later. Both `CLAUDE.md` and the results schema claim greedy decoding;
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
