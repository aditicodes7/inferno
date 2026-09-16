# Inferno — Project Log

Master log. Append-only: entries are never rewritten or removed, only added to.
Maintained automatically as part of finishing each chunk of work.

Phases map to the rung plan in `CLAUDE.md`: R0 baseline, R1 own attention + KV
cache, R2 static batching, R3 continuous batching, R4 paged KV cache,
R5 prefix caching.

**Status: R2 in progress — batch parity fails at batch_size=4 (Bugs §B4). R1 complete and accepted (parity 50/50).**

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
later. `CLAUDE.md` and the results schema both claim greedy; now that is true.
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

Published: https://claude.ai/artifact/RpzegYya34hiJMEjD7XWiX

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
differently-shaped matmuls.* That is a criterion decision, not a defect — see Q2,
which this promotes from downgraded back to blocking.

*Build note:* one self-inflicted error before this — the patch that added the
padded-mask branch dropped the `cos, sin = self.rope(...)` call, giving a clean
`NameError` on the first run. Mentioned only because it is the contrast case:
a structural mistake fails loudly and instantly, which is the cheap kind.

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
is what `CLAUDE.md` and the results files both claim.

Note that the R0 records currently assert `"greedy": true`, which is not
accurate as written. Either way the results schema should record the full
logits-processor stack, not just the sampling mode.

### Q2 — Should the parity criterion be hardware- and dtype-specific? **[OPEN, BLOCKING R2]**
Promoted back to blocking on 2026-09-16 by B4(b). The evidence is now direct:
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

### Q3 — Was B2 actually Metal shader caching? **[OPEN, low priority]**
Never directly proven. Clearing the Metal cache and re-running would settle it.

### Q4 — Does the eager parity reference need regenerating for timings? **[RESOLVED 2026-09-16]**
Moot: both references were regenerated for the greedy fix, warm, on a machine
whose shader cache was long since populated. The eager TTFT p95 dropped from
203.5 ms to 78.1 ms, consistent with B2.
