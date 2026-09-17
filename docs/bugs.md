# Bug / anomaly log

Written BEFORE fixing, every time.

---

## 2026-09-15 — R0 TTFT p95 not reproducible (297ms vs 88ms)

**Symptom:**
First full R0 run reported TTFT p95 = 297.3 ms, with five requests above
250 ms (`medium-00` 293ms, `medium-01` 303ms, `medium-06` 320ms, `long-00`
301ms, `long-01` 258ms). TTFT did not increase monotonically with prompt
length: a 64-token prompt took 320 ms while a 73-token prompt took 21 ms and
a 485-token prompt took 301 ms.

**Expected:**
TTFT is prefill time. Prefill cost should rise with prompt length, so TTFT
should be roughly monotonic in `n_prompt_tokens`, and p95 should reproduce
across runs within the ~5% R0 acceptance band.

**Hypotheses:**
1. Shape-triggered kernel compilation on MPS — an unseen sequence length
   forces a compile, and the cost lands on whichever request hits that shape
   first. Predicts: slow requests are the first occurrence of their shape,
   and the effect moves when run order changes.
2. Insufficient warm-up — 3 warm-ups on a single prompt shape does not warm
   the shapes used later in the run.
3. Host-side noise — thermal throttling, background process, memory pressure
   during that particular run.

**Test that distinguishes them:**
Re-run identically (isolates noise), then re-run with `--reverse` so each
prompt occupies a different position (isolates position-in-run from
prompt-identity). Warm-up pinned to a fixed prompt in both orderings so that
ordering is the only changed variable.

**Actual result:**

|                        | run1 fwd (first ever) | run2 fwd | run3 reversed |
|---|---|---|---|
| decode tok/s aggregate | 34.77 | 35.75 (+2.8%) | 35.63 (+2.5%) |
| TTFT p50               | 61.9 ms | 61.1 ms (-1.3%) | 64.3 ms (+4.0%) |
| TTFT p95               | 297.3 ms | 87.8 ms (-70.5%) | 87.4 ms (-70.6%) |
| requests over 150 ms   | 5 | **0** | **0** |

The outliers did **not** follow the prompt: `medium-06` went 320 → 73 → 86 ms.
They did **not** follow position: they sat at positions 20, 21, 26, 35, 36 in
run1, and run3 placed those same prompts at 14, 13, 23, 28, 29 with no
outliers anywhere. They appeared only in the first run executed on this
machine, and never again in any subsequent process.

Within-run warm-up drift is visible and small: run1 pos<5 mean 36.8 ms vs
pos>=5 mean 86.0 ms, but run3 (reversed) shows pos<5 at 81.9 ms and pos>=5 at
55.1 ms — i.e. that gap tracks which prompts landed early, not position.

**Actual cause:**
Hypothesis 3 is ruled out — the effect is perfectly reproducible in its
absence (two independent later runs, zero outliers each). Hypothesis 2 is
ruled out as stated — more warm-up within a process would not help, because
run2 used the same 3 warm-ups and had no outliers at all.

Hypothesis 1 survives, in a refined form: the compilation cost is real but the
cache is **persistent across processes**, not per-process. macOS keeps a
Metal shader/pipeline cache on disk, so the first process to encounter a given
shape pays the compile and every later process reuses it.

Stated honestly: the data proves *"first execution on this machine only,
independent of prompt and of position"*. Persistent shader caching is the
leading explanation consistent with that, but it has not been directly proven.
A direct test would be to clear the Metal cache and re-run — not done yet.

**Fix:**
None applied to the code. The methodology changes instead: the reported R0
baseline is a warm run, not the first-ever run. See `decisions.md`.

**Why it worked:**
n/a — nothing was fixed. What the investigation bought was the knowledge that
**the first benchmark run on any new machine is untrustworthy for tail
latency**. This matters directly for the GPU runs later: a cold CUDA context
pays kernel autotuning the same way, so the first run on rented hardware must
be discarded rather than reported.

---

## 2026-09-16 — R1 parity: 1/50, root cause = repetition_penalty

Full write-up with all four distinguishing experiments is in `PROJECT_LOG.md`
§B3. Summary:

Symptom:      Hand-written engine produced fluent text, never crashed, matched
              the reference for up to 34 tokens, then diverged. 1/50 exact.
Expected:     Token-identical output to R0.
Hypotheses:   1) harmless-looking logic bug (mask off-by-one, position edge
                 case) 2) precision / accumulation order 3) mismatch outside
                 the model (template, EOS, sampling rule)
Test:         Teacher-forced argmax agreement on raw logits, mps/fp16 and
              cpu/fp32; determinism check; cache-capacity sensitivity check.
Actual cause: `generation_config.json` shipped with Qwen2.5-0.5B-Instruct sets
              `repetition_penalty: 1.1`. HF applies it in GREEDY decoding -
              it is not gated on do_sample. Inferno took raw argmax.
              Teacher-forced agreement was 371/371 (mps) and 32/32 (cpu):
              the forward pass was correct all along.
Fix:          NOT YET APPLIED - two legitimate options, see PROJECT_LOG Q5.
Why it worked: n/a. Confirmed by toggling the penalty alone: 1/12 -> 12/12.

---

## 2026-09-16 — R2 batch parity fails at batch_size=4 [RESOLVED]

Symptom:      batch 1 PASS, batch 2 PASS, batch 4 FAIL. Two failure shapes:
              (a) medium-00/01 emit token id 0 forever; (b) long-01 diverges
              at token 67 after 67 exact matches.
Expected:     identical tokens at every batch size.
Hypotheses:   1) fully-masked rows -> NaN  2) wrong position ids for padded
              sequences  3) batched matmul reduction order.
Test:         (i) compare each prompt batched-of-4 vs alone, same code path,
              both dtypes; (ii) instrument every tensor inside the layer.
Result (i):   float32 - all four IDENTICAL. So the batching LOGIC is correct:
              mask, position ids, cache offsets and left-padding are all right.
              float16 - (a) and (b) are two DIFFERENT effects. long-01 has
              exactly 11 pad slots in the batch-of-2 run that passed and 11 in
              the batch-of-4 run that diverged; padding identical, batch size
              not. (b) is B1 one level up: not a defect, see Q2.
Result (ii):  first nonfinite tensor is SOFTMAX at layer 8, on padding rows.

Actual cause: the additive mask used `torch.finfo(dtype).min`. float16's most
              negative finite value is -65504, so adding any score beyond about
              -16 rounds past the end of the range to -inf. Scores reach +/-225
              by layer 8. A fully-masked row then becomes ALL -inf, and softmax
              computes exp(-inf - (-inf)) = NaN.

              Fully-masked rows exist only because of LEFT padding: a leading
              pad query row has no real key at or before it. 398 such rows in
              one 485-wide batch.

              The NaN then escaped into REAL tokens one layer later: pad K/V
              live in the same cache, and a masked weight is exactly 0 after
              the float32 softmax - but 0 * NaN = NaN in the value matmul.
              Masking does not protect a real query from a NaN pad value.

Fix:          `mask_fill_value(dtype) = torch.finfo(dtype).min / 2`.

Why it worked: a masked position must contribute exactly zero weight, which
              requires a value that is very negative - not maximally negative.
              Halving leaves ~32752 of headroom for the score while
              exp(-32752 - max) still underflows to exactly 0 in the float32
              softmax, so masked positions still contribute nothing.
              finfo.min is the obvious choice and is precisely the one value
              that cannot absorb an addition.

TWO MISTAKES WORTH KEEPING:
  1. The probe that misled me. I tested `(-5) + finfo.min`, saw -65504.0 and
     concluded "does not overflow". -5 is within half a ULP of the endpoint so
     it rounds back; -50 and beyond do not. A probe value chosen without
     thinking about the real magnitude of the quantity produced a confident
     wrong conclusion that cost an entire wrong fix.
  2. The wrong fix. Zeroing the residual stream at padding positions is
     provably output-neutral and sounded principled, but it addressed
     ACTIVATION growth when the overflow was in the MASK ADDITION. It moved
     the first NaN from layer 1 to layer 8 and made long-01 worse. A fix that
     moves a symptom without removing it is evidence the mechanism is still
     wrong - and reverting it was the right call, not patching on top.

---

## 2026-09-16 — R2 batch parity fails at batch_size=4, passes at 1 and 2

Symptom:      batch 1 PASS, batch 2 PASS, batch 4 FAIL. Two distinct failure
              shapes in the same batch:
                (a) medium-00 and medium-01 diverge at token 0 and emit
                    token id 0 repeatedly: got [0, 0, 0] vs want [95456, 0, 6771].
                    Catastrophic, not drift.
                (b) long-01 diverges at token 67 after 67 EXACT matches, with a
                    plausible continuation. Looks like ordinary numeric drift.
Expected:     identical tokens at every batch size.
Key context:  MIXED[:8] chunked by 4 puts medium-00 (87 tok) and medium-01
              (73 tok) in the SAME batch as long-00 (485) and long-01 (474).
              At batch 2 the chunks are [m0,m1] and [l0,l1] - similar lengths,
              almost no padding. The failure appears exactly when the padding
              ratio becomes large (~400 pad slots on a 485-wide batch).
Hypotheses:   1) Fully-masked rows produce NaN, which then contaminates real
                 positions. A leading pad query row can see no keys at all
                 (every key at or before it is also padding), so its whole
                 score row is masked. Two sub-mechanisms worth separating:
                 whether softmax over an all-masked row yields NaN at all, and
                 if so whether NaN reaches real tokens - note that a masked
                 weight is 0 but 0 * NaN = NaN in the value matmul, and pad
                 rows do get written into the KV cache.
              2) Position IDs wrong for left-padded sequences - would give
                 wrong-but-finite tokens, and should not depend on how much
                 padding there is, only on whether there is any.
              3) Batched matmul reduction order - a real effect already proven
                 in B1, but it produces plausible drift, not repeated token 0.
                 This could explain (b) while being irrelevant to (a).
Test that distinguishes them:
              Run one padded batch prefill and inspect the intermediate tensors
              directly: check logits and per-layer hidden states for NaN;
              check whether the additive mask value overflows to -inf in float16
              (finfo.min plus a negative score is outside float16 range);
              and check whether the real rows are already corrupted at layer 0
              or only after several layers. Separately, re-run batch 4 in
              float32 - if (b) survives but (a) disappears, they are two
              different bugs.
Actual cause: CONFIRMED IN PART - float16 range, triggered by fully-masked
              padding rows. Measured on the medium-00 + long-00 pair (87 real
              tokens vs 485, so 398 pad slots on the medium sequence):

                query rows that can see NO key at all : 398, all padding rows
                mask contains -inf                    : False
                (-5) + finfo.min in fp16              : -65504.0, NOT -inf
                first layer with ANY NaN              : 1
                first layer with NaN in REAL rows     : 2
                same batch in float32                 : no NaN at any layer,
                                                        logits clean, argmax agrees

              The hypothesised first step was WRONG. Softmax over a fully
              masked row does not produce NaN: every entry is finfo.min, so
              after the float32 softmax the row is a finite UNIFORM
              distribution (verified: sums to 1.0, no NaN). And finfo.min does
              not overflow to -inf when a score is added to it - fp16
              saturates at -65504.

              What is confirmed:
                - NaN originates in PADDING rows, at layer 1, and pad-row
                  values at layer 0 are small and finite (max |x| = 1.5).
                  No inf appears in any layer OUTPUT, so the overflow happens
                  in an intermediate INSIDE layer 1 and is consumed into a NaN
                  before reaching the layer output. Which intermediate is not
                  yet pinned.
                - It reaches REAL tokens one layer later. Pad K/V are written
                  to the shared cache, and a masked weight is exactly 0 after
                  the float32 softmax - but 0 * NaN = NaN in the value matmul,
                  so masking does not protect a real query from a NaN pad
                  value. This is the contamination path.
                - It is float16-specific. float32 is clean end to end.
                - Fully-masked rows only exist because of LEFT padding: a
                  leading pad query row has no key at or before it that is
                  real. This is structural to the padding scheme, not an
                  edge case.

              SEPARATION TEST (2026-09-16): compare each prompt batched-of-4
              against the same engine running it alone - same code path both
              ways, so only batch COMPOSITION changes.

                float16   medium-00  diverge@0   (all-zeros = NaN)
                          medium-01  diverge@0   (all-zeros = NaN)
                          long-00    IDENTICAL   (485 tok, ZERO padding)
                          long-01    diverge@67
                float32   all four   IDENTICAL

              THE BATCHING LOGIC IS CORRECT. In float32, batched output is
              bit-identical to unbatched for every prompt. The mask, the
              position ids, the cache offsets and the left-padding scheme are
              all right. There is no batching bug.

              There are TWO float16 effects, and they are different:
                (a) NaN from fully-masked padding rows - catastrophic, scales
                    with padding. medium-* carry ~400 pad slots and die at
                    token 0; long-00 carries ZERO padding and is untouched.
                (b) Near-tie argmax flips from batch-shape-dependent reduction
                    order - subtle drift, nothing to do with padding.
                    Decisive evidence: long-01 has exactly 11 pad slots in the
                    batch-of-2 run (which PASSED) and exactly 11 in the
                    batch-of-4 run (which diverged at 67). Padding identical,
                    batch size different. That is B1 again, one level up.

              (b) is not fixable - it is the same float-associativity property
              proven in B1. Its existence means "parity at every batch size"
              may be unachievable in float16 between differently-shaped
              matmuls. That is a criterion decision, not a bug. See Q2.
Fix:          NOT APPLIED - (a) awaits a decision; (b) is a property.
Why it worked: n/a


---

## 2026-09-16 — R4 preemption livelock: two requests evict each other forever

Symptom:      `pytest tests/test_paged_parity.py` produced no output and did
              not terminate. No error, no crash, no progress. Killed at 600s.
Expected:     the memory-pressure test to preempt a few times and finish.
Hypotheses:   1) livelock in the admission/preemption loop  2) the gather is
              simply far more expensive than expected in fp32  3) a hang inside
              the model forward.
Test that distinguishes them:
              replay the ADMISSION POLICY alone - BlockManager + Scheduler, no
              model, no tensors. If it cycles, it is (1) and nothing to do with
              the engine. Took under a second.
Actual cause: CONFIRMED (1). With an 8-block pool and block_size 16:

                short-00 needs 3 blocks, medium-00 needs 6.

                iter 0  admit short-00               free=5
                iter 1  admit medium-00, evict short-00   free=2
                iter 2  admit short-00, evict medium-00   free=5
                iter 3  admit medium-00, evict short-00   free=2   ... forever

              Two decisions that are each individually defensible compose into
              a cycle:

                * `preempt()` returns the victim to the FRONT of the queue, so
                  a preempted request is not starved by everything behind it;
                * FCFS admission only ever considers the queue HEAD.

              Together: the victim becomes the head, is immediately re-admitted,
              and immediately forces the next head to evict it again. Neither
              rule is wrong on its own. The bug is in their composition, which
              is why reviewing either one in isolation would not have found it.

              Note this is NOT starvation - the failure mode predicted for R3.
              Both requests are repeatedly ADMITTED. They just never keep their
              memory long enough to produce a token. Forward progress is zero
              while the system looks perfectly busy.
Fix:          PREEMPTION NOW ONLY EVER SERVES A RUNNING SEQUENCE THAT CANNOT
              GROW. A waiting request that does not fit simply waits; the
              engine declines the admission and lets the running set drain a
              step instead.
Why it worked: the cycle needed a waiting request to be able to evict a running
              one. Remove that and only a running sequence can trigger
              eviction - and a running sequence that grows is making progress
              by definition, so the cycle cannot form. This is structural, not
              damping: there is no tuning constant and no thrash budget.
              Rejected alternatives: sending the victim to the BACK of the
              queue (breaks the cycle but reintroduces the starvation that
              front-insertion existed to prevent), and an anti-thrash rule
              (keeps both properties but adds state, a tuning knob, and only
              bounds the thrashing rather than removing it).

              Verified by replaying the admission policy with no model:
              previously an endless 2-cycle, now terminates in 8 iterations
              with all blocks returned.

---

## 2026-09-16 — R4 benchmark reported 126% cache utilisation

Symptom:      the R3 baseline row printed `blk util 126.1%`.
Expected:     a fraction, so at most 100%.
Actual cause: the metric was `total tokens across all served requests /
              (n_slots * max_len)`. That is only correct when every request
              occupies a slot at the same time. On the capacity run - 56
              requests through 7 slots - it over-counted by roughly 8x.
Fix:          measure how full a slot is WHILE OCCUPIED: mean over requests of
              (n_prompt + n_generated) / max_len. Directly comparable to R4's
              block utilisation.
Why it worked: it stopped conflating throughput-over-time with
              occupancy-at-an-instant.
Corrected:    R3 utilisation 39.0% -> 15.8%. The error FLATTERED the baseline,
              so the real R4 advantage is larger than first reported
              (15.8% vs 86.0%, not 39.0% vs 86.0%).

WORTH KEEPING: this is the third error this session of the same shape - a
constant or formula that is right for the case in mind, applied to a case where
it is not (see also the -5 mask probe in B4, and the 22-block pool that was
sized from max_new_tokens when the prompts stop at EOS long before). What caught
it was that the number was IMPOSSIBLE rather than merely wrong. Had it printed
84% it would have gone straight into the log unchallenged. Prefer metrics that
can be visibly out of range over metrics that degrade quietly.
