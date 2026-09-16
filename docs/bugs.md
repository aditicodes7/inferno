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
