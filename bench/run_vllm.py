"""vLLM on the same 50 prompts, for the gap analysis.

NOT YET RUN. Written on a machine with no CUDA, so it is untested against a real
vLLM install - treat the first GPU run as a debugging run, not a measurement.

Metric definitions match bench/run_baseline.py exactly, because a comparison
against differently-defined metrics is worse than no comparison:
  * decode throughput EXCLUDES prefill and divides by (n_new - 1), since the
    first token came out of prefill and is not a decode step;
  * TTFT is measured to the first token, per request.

Parity note: vLLM's kernels will NOT reproduce Inferno's tokens exactly, and
that is expected rather than a defect - it is the same float-associativity
property measured in PROJECT_LOG.md B1, where HuggingFace's own sdpa and eager
paths disagreed on 10 of 50 prompts. So token agreement is REPORTED as a
percentage, never asserted.
"""
from __future__ import annotations
import argparse, json, platform, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.run_baseline import percentile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="disable vLLM's prefix cache. REQUIRED for a clean "
                         "sequential measurement: the max_tokens=1 TTFT probe "
                         "populates the cache, so the full run that follows "
                         "gets a prefix hit and the subtraction undercounts "
                         "decode time. Left on, the first GPU run reported "
                         "167 tok/s against vLLM's own internal 79 tok/s.")
    ap.add_argument("--reference", default=None,
                    help="an Inferno results file to report token agreement against")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams          # imported late: CUDA only
    import torch

    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    prompts = spec["prompts"][: args.limit]

    # disable_log_stats defaults to True in the offline LLM class, and that
    # suppresses per-request metrics entirely - the first run reported nan for
    # every latency figure.
    llm = LLM(model=args.model, dtype="float16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=False, disable_log_stats=False,
              enable_prefix_caching=not args.no_prefix_cache)
    tok = llm.get_tokenizer()

    # Same chat template as every other rung (docs/decisions.md).
    texts = [tok.apply_chat_template([{"role": "user", "content": p["text"]}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]

    # Greedy, and repetition_penalty EXPLICITLY 1.0 - the checkpoint ships 1.1
    # and inheriting it silently cost a full debugging cycle (B3).
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                        repetition_penalty=1.0, max_tokens=args.max_new_tokens)

    for _ in range(2):                            # warm up (B2)
        llm.generate(texts[:2], sp, use_tqdm=False)

    # (a) BATCHED: everything at once. This is vLLM operating the way it is
    #     meant to, and the number to compare against Inferno's R2/R4 throughput.
    t0 = time.perf_counter()
    outs = llm.generate(texts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0

    # (b) SEQUENTIAL: one prompt at a time, which is how R0 and R1 were measured.
    #     TTFT comes from a max_tokens=1 run and decode from the difference, so
    #     the definitions match bench/run_baseline.py exactly instead of relying
    #     on vLLM's internal metrics being populated.
    sp1 = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                         repetition_penalty=1.0, max_tokens=1)
    seq_ttft, seq_decode_tok_s = [], []
    for text in texts:
        t = time.perf_counter()
        llm.generate([text], sp1, use_tqdm=False)
        ttft1 = time.perf_counter() - t
        t = time.perf_counter()
        o = llm.generate([text], sp, use_tqdm=False)[0]
        full = time.perf_counter() - t
        n = len(o.outputs[0].token_ids)
        seq_ttft.append(ttft1)
        if n > 1 and full > ttft1:
            seq_decode_tok_s.append((n - 1) / (full - ttft1))

    def _ts(m, *names):
        """vLLM renames these between versions (first_token_time became
        first_token_ts in 0.29). The sequential pass above is the number we
        actually compare on, so this is best-effort and must never be fatal."""
        for n in names:
            v = getattr(m, n, None)
            if v is not None:
                return v
        return None

    records, ttfts = [], []
    for p, o in zip(prompts, outs):
        out = o.outputs[0]
        ids = list(out.token_ids)
        m = getattr(o, "metrics", None)
        ttft = decode_s = None
        if m is not None:
            arrival = _ts(m, "arrival_time", "arrival_ts")
            first = _ts(m, "first_token_ts", "first_token_time")
            last = _ts(m, "last_token_ts", "finished_time", "finished_ts")
            if arrival is not None and first is not None:
                ttft = first - arrival
            if first is not None and last is not None:
                decode_s = last - first
        if ttft is not None:
            ttfts.append(ttft)
        records.append({
            "prompt_id": p["id"], "category": p["category"],
            "n_prompt_tokens": len(o.prompt_token_ids), "n_generated": len(ids),
            "ttft_s": ttft, "decode_s": decode_s,
            "decode_tok_s": ((len(ids) - 1) / decode_s
                             if decode_s and len(ids) > 1 else None),
            "token_ids": ids,
        })

    gen = sum(r["n_generated"] - 1 for r in records)
    dec = sum(r["decode_s"] for r in records if r["decode_s"]) or 0.0
    summary = {
        "sequential_decode_tok_s": st.mean(seq_decode_tok_s) if seq_decode_tok_s else None,
        "sequential_ttft_p50_ms": percentile(seq_ttft, .50) * 1000,
        "sequential_ttft_p95_ms": percentile(seq_ttft, .95) * 1000,
        "decode_tok_s_aggregate": gen / dec if dec else None,
        "throughput_tok_s_wall": sum(r["n_generated"] for r in records) / wall,
        "ttft_p50_ms": percentile(ttfts, .50) * 1000 if ttfts else None,
        "ttft_p95_ms": percentile(ttfts, .95) * 1000 if ttfts else None,
        "vllm_metrics_available": bool(ttfts),
        "wall_clock_s": wall,
        "n_prompts": len(records),
    }

    if args.reference:
        ref = {r["prompt_id"]: r["token_ids"]
               for r in json.loads(Path(args.reference).read_text())["records"]}
        same = sum(1 for r in records if ref.get(r["prompt_id"]) == r["token_ids"])
        firsts = [next((i for i, (a, b) in enumerate(
                      zip(r["token_ids"], ref.get(r["prompt_id"], []))) if a != b), None)
                  for r in records if r["prompt_id"] in ref]
        diverged = [f for f in firsts if f is not None]
        summary["token_agreement"] = {
            "identical": same, "of": len(records),
            "median_first_divergence": st.median(diverged) if diverged else None,
            "note": "expected to be well below 100% - different kernels reduce in "
                    "different orders and greedy decoding amplifies the last bit "
                    "(PROJECT_LOG.md B1). Reported, never asserted.",
        }

    payload = {
        "rung": "vllm", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"model": args.model, "engine": "vllm", "dtype": "float16",
                   "prefix_caching": not args.no_prefix_cache,
                   "max_new_tokens": args.max_new_tokens, "greedy": True,
                   "repetition_penalty": 1.0,
                   "gpu_memory_utilization": args.gpu_memory_utilization,
                   "prompts_version": spec["version"]},
        "environment": {"platform": platform.platform(),
                        "torch": torch.__version__,
                        "cuda_device": torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None},
        "summary": summary, "records": records,
    }
    out = ROOT / "results" / "gpu" / f"vllm_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n{'='*62}\nvLLM  [{payload['environment']['cuda_device']}]"
          f"  prefix_cache={'off' if args.no_prefix_cache else 'ON'}\n{'='*62}")
    print(f"  BATCHED (all 50 at once, vLLM as intended)")
    print(f"    wall throughput   {summary['throughput_tok_s_wall']:8.2f} tok/s")
    print(f"  SEQUENTIAL (one at a time - comparable to R0/R1)")
    print(f"    decode            {summary['sequential_decode_tok_s'] or float('nan'):8.2f} tok/s")
    print(f"    TTFT p50          {summary['sequential_ttft_p50_ms']:8.1f} ms")
    print(f"    TTFT p95          {summary['sequential_ttft_p95_ms']:8.1f} ms")
    if summary["decode_tok_s_aggregate"]:
        print(f"  (vllm internal metrics: {summary['decode_tok_s_aggregate']:.2f} tok/s decode)")
    if "token_agreement" in summary:
        a = summary["token_agreement"]
        print(f"  token agreement     {a['identical']}/{a['of']} identical to Inferno")
    print(f"\n  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
