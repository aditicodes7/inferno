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
    ap.add_argument("--reference", default=None,
                    help="an Inferno results file to report token agreement against")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams          # imported late: CUDA only
    import torch

    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    prompts = spec["prompts"][: args.limit]

    llm = LLM(model=args.model, dtype="float16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=False)
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

    t0 = time.perf_counter()
    outs = llm.generate(texts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0

    records, ttfts = [], []
    for p, o in zip(prompts, outs):
        out = o.outputs[0]
        ids = list(out.token_ids)
        m = o.metrics
        ttft = (m.first_token_time - m.arrival_time) if m and m.first_token_time else None
        decode_s = ((m.finished_time - m.first_token_time)
                    if m and m.finished_time and m.first_token_time else None)
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
    dec = sum(r["decode_s"] for r in records if r["decode_s"])
    summary = {
        "decode_tok_s_aggregate": gen / dec if dec else None,
        "throughput_tok_s_wall": sum(r["n_generated"] for r in records) / wall,
        "ttft_p50_ms": percentile(ttfts, .50) * 1000 if ttfts else None,
        "ttft_p95_ms": percentile(ttfts, .95) * 1000 if ttfts else None,
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

    print(f"\n{'='*62}\nvLLM  [{payload['environment']['cuda_device']}]\n{'='*62}")
    print(f"  decode throughput   {summary['decode_tok_s_aggregate'] or float('nan'):8.2f} tok/s")
    print(f"  wall throughput     {summary['throughput_tok_s_wall']:8.2f} tok/s")
    print(f"  TTFT p50            {summary['ttft_p50_ms'] or float('nan'):8.1f} ms")
    print(f"  TTFT p95            {summary['ttft_p95_ms'] or float('nan'):8.1f} ms")
    if "token_agreement" in summary:
        a = summary["token_agreement"]
        print(f"  token agreement     {a['identical']}/{a['of']} identical to Inferno")
    print(f"\n  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
