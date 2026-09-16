"""R1 measurement: Inferno's own throughput, on the same 50 prompts as R0.

Metric definitions are identical to bench/run_baseline.py so the numbers are
comparable: decode throughput excludes prefill, and divides by (n_new - 1)
because the first token came out of prefill and is not a decode step.

Also records KV cache occupancy, which is the R4 motivation in numeric form.
"""
from __future__ import annotations
import argparse, json, platform, resource, statistics, sys, time
from pathlib import Path

import torch, transformers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inferno.engine import InfernoEngine
from inferno.cache import KVCache
from bench.run_baseline import percentile, peak_memory_bytes, sync


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    prompts = spec["prompts"][: args.limit]
    eng = InfernoEngine(args.model, device=args.device, dtype=args.dtype, attn="eager")
    dev = eng.device

    print(f"model={args.model} device={dev} dtype={eng.dtype} prompts={len(prompts)}\n")
    for i in range(args.warmup):
        eng.generate(prompts[0]["text"], max_new_tokens=args.max_new_tokens)
        print(f"  warmup {i+1}/{args.warmup} discarded")
    print()

    records = []
    for i, p in enumerate(prompts):
        ids = eng.encode(p["text"])
        n_prompt = ids.shape[1]
        cache = KVCache(eng.cfg.n_layers, eng.cfg.n_kv_heads, eng.cfg.head_dim,
                        n_prompt + args.max_new_tokens, eng.dtype, dev)
        sync(dev); t0 = time.perf_counter()
        logits = eng.model.forward(ids, cache)
        tok = int(logits[0, -1].argmax())
        sync(dev); t_first = time.perf_counter()
        out = [tok]
        while len(out) < args.max_new_tokens and tok not in eng.eos_ids:
            logits = eng.model.forward(torch.tensor([[tok]], device=dev), cache)
            tok = int(logits[0, -1].argmax()); out.append(tok)
        sync(dev); t_end = time.perf_counter()

        ttft = t_first - t0
        decode_s = t_end - t_first
        tok_s = (len(out) - 1) / decode_s if len(out) > 1 else float("nan")
        records.append({
            "prompt_id": p["id"], "category": p["category"],
            "n_prompt_tokens": n_prompt, "n_generated": len(out),
            "ttft_s": ttft, "decode_s": decode_s, "decode_tok_s": tok_s,
            "kv_seq_len": cache.length,
            "kv_bytes_allocated": cache.bytes_allocated(),
            "kv_bytes_used": cache.bytes_used(),
            "kv_utilisation": cache.utilisation(),
            "token_ids": out,
        })
        print(f"[{i+1:2d}/{len(prompts)}] {p['id']:<10} prompt={n_prompt:>5}tok  "
              f"ttft={ttft*1000:7.1f}ms  decode={tok_s:6.2f} tok/s  "
              f"kv_util={cache.utilisation()*100:5.1f}%")

    ttfts = [r["ttft_s"] for r in records]
    summary = {
        "decode_tok_s_aggregate": sum(r["n_generated"]-1 for r in records) /
                                  sum(r["decode_s"] for r in records),
        "ttft_p50_ms": percentile(ttfts, .50)*1000,
        "ttft_p95_ms": percentile(ttfts, .95)*1000,
        "kv_utilisation_mean": statistics.mean(r["kv_utilisation"] for r in records),
        "kv_bytes_allocated_total": sum(r["kv_bytes_allocated"] for r in records),
        "kv_bytes_used_total": sum(r["kv_bytes_used"] for r in records),
        "memory": peak_memory_bytes(dev),
        "by_category": {c: {
            "n": len([r for r in records if r["category"] == c]),
            "decode_tok_s": statistics.mean(r["decode_tok_s"] for r in records if r["category"] == c),
        } for c in sorted({r["category"] for r in records})},
    }
    payload = {"rung": "R1", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "config": {"model": args.model, "device": dev, "dtype": str(eng.dtype),
                          "attn_implementation": "eager", "engine": "inferno",
                          "max_new_tokens": args.max_new_tokens,
                          "sampling": {"do_sample": False, "num_beams": 1,
                                       "repetition_penalty": 1.0,
                                       "no_repeat_ngram_size": 0,
                                       "renormalize_logits": False},
                          "prompts_version": spec["version"], "n_prompts": len(prompts)},
               "environment": {"torch": torch.__version__,
                               "transformers": transformers.__version__,
                               "python": platform.python_version(),
                               "platform": platform.platform()},
               "summary": summary, "records": records}
    out = ROOT / "results" / ("gpu" if dev == "cuda" else "mac") / \
        f"r1_inferno_{dev}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n{'='*62}\nR1 INFERNO  [{dev} / {eng.dtype} / eager]\n{'='*62}")
    print(f"  decode throughput   {summary['decode_tok_s_aggregate']:8.2f} tok/s")
    print(f"  TTFT p50            {summary['ttft_p50_ms']:8.1f} ms")
    print(f"  TTFT p95            {summary['ttft_p95_ms']:8.1f} ms")
    print(f"  KV utilisation      {summary['kv_utilisation_mean']*100:8.1f} %  "
          f"({summary['kv_bytes_used_total']/1e6:.1f} MB used of "
          f"{summary['kv_bytes_allocated_total']/1e6:.1f} MB allocated)")
    for c, s in summary["by_category"].items():
        print(f"    {c:<8} n={s['n']:<3} {s['decode_tok_s']:6.2f} tok/s")
    print(f"\n  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
