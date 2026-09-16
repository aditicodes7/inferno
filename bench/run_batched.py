"""R2 measurement: throughput vs batch size, and head-of-line blocking.

Two numbers per request, and the gap between them is the point:

  ready_at     the decode step at which THIS request produced its last token.
  delivered_at the decode step at which the whole batch finished - which is
               when the caller can actually have the result, because a static
               batch runs until every sequence in it is done.

blocked_steps = delivered_at - ready_at is head-of-line blocking measured
directly. It is not a tuning problem: a batch is fixed for its lifetime, so a
request that needed 8 tokens waits for one that needed 128. Removing it is what
R3 is for.

Prompts are NOT sorted by length. Sorting would flatten the tail and hide the
effect this rung exists to demonstrate.
"""
from __future__ import annotations
import argparse, json, platform, statistics, sys, time
from pathlib import Path

import torch, transformers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inferno.batch_engine import BatchEngine
from bench.run_baseline import percentile, peak_memory_bytes, sync


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    prompts = spec["prompts"][: args.limit]
    sizes = [int(b) for b in args.batch_sizes.split(",")]

    eng = BatchEngine("Qwen/Qwen2.5-0.5B-Instruct", device=args.device,
                      dtype=args.dtype, attn="eager")
    dev = eng.device
    for _ in range(args.warmup):
        eng.generate_batch([prompts[0]["text"]] * 2, max_new_tokens=args.max_new_tokens)

    results = {}
    for bs in sizes:
        recs, t_total, gen_total = [], 0.0, 0
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            sync(dev); t0 = time.perf_counter()
            outs, st = eng.generate_batch([p["text"] for p in chunk],
                                          max_new_tokens=args.max_new_tokens,
                                          return_stats=True)
            sync(dev); dt = time.perf_counter() - t0
            t_total += dt
            per_step = dt / st["steps_run"]
            for p, o, fin, blk, plen in zip(chunk, outs, st["finished_at"],
                                            st["blocked_steps"], st["prompt_lens"]):
                gen_total += len(o)
                recs.append({
                    "prompt_id": p["id"], "category": p["category"],
                    "prompt_tokens": plen, "n_generated": len(o),
                    "ready_at_step": fin, "delivered_at_step": st["steps_run"],
                    "blocked_steps": blk,
                    "ready_s": fin * per_step, "delivered_s": dt,
                    "blocked_s": blk * per_step,
                    "batch_padded_width": st["prompt_len_padded"],
                })

        blocked = [r["blocked_steps"] for r in recs]
        wasted = sum(blocked)
        results[bs] = {
            "batch_size": bs,
            "throughput_tok_s": gen_total / t_total,
            "wall_clock_s": t_total,
            "tokens_generated": gen_total,
            "latency_p50_s": percentile([r["delivered_s"] for r in recs], .50),
            "latency_p95_s": percentile([r["delivered_s"] for r in recs], .95),
            "blocked_steps_mean": statistics.mean(blocked),
            "blocked_steps_max": max(blocked),
            "blocked_steps_total": wasted,
            "wasted_step_fraction": wasted / sum(r["delivered_at_step"] for r in recs),
            "records": recs,
        }
        r = results[bs]
        print(f"bs={bs:<3} {r['throughput_tok_s']:7.2f} tok/s  "
              f"wall {r['wall_clock_s']:6.1f}s  "
              f"lat p50 {r['latency_p50_s']:5.2f}s p95 {r['latency_p95_s']:5.2f}s  "
              f"blocked mean {r['blocked_steps_mean']:6.1f} max {r['blocked_steps_max']:3d} "
              f"({r['wasted_step_fraction']*100:4.1f}% of steps wasted)")

    payload = {
        "rung": "R2", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "device": dev,
                   "dtype": str(eng.dtype), "attn_implementation": "eager",
                   "engine": "inferno-batched", "batching": "static, left-padded",
                   "max_new_tokens": args.max_new_tokens,
                   "sampling": {"do_sample": False, "num_beams": 1,
                                "repetition_penalty": 1.0},
                   "prompts_version": spec["version"], "n_prompts": len(prompts),
                   "sorted_by_length": False},
        "environment": {"torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "python": platform.python_version(),
                        "platform": platform.platform()},
        "memory": peak_memory_bytes(dev),
        "by_batch_size": results,
    }
    out = ROOT / "results" / ("gpu" if dev == "cuda" else "mac") / \
        f"r2_batched_{dev}_bs{'-'.join(map(str,sizes))}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
