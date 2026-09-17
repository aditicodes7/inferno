"""R5 measurement: prefix cache hit rate and TTFT reduction.

The workload is the one prefix caching exists for and the one this benchmark set
was built with from the start: fifteen long prompts that share a single ~2000
character document and then ask different questions about it. Without prefix
caching that document is re-prefilled fifteen times.

A/B on the same engine, same blocks, same everything - only the cache is toggled,
so the difference cannot come from anything else.
"""
from __future__ import annotations
import argparse, json, platform, statistics as st, sys, time
from pathlib import Path

import torch, transformers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inferno.prefix_engine import PrefixEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="rag", choices=["rag", "mixed"])
    ap.add_argument("--n-requests", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--n-blocks", type=int, default=768)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    by_cat = {}
    for p in spec["prompts"]:
        by_cat.setdefault(p["category"], []).append(p)

    if args.workload == "rag":
        pool = by_cat["long"]                      # all share one document
    else:
        pool = [x for pair in zip(by_cat["long"], by_cat["short"]) for x in pair]
    chosen = [pool[i % len(pool)] for i in range(args.n_requests)]
    texts = [p["text"] for p in chosen]
    ids = [f"{p['id']}#{i}" for i, p in enumerate(chosen)]

    eng = PrefixEngine("Qwen/Qwen2.5-0.5B-Instruct", device=args.device,
                       dtype=args.dtype, attn="eager")
    # Warm up so shader compilation does not land on the first measured prefill
    # (PROJECT_LOG.md B2).
    eng.run(eng.make_requests(texts[:2], max_new_tokens=4, ids=["w0", "w1"]),
            n_blocks=args.n_blocks, block_size=args.block_size)

    print(f"workload={args.workload}  n={args.n_requests}  "
          f"block_size={args.block_size}  device={args.device} {args.dtype}")
    print(f"prompt lengths: {[len(eng.encode(t)[0]) for t in texts[:3]]}...\n")

    out = {"rung": "R5", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "config": {"workload": args.workload, "n_requests": args.n_requests,
                      "max_new_tokens": args.max_new_tokens,
                      "block_size": args.block_size, "n_blocks": args.n_blocks,
                      "device": eng.device, "dtype": args.dtype,
                      "model": "Qwen/Qwen2.5-0.5B-Instruct",
                      "prompts_version": spec["version"]},
           "environment": {"torch": torch.__version__,
                           "transformers": transformers.__version__,
                           "platform": platform.platform()},
           "runs": {}}

    print(f"{'prefix cache':>14} {'hit rate':>9} {'prefill tok':>13} {'saved':>7} "
          f"{'cold TTFT':>10} {'hit TTFT':>9} {'wall':>7} {'tok/s':>8}")
    for flag in (False, True):
        reqs = eng.make_requests(texts, max_new_tokens=args.max_new_tokens, ids=ids)
        served, s = eng.run(reqs, n_blocks=args.n_blocks,
                            block_size=args.block_size, enable_prefix_cache=flag)
        pr = s["per_request"]
        cold = [x for x in pr if not x["hit"]]
        hot = [x for x in pr if x["hit"]]
        row = {
            "prefix_cache": flag, "hit_rate": s["hit_rate"],
            "prefill_tokens_computed": s["prefill_tokens_computed"],
            "prefill_tokens_total": s["prefill_tokens_total"],
            "prefill_saved_frac": s["prefill_tokens_saved_frac"],
            "cold_prefill_ms": st.mean(x["prefill_s"] for x in cold) * 1000 if cold else None,
            "hit_prefill_ms": st.mean(x["prefill_s"] for x in hot) * 1000 if hot else None,
            "n_cold": len(cold), "n_hit": len(hot),
            "wall_clock_s": s["wall_clock_s"],
            "throughput_tok_s": s["tokens_generated"] / s["wall_clock_s"],
            "preemptions": s["preemptions"],
            "per_request": pr,
        }
        out["runs"]["on" if flag else "off"] = row
        print(f"{str(flag):>14} {s['hit_rate']*100:>8.1f}% "
              f"{s['prefill_tokens_computed']:>6}/{s['prefill_tokens_total']:<6} "
              f"{s['prefill_tokens_saved_frac']*100:>6.1f}% "
              f"{row['cold_prefill_ms'] or 0:>9.1f}ms "
              f"{(row['hit_prefill_ms'] or 0):>8.1f}ms "
              f"{s['wall_clock_s']:>7.1f} {row['throughput_tok_s']:>8.2f}")

    a, b = out["runs"]["off"], out["runs"]["on"]
    if b["hit_prefill_ms"]:
        print(f"\n  TTFT on a cache hit: {a['cold_prefill_ms']:.1f} ms -> "
              f"{b['hit_prefill_ms']:.1f} ms  "
              f"({a['cold_prefill_ms']/b['hit_prefill_ms']:.2f}x)")
    print(f"  wall clock: {a['wall_clock_s']:.1f}s -> {b['wall_clock_s']:.1f}s  "
          f"({a['wall_clock_s']/b['wall_clock_s']:.2f}x)")

    p = ROOT / "results" / ("gpu" if eng.device == "cuda" else "mac") / \
        f"r5_prefix_{args.device}_{args.workload}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
