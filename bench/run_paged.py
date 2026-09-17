"""R4 measurement: paged vs contiguous KV at a FIXED memory budget.

The comparison that matters is not "is paging faster" - it is not. It is how
many sequences each design fits in the same bytes.

R3 gives every sequence a contiguous reservation of `max_len`, so concurrency is
`budget / (max_len * bytes_per_token)`, decided in advance and independent of
how long sequences turn out to be. R4 hands out blocks on demand, so
concurrency is bounded by what sequences ACTUALLY use.

Also measured, because rule 8 says so: the throughput each design achieves when
neither is memory constrained. That difference is the cost of the gather - the
copy a fused paged-attention kernel would avoid - and it is expected to be the
largest single line item in the final gap analysis.
"""
from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path

import torch, transformers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inferno.continuous_engine import ContinuousEngine
from inferno.paged_engine import PagedEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-mb", type=float, default=48.0)
    ap.add_argument("--n-requests", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--block-sizes", default="4,16,64")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    # Short + medium chat traffic, with max_len still sized for the long tail -
    # which is exactly the situation contiguous reservation handles worst.
    pool = [p for p in spec["prompts"] if p["category"] in ("short", "medium")]
    texts = [pool[i % len(pool)]["text"] for i in range(args.n_requests)]

    paged = PagedEngine("Qwen/Qwen2.5-0.5B-Instruct", device=args.device,
                        dtype=args.dtype, attn="eager")
    cfg = paged.cfg
    bytes_per_token = 2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * \
        torch.empty(0, dtype=paged.dtype).element_size()
    budget = int(args.budget_mb * 1e6)

    reqs0 = paged.make_requests(texts, max_new_tokens=args.max_new_tokens)
    prompt_lens = [r.n_prompt for r in reqs0]
    # max_len must cover the worst case the SYSTEM may see, not this workload's
    # mean - that is the whole premise of contiguous reservation.
    longest_possible = max(p["id"] and len(paged.encode(p["text"])[0])
                           for p in spec["prompts"])
    max_len = longest_possible + args.max_new_tokens

    n_slots = max(budget // (max_len * bytes_per_token), 1)
    print(f"budget {args.budget_mb:.0f} MB | {bytes_per_token} bytes/token | "
          f"max_len {max_len} (longest prompt {longest_possible} + "
          f"{args.max_new_tokens})")
    print(f"prompt lengths in this workload: min {min(prompt_lens)} "
          f"median {sorted(prompt_lens)[len(prompt_lens)//2]} max {max(prompt_lens)}")
    print(f"\nR3 contiguous: {n_slots} slots x {max_len} tokens\n")

    out = {"rung": "R4", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "config": {"budget_bytes": budget, "bytes_per_token": bytes_per_token,
                      "max_len": max_len, "n_requests": args.n_requests,
                      "max_new_tokens": args.max_new_tokens,
                      "device": paged.device, "dtype": args.dtype,
                      "model": "Qwen/Qwen2.5-0.5B-Instruct",
                      "prompts_version": spec["version"]},
           "environment": {"torch": torch.__version__,
                           "transformers": transformers.__version__,
                           "platform": platform.platform()},
           "runs": {}}

    print(f"{'design':>22} {'concurrent':>11} {'tok/s':>8} {'wall':>7} "
          f"{'kv MB':>7} {'blk util':>9} {'preempt':>8}")

    cont = ContinuousEngine("Qwen/Qwen2.5-0.5B-Instruct", device=args.device,
                            dtype=args.dtype, attn="eager")
    reqs = cont.make_requests(texts, max_new_tokens=args.max_new_tokens)
    t = time.perf_counter()
    served, summary = cont.run(reqs, n_slots=n_slots, max_len=max_len)
    wall = time.perf_counter() - t
    toks = sum(r.n_generated for r in served)
    # How full a slot is WHILE OCCUPIED, which is what compares to R4's block
    # utilisation. Dividing total tokens by (slots * max_len) over-counts as
    # soon as more requests are served than there are slots - it reported 126%
    # on the 56-request run, which is how the error surfaced.
    occupancy = [(r.n_prompt + r.n_generated) / max_len for r in served]
    r3 = {"design": "R3 contiguous", "concurrent_capacity": n_slots,
          "peak_concurrent": min(n_slots, args.n_requests),
          "throughput_tok_s": toks / wall, "wall_clock_s": wall,
          "kv_bytes": summary["kv_bytes_allocated"], "preemptions": 0,
          "utilisation": sum(occupancy) / len(occupancy)}
    out["runs"]["r3"] = r3
    print(f"{'R3 contiguous':>22} {r3['concurrent_capacity']:>11} "
          f"{r3['throughput_tok_s']:>8.2f} {wall:>7.1f} "
          f"{r3['kv_bytes']/1e6:>7.1f} {r3['utilisation']*100:>8.1f}% {0:>8}")
    del cont

    for bs in [int(b) for b in args.block_sizes.split(",")]:
        n_blocks = int(budget // (bs * bytes_per_token))
        reqs = paged.make_requests(texts, max_new_tokens=args.max_new_tokens)
        t = time.perf_counter()
        served, summary = paged.run(reqs, n_blocks=n_blocks, block_size=bs,
                                    max_concurrent=args.n_requests)
        wall = time.perf_counter() - t
        toks = sum(r.n_generated for r in served)
        row = {"design": f"R4 paged bs={bs}", "n_blocks": n_blocks,
               "concurrent_capacity": None,
               "peak_concurrent": summary["peak_concurrent_sequences"],
               "throughput_tok_s": toks / wall, "wall_clock_s": wall,
               "kv_bytes": summary["kv_bytes_allocated"],
               "block_utilisation": summary["block_utilisation_mean"],
               "preemptions": summary["preemptions"],
               "blocks_leaked": summary["blocks_leaked"]}
        out["runs"][f"r4_bs{bs}"] = row
        print(f"{'R4 paged bs=' + str(bs):>22} {row['peak_concurrent']:>11} "
              f"{row['throughput_tok_s']:>8.2f} {wall:>7.1f} "
              f"{row['kv_bytes']/1e6:>7.1f} "
              f"{row['block_utilisation']*100:>8.1f}% {row['preemptions']:>8}")

    p = ROOT / "results" / ("gpu" if paged.device == "cuda" else "mac") / \
        f"r4_paged_{args.device}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
