"""R3 measurement: continuous vs static batching under REAL ARRIVALS.

Every earlier benchmark assumed all 50 prompts exist at t=0. Under that
assumption continuous batching has little to show - its whole advantage is
absorbing a request that turns up mid-flight. So this harness generates an
arrival schedule and replays it against both schedulers.

The static baseline is R2's policy made arrival-aware, which is the honest
comparison: take whatever has arrived (up to batch_size), run that batch to
completion, then look again. A request that arrives one iteration after the
batch starts waits for the whole thing to finish - that is the cost R3 removes.

Headline metric: the highest arrival rate each policy sustains with
p95 TTFT under the budget.
"""
from __future__ import annotations
import argparse, json, platform, random, sys, time
from pathlib import Path

import torch, transformers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inferno.continuous_engine import ContinuousEngine
from inferno.batch_engine import BatchEngine
from bench.run_baseline import percentile


def arrivals(pattern: str, n: int, rate: float, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    if pattern == "allatonce":
        return [0.0] * n
    if pattern == "poisson":
        t, out = 0.0, []
        for _ in range(n):
            t += rng.expovariate(rate)
            out.append(t)
        return out
    if pattern == "burst":
        # Three tight bursts separated by quiet gaps.
        out, t = [], 0.0
        for i in range(n):
            if i and i % (n // 3 or 1) == 0:
                t += 3.0
            out.append(t + rng.uniform(0, 0.05))
        return out
    raise ValueError(pattern)


def run_continuous(eng, texts, arr, max_new, slots):
    reqs = eng.make_requests(texts, arrivals=arr, max_new_tokens=max_new,
                             ids=[f"r{i}" for i in range(len(texts))])
    t0 = time.perf_counter()
    served, summary = eng.run(reqs, n_slots=slots)
    wall = time.perf_counter() - t0
    recs = [{"id": r.id, "arrival": a, "n_prompt": r.n_prompt,
             "n_generated": r.n_generated,
             "ttft_s": max(r.first_token_at - a, 0.0),
             "latency_s": max(r.finished_at - a, 0.0)}
            for r, a in zip(served, arr)]
    return recs, wall, summary


def run_static(eng, texts, arr, max_new, batch_size):
    """R2's policy, arrival-aware: a batch runs to completion before re-looking."""
    order = sorted(range(len(texts)), key=lambda i: arr[i])
    recs, pending, t0 = [], list(order), time.perf_counter()
    while pending:
        now = time.perf_counter() - t0
        ready = [i for i in pending if arr[i] <= now]
        if not ready:
            time.sleep(0.001)
            continue
        chunk = ready[:batch_size]
        start = time.perf_counter() - t0
        outs, st = eng.generate_batch([texts[i] for i in chunk],
                                      max_new_tokens=max_new, return_stats=True)
        end = time.perf_counter() - t0
        per_step = (end - start) / max(st["steps_run"], 1)
        for i, o in zip(chunk, outs):
            recs.append({"id": f"r{i}", "arrival": arr[i], "n_prompt": 0,
                         "n_generated": len(o),
                         # first token lands one step after the batch starts
                         "ttft_s": max(start + per_step - arr[i], 0.0),
                         "latency_s": max(end - arr[i], 0.0)})
        for i in chunk:
            pending.remove(i)
    return recs, time.perf_counter() - t0, {}


def summarise(recs, wall, budget):
    ttfts = [r["ttft_s"] for r in recs]
    lats = [r["latency_s"] for r in recs]
    return {
        "n_requests": len(recs),
        "tokens_generated": sum(r["n_generated"] for r in recs),
        "wall_clock_s": wall,
        "throughput_tok_s": sum(r["n_generated"] for r in recs) / wall,
        "ttft_p50_s": percentile(ttfts, .50), "ttft_p95_s": percentile(ttfts, .95),
        "latency_p50_s": percentile(lats, .50), "latency_p95_s": percentile(lats, .95),
        "meets_budget": percentile(ttfts, .95) < budget,
        "frac_under_budget": sum(t < budget for t in ttfts) / len(ttfts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="poisson",
                    choices=["poisson", "burst", "allatonce"])
    ap.add_argument("--rates", default="2,4,8,16")
    ap.add_argument("--n-requests", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--budget", type=float, default=0.5, help="p95 TTFT budget, seconds")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    pool = spec["prompts"]
    texts = [pool[i % len(pool)]["text"] for i in range(args.n_requests)]

    cont = ContinuousEngine("Qwen/Qwen2.5-0.5B-Instruct", device=args.device,
                            dtype=args.dtype, attn="eager")
    stat = BatchEngine("Qwen/Qwen2.5-0.5B-Instruct", device=args.device,
                       dtype=args.dtype, attn="eager")
    for _ in range(2):
        stat.generate_batch([texts[0]] * 2, max_new_tokens=8)

    out = {"rung": "R3", "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "config": {"pattern": args.pattern, "n_requests": args.n_requests,
                      "max_new_tokens": args.max_new_tokens, "slots": args.slots,
                      "batch_size": args.batch_size, "budget_s": args.budget,
                      "device": cont.device, "dtype": args.dtype,
                      "model": "Qwen/Qwen2.5-0.5B-Instruct",
                      "prompts_version": spec["version"]},
           "environment": {"torch": torch.__version__,
                           "transformers": transformers.__version__,
                           "platform": platform.platform()},
           "by_rate": {}}

    print(f"pattern={args.pattern}  n={args.n_requests}  max_new={args.max_new_tokens}  "
          f"slots={args.slots}  static_bs={args.batch_size}  budget p95 TTFT < {args.budget}s\n")
    print(f"{'rate':>6} {'policy':>11} {'tok/s':>8} {'wall':>7} {'TTFT p50':>9} "
          f"{'TTFT p95':>9} {'lat p95':>8} {'budget':>7}")
    for rate in [float(r) for r in args.rates.split(",")]:
        arr = arrivals(args.pattern, args.n_requests, rate)
        row = {}
        for name, fn in (("continuous", lambda: run_continuous(cont, texts, arr, args.max_new_tokens, args.slots)),
                         ("static", lambda: run_static(stat, texts, arr, args.max_new_tokens, args.batch_size))):
            recs, wall, _ = fn()
            s = summarise(recs, wall, args.budget)
            s["records"] = recs
            row[name] = s
            print(f"{rate:>6.1f} {name:>11} {s['throughput_tok_s']:>8.2f} "
                  f"{s['wall_clock_s']:>7.1f} {s['ttft_p50_s']:>9.3f} "
                  f"{s['ttft_p95_s']:>9.3f} {s['latency_p95_s']:>8.2f} "
                  f"{'OK' if s['meets_budget'] else 'MISS':>7}")
        out["by_rate"][str(rate)] = row
        print()

    p = ROOT / "results" / ("gpu" if cont.device == "cuda" else "mac") / \
        f"r3_continuous_{args.device}_{args.pattern}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"-> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
