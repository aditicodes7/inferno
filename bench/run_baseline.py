"""R0 baseline: HuggingFace generate(), greedy, one request at a time.

This establishes two things that every later rung depends on:

  1. A performance floor (decode tok/s, TTFT, peak memory) to compare against.
  2. The parity ground truth - the exact token IDs each prompt produces under
     greedy decoding. From R1 onward, Inferno must reproduce these EXACTLY.

Because of (2), the output file records device, dtype, attention implementation
and library versions. Change any of those and the token IDs may shift, which
would silently invalidate every downstream parity test.

Usage:
    .venv/bin/python bench/run_baseline.py                 # honest speed baseline
    .venv/bin/python bench/run_baseline.py --attn eager    # parity reference run
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
                          StoppingCriteria)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# --------------------------------------------------------------------------
# timing instrumentation
# --------------------------------------------------------------------------

class FirstTokenTimer(StoppingCriteria):
    """Records a timestamp the first time generate() completes a decode step.

    StoppingCriteria is invoked after each new token has been appended, so the
    first invocation is the moment the first token actually exists. This never
    stops generation - it only observes.
    """

    def __init__(self) -> None:
        self.t_first: float | None = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.t_first is None:
            self.t_first = time.perf_counter()
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def sync(device: str) -> None:
    """Make async device work observable to the host clock before we time it."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class Record:
    prompt_id: str
    category: str
    position: int           # index within this run - for ordering-effect analysis
    n_prompt_tokens: int
    n_generated: int
    ttft_s: float
    decode_s: float
    decode_tok_s: float
    token_ids: list[int]     # the parity ground truth
    text: str


def peak_memory_bytes(device: str) -> dict[str, int]:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024                      # Linux reports KiB, macOS reports bytes
    out = {"peak_rss_bytes": rss}
    if device == "mps":
        out["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        out["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    elif device == "cuda":
        out["cuda_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
        out["cuda_max_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------

def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    # float16 on CPU is emulated and pathologically slow; keep CPU in float32.
    return torch.float32 if device == "cpu" else torch.float16


def build_inputs(tokenizer, text: str, device: str):
    """Apply the Instruct chat template - this is the realistic serving path."""
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt, return_tensors="pt").to(device)


def build_generation_config(model_id: str, tokenizer, max_new_tokens: int) -> GenerationConfig:
    """Construct the decoding policy explicitly. Do NOT inherit the checkpoint's.

    Qwen2.5-0.5B-Instruct ships generation_config.json with
    repetition_penalty=1.1, and HuggingFace applies RepetitionPenaltyLogitsProcessor
    in GREEDY decoding - it is not gated on do_sample. Inheriting it silently
    means "greedy" is not greedy. That cost us a full R1 parity debugging cycle
    (PROJECT_LOG.md B3), so the policy is now stated here in full rather than
    picked up from a file in the checkpoint.
    """
    shipped = GenerationConfig.from_pretrained(model_id)
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,             # greedy
        repetition_penalty=1.0,      # OFF - the whole point of this function
        no_repeat_ngram_size=0,
        renormalize_logits=False,
        temperature=None, top_p=None, top_k=None, min_p=None,
        num_beams=1,
        eos_token_id=shipped.eos_token_id,
        pad_token_id=shipped.pad_token_id or tokenizer.eos_token_id,
    )


def describe_sampling(gc: GenerationConfig) -> dict:
    """Record the full logits-processor stack, not just the sampling mode."""
    return {
        "do_sample": gc.do_sample, "num_beams": gc.num_beams,
        "repetition_penalty": gc.repetition_penalty,
        "no_repeat_ngram_size": gc.no_repeat_ngram_size,
        "renormalize_logits": gc.renormalize_logits,
        "temperature": gc.temperature, "top_p": gc.top_p, "top_k": gc.top_k,
        "min_p": gc.min_p, "eos_token_id": gc.eos_token_id,
    }


@torch.inference_mode()
def generate_once(model, tokenizer, inputs, gen_cfg, device: str):
    timer = FirstTokenTimer()
    sync(device)
    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        generation_config=gen_cfg,
        use_cache=True,
        stopping_criteria=[timer],
    )
    sync(device)
    t_end = time.perf_counter()

    assert timer.t_first is not None, "no token was generated"
    n_prompt = inputs["input_ids"].shape[1]
    new_ids = out[0, n_prompt:].tolist()
    n_new = len(new_ids)

    ttft = timer.t_first - t0
    decode_s = t_end - timer.t_first     # excludes prefill by construction
    tok_s = (n_new - 1) / decode_s if n_new > 1 and decode_s > 0 else float("nan")
    return new_ids, n_prompt, ttft, decode_s, tok_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                    help="sdpa = honest speed baseline; eager = parity reference")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="run only first N prompts")
    ap.add_argument("--reverse", action="store_true",
                    help="run the prompt set in reverse order; isolates "
                         "position-in-run effects from per-prompt effects")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    spec = json.loads((ROOT / "bench" / "prompts.json").read_text())
    prompts = spec["prompts"][: args.limit]
    # Warm up on a fixed prompt regardless of ordering, so that a reversed run
    # differs from a forward run in exactly one variable: position in the run.
    warmup_text = prompts[0]["text"]
    if args.reverse:
        prompts = list(reversed(prompts))

    print(f"model={args.model} device={device} dtype={dtype} attn={args.attn}")
    print(f"prompts={len(prompts)} max_new_tokens={args.max_new_tokens} "
          f"warmup={args.warmup}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation=args.attn
    ).to(device).eval()

    # Warm-up: kernel autotuning, lazy allocator growth and MPS graph capture all
    # make the first few generations unrepresentative. Discard them.
    gen_cfg = build_generation_config(args.model, tokenizer, args.max_new_tokens)
    print(f"sampling: {describe_sampling(gen_cfg)}\n")
    warm_inputs = build_inputs(tokenizer, warmup_text, device)
    for i in range(args.warmup):
        generate_once(model, tokenizer, warm_inputs, gen_cfg, device)
        print(f"  warmup {i + 1}/{args.warmup} discarded")
    print()

    records: list[Record] = []
    for i, p in enumerate(prompts):
        inputs = build_inputs(tokenizer, p["text"], device)
        ids, n_prompt, ttft, decode_s, tok_s = generate_once(
            model, tokenizer, inputs, gen_cfg, device
        )
        records.append(Record(
            prompt_id=p["id"], category=p["category"], position=i,
            n_prompt_tokens=n_prompt, n_generated=len(ids),
            ttft_s=ttft, decode_s=decode_s, decode_tok_s=tok_s,
            token_ids=ids, text=tokenizer.decode(ids, skip_special_tokens=True),
        ))
        print(f"[{i + 1:2d}/{len(prompts)}] {p['id']:<10} "
              f"prompt={n_prompt:>5}tok  ttft={ttft * 1000:7.1f}ms  "
              f"decode={tok_s:6.2f} tok/s")

    ttfts = [r.ttft_s for r in records]
    total_new = sum(r.n_generated - 1 for r in records)
    total_decode = sum(r.decode_s for r in records)

    summary = {
        "decode_tok_s_aggregate": total_new / total_decode,
        "decode_tok_s_mean_per_request": statistics.mean(r.decode_tok_s for r in records),
        "ttft_p50_ms": percentile(ttfts, 0.50) * 1000,
        "ttft_p95_ms": percentile(ttfts, 0.95) * 1000,
        "total_generated_tokens": sum(r.n_generated for r in records),
        "wall_clock_s": total_decode + sum(ttfts),
        "memory": peak_memory_bytes(device),
        "by_category": {
            c: {
                "n": len([r for r in records if r.category == c]),
                "decode_tok_s": statistics.mean(
                    r.decode_tok_s for r in records if r.category == c),
                "ttft_p50_ms": percentile(
                    [r.ttft_s for r in records if r.category == c], 0.5) * 1000,
            }
            for c in sorted({r.category for r in records})
        },
    }

    payload = {
        "rung": "R0",
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "model": args.model, "device": device, "dtype": str(dtype),
            "attn_implementation": args.attn,
            "max_new_tokens": args.max_new_tokens, "warmup": args.warmup,
            "chat_template": True, "reverse": args.reverse,
            "sampling": describe_sampling(gen_cfg),
            "prompts_version": spec["version"], "n_prompts": len(prompts),
        },
        "environment": {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
        },
        "summary": summary,
        "records": [asdict(r) for r in records],
    }

    hw = "gpu" if device == "cuda" else "mac"
    out = Path(args.out) if args.out else (
        ROOT / "results" / hw /
        f"r0_baseline_{device}_{args.attn}"
        f"{'_rev' if args.reverse else ''}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    mem_mb = summary["memory"]["peak_rss_bytes"] / 1e6
    print(f"\n{'=' * 62}\nR0 BASELINE  [{device} / {dtype} / attn={args.attn}]")
    print(f"{'=' * 62}")
    print(f"  decode throughput   {summary['decode_tok_s_aggregate']:8.2f} tok/s "
          f"(aggregate, prefill excluded)")
    print(f"  TTFT p50            {summary['ttft_p50_ms']:8.1f} ms")
    print(f"  TTFT p95            {summary['ttft_p95_ms']:8.1f} ms")
    print(f"  peak RSS            {mem_mb:8.1f} MB")
    for c, s in summary["by_category"].items():
        print(f"    {c:<8} n={s['n']:<3} {s['decode_tok_s']:6.2f} tok/s  "
              f"ttft p50 {s['ttft_p50_ms']:7.1f} ms")
    print(f"\n  -> {out.relative_to(ROOT)}")
    if args.attn == "sdpa":
        print("  NOTE: this is the speed baseline. Run again with --attn eager "
              "to\n        produce the parity reference used from R1 onward.")


if __name__ == "__main__":
    main()
