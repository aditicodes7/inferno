"""R1 parity diagnostic. Distinguishes a logic bug from accumulation noise.

Two experiments:

A) TEACHER-FORCED AGREEMENT (mps/fp16). Feed both implementations the *same*
   prefix at every step - the reference's own tokens - and compare argmax.
   This decouples "our logits are wrong" from "we diverged once and then
   drifted". Free-running divergence is expected under either hypothesis;
   teacher-forced disagreement is only expected under a logic bug.

   At each disagreement it records HuggingFace's top1-minus-top2 margin. If
   disagreements happen only where that margin is near zero, the logits are
   fine and argmax is simply resolving a tie differently.

B) FLOAT32 ON CPU. Accumulation order matters far less at fp32, and CPU
   kernels are more stable. If free-running parity holds here and fails on
   mps/fp16, the cause is precision, not logic.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from inferno.engine import InfernoEngine
from inferno.cache import KVCache

def _reference(device: str):
    """Newest eager reference for THIS device. The path used to be hardcoded to
    a file that was later superseded and moved, so this script was broken even
    on the machine it was written on."""
    hw = "gpu" if device == "cuda" else "mac"
    found = sorted((ROOT / "results" / hw).glob(f"r0_baseline_{device}_eager_*.json"))
    if not found:
        raise SystemExit(
            f"No eager reference for device={device!r}. Generate one:\n"
            f"    python bench/run_baseline.py --device {device} --attn eager")
    return found[-1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load(device="mps"):
    ref = json.loads(_reference(device).read_text())
    prompts = {p["id"]: p for p in json.loads((ROOT / "bench/prompts.json").read_text())["prompts"]}
    return ref, {r["prompt_id"]: r for r in ref["records"]}, prompts


def experiment_a(pids, device="mps", dtype=torch.float16):
    print(f"\n{'='*70}\nA) TEACHER-FORCED AGREEMENT  [{device} / {dtype}]\n{'='*70}")
    ref, exp, prompts = load(device)
    eng = InfernoEngine(MODEL, device=device, dtype=str(dtype).removeprefix("torch."), attn="eager")
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, attn_implementation="eager").to(device).eval()

    for pid in pids:
        want = exp[pid]["token_ids"]
        ids = eng.encode(prompts[pid]["text"])
        full = torch.cat([ids, torch.tensor([want], device=device)], dim=1)

        # HF: one pass over the whole teacher-forced sequence.
        with torch.inference_mode():
            hf_logits = hf(full).logits[0].float()          # [T, V]

        n_prompt = ids.shape[1]
        cache = KVCache(eng.cfg.n_layers, eng.cfg.n_kv_heads, eng.cfg.head_dim,
                        full.shape[1] + 1, eng.dtype, device)
        inf_logits = [eng.model.forward(ids, cache)[0, -1].float()]
        for t in want[:-1]:
            inf_logits.append(eng.model.forward(
                torch.tensor([[t]], device=device), cache)[0, -1].float())
        inf_logits = torch.stack(inf_logits)                # [n_gen, V]

        hf_slice = hf_logits[n_prompt - 1: n_prompt - 1 + len(inf_logits)]
        assert hf_slice.shape == inf_logits.shape

        dmax = (hf_slice - inf_logits).abs().max().item()
        scale = hf_slice.abs().max().item()
        hf_top = hf_slice.argmax(-1)
        inf_top = inf_logits.argmax(-1)
        agree = (hf_top == inf_top)
        bad = (~agree).nonzero().flatten().tolist()

        top2 = hf_slice.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1])

        print(f"\n{pid}  (prompt {n_prompt} tok, {len(want)} generated)")
        print(f"  max |logit diff|      {dmax:.5f}   (logit scale ~{scale:.1f}, "
              f"relative {dmax/scale:.2e})")
        print(f"  teacher-forced top-1  {int(agree.sum())}/{len(agree)} agree")
        if bad:
            print(f"  disagree at steps     {bad[:10]}")
            print(f"  HF margin AT those    {[round(margin[i].item(), 4) for i in bad[:10]]}")
            others = margin[agree]
            print(f"  HF margin elsewhere   median {others.median().item():.4f}  "
                  f"min {others.min().item():.4f}")
        else:
            print("  -> every step agrees under teacher forcing")


def experiment_b(pids, n_new=64):
    print(f"\n{'='*70}\nB) FREE-RUNNING PARITY IN FLOAT32 ON CPU\n{'='*70}")
    _, _, prompts = load()
    eng = InfernoEngine(MODEL, device="cpu", dtype="float32", attn="eager")
    hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                              attn_implementation="eager").to("cpu").eval()
    ok = 0
    for pid in pids:
        ids = eng.encode(prompts[pid]["text"])
        with torch.inference_mode():
            out = hf.generate(ids, max_new_tokens=n_new, do_sample=False,
                              temperature=None, top_p=None, top_k=None,
                              pad_token_id=eng.tokenizer.eos_token_id)
        want = out[0, ids.shape[1]:].tolist()
        got = eng.generate(prompts[pid]["text"], max_new_tokens=n_new)
        n = min(len(got), len(want))
        idx = next((i for i in range(n) if got[i] != want[i]), None)
        match = idx is None and len(got) == len(want)
        ok += match
        print(f"  {pid:<11} {'MATCH' if match else f'diverge@{idx}':<14}"
              f" got {len(got):>3} want {len(want):>3}")
    print(f"\n  fp32/CPU parity: {ok}/{len(pids)}")


if __name__ == "__main__":
    experiment_a(["short-17", "long-01", "short-09", "medium-06"])
    experiment_b(["short-00", "short-09", "short-17", "medium-06", "long-01"])
