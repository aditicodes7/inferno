"""R2: static batching with left-padding.

Why batching helps at all: a decode step loads every weight matrix from memory
and uses each one for a single row of activations. Running B sequences in the
same step loads those weights exactly once and multiplies a B-row matrix
instead. B times the useful work, nearly the same time - until something else
saturates. Finding where it stops scaling is R2's number.

Why LEFT padding: after prefill, the next token for every sequence must come
from the same tensor index. Right-padding would put each sequence's last real
token at a different column, so `logits[:, -1]` would be the wrong position for
every sequence except the longest. Left-padding right-aligns them, which has a
second benefit: all sequences share one cache write offset, so `cache.length`
can stay a single integer for the whole batch.

R2 DESIGN CHOICE, deliberately naive: a batch runs until EVERY sequence in it
finishes. Sequences that hit EOS early keep being stepped and their tokens are
discarded. That is head-of-line blocking, and measuring it is the point - the
latency histogram it produces is what motivates R3.
"""

from __future__ import annotations

import torch

from inferno.cache import KVCache
from inferno.engine import InfernoEngine


class BatchEngine(InfernoEngine):
    """Batched greedy decoding. Inherits weight loading and tokenisation."""

    def encode_batch(self, prompts: list[str]):
        """Left-pad to the longest prompt. Returns (input_ids, pad_mask)."""
        seqs = [self.encode(p)[0] for p in prompts]
        max_len = max(s.shape[0] for s in seqs)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        input_ids = torch.full((len(seqs), max_len), pad_id,
                               dtype=torch.long, device=self.device)
        pad_mask = torch.zeros((len(seqs), max_len),
                               dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            input_ids[i, max_len - s.shape[0]:] = s     # LEFT pad
            pad_mask[i, max_len - s.shape[0]:] = 1
        return input_ids, pad_mask

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str], max_new_tokens: int = 128,
                       return_stats: bool = False):
        """Greedy decode a whole batch. One output list per input prompt."""
        input_ids, pad_mask = self.encode_batch(prompts)
        n, prompt_len = input_ids.shape

        cache = KVCache(
            n_layers=self.cfg.n_layers, n_kv_heads=self.cfg.n_kv_heads,
            head_dim=self.cfg.head_dim, max_len=prompt_len + max_new_tokens,
            dtype=self.dtype, device=self.device, batch_size=n,
        )

        # Prefill the whole batch in one pass.
        logits = self.model.forward(input_ids, cache, pad_mask)
        tokens = logits[:, -1].argmax(-1)                       # [B]

        outs: list[list[int]] = [[int(t)] for t in tokens]
        finished = [int(t) in self.eos_ids for t in tokens]
        finished_at = [1 if f else None for f in finished]

        step = 1
        while step < max_new_tokens and not all(finished):
            # Every generated position is real, for every sequence in the batch.
            pad_mask = torch.cat(
                [pad_mask, torch.ones((n, 1), dtype=torch.long, device=self.device)],
                dim=1,
            )
            logits = self.model.forward(tokens.unsqueeze(1), cache, pad_mask)
            tokens = logits[:, -1].argmax(-1)
            step += 1
            for i in range(n):
                if finished[i]:
                    continue        # still being stepped; its token is discarded
                tok = int(tokens[i])
                outs[i].append(tok)
                if tok in self.eos_ids:
                    finished[i] = True
                    finished_at[i] = step

        if not return_stats:
            return outs

        stats = {
            "batch_size": n,
            "prompt_len_padded": prompt_len,
            "prompt_lens": [int(m.sum()) for m in pad_mask[:, :prompt_len]],
            "steps_run": step,
            "finished_at": [f if f is not None else step for f in finished_at],
            # Steps a sequence sat in the batch after finishing, producing
            # nothing. This is head-of-line blocking, in units of decode steps.
            "blocked_steps": [step - (f if f is not None else step)
                              for f in finished_at],
            "kv_utilisation": cache.utilisation(),
        }
        return outs, stats
