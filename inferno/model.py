"""Qwen2 forward pass, written by hand.

Every tensor operation between token IDs and logits is in this file. Nothing
is imported from transformers except the weights themselves.

The numerics are not a free choice: R1 is accepted only on producing tokens
identical to HuggingFace's eager path, and greedy decoding amplifies any
difference into total divergence once argmax flips on a near-tie. Three places
where HuggingFace upcasts to float32 are reproduced deliberately and marked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ModelConfig:
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rms_eps: float
    rope_theta: float
    vocab_size: int

    @property
    def n_rep(self) -> int:
        """Query heads per KV head. 7 for Qwen2.5-0.5B (14 query, 2 KV)."""
        return self.n_heads // self.n_kv_heads

    @property
    def scaling(self) -> float:
        return self.head_dim ** -0.5


@dataclass
class LayerWeights:
    q_w: torch.Tensor; q_b: torch.Tensor
    k_w: torch.Tensor; k_b: torch.Tensor
    v_w: torch.Tensor; v_b: torch.Tensor
    o_w: torch.Tensor                      # Qwen2 has NO bias on o_proj
    gate_w: torch.Tensor
    up_w: torch.Tensor
    down_w: torch.Tensor
    in_norm_w: torch.Tensor
    post_norm_w: torch.Tensor


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """UPCAST 1/3: variance is accumulated in float32, then cast back.

    In float16 the sum of 896 squared activations loses enough precision to
    shift the normalised output, which greedy decoding can turn into a
    different token.
    """
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding:
    """RoPE. Rotates Q and K by an angle proportional to absolute position.

    The dot product of a rotated query at position i with a rotated key at
    position j then depends only on i - j, which is how relative position
    enters attention without a separate bias term.
    """

    def __init__(self, head_dim: int, theta: float, device: str) -> None:
        exponent = torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim
        self.inv_freq = (1.0 / (theta ** exponent)).to(device)

    def __call__(self, position_ids: torch.Tensor, dtype: torch.dtype):
        """UPCAST 2/3: angles are computed in float32, then cast to the model dtype.

        cos/sin come back shaped [B, T, head_dim].
        """
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)          # [B, 1, T, head_dim] to broadcast over heads
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand n_kv_heads to n_heads for grouped-query attention.

    Query head i is served by KV head i // n_rep. The expand-then-reshape
    ordering is what produces that pairing; interleaving instead would pair
    every query head with the wrong keys and still run without error.
    """
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


def mask_fill_value(dtype: torch.dtype) -> torch.Tensor:
    """The "invisible" value for an ADDITIVE attention mask.

    NOT `finfo.min`, which is the obvious choice and is wrong in float16.
    The mask is added to the raw scores, and float16's most negative finite
    value is -65504: adding any score beyond about -16 to it rounds past the
    end of the range and becomes -inf. Attention scores reach -225 by layer 8
    of this model. A row that is entirely masked then becomes all -inf, and
    softmax computes exp(-inf - (-inf)) = NaN.

    That NaN then escapes into real tokens, because pad K/V live in the same
    cache and a masked weight is exactly 0 - but 0 * NaN = NaN in the value
    matmul. Masking does not protect a real query from a NaN pad value.
    (PROJECT_LOG.md B4.)

    Halving the value leaves ~32752 of headroom for the score, while
    exp(-32752 - max) still underflows to exactly 0 in the float32 softmax -
    so masked positions contribute nothing, which is the whole requirement.
    """
    return torch.tensor(torch.finfo(dtype).min / 2, dtype=dtype)


def build_causal_mask(q_len: int, kv_len: int, dtype: torch.dtype, device: str):
    """Additive mask, or None when every key is visible.

    During decode q_len == 1 and the single query attends to the whole cache,
    so no mask is needed at all. During prefill the mask is rectangular, not
    square, whenever the cache is already non-empty: query row i sits at
    absolute position (kv_len - q_len + i).
    """
    if q_len == 1:
        return None
    offset = kv_len - q_len
    keys = torch.arange(kv_len, device=device)[None, :]
    rows = torch.arange(q_len, device=device)[:, None] + offset
    blocked = mask_fill_value(dtype).to(device)
    allowed = torch.zeros((), dtype=dtype, device=device)
    return torch.where(keys <= rows, allowed, blocked)[None, None]


def build_padded_causal_mask(pad_mask: torch.Tensor, q_len: int,
                             dtype: torch.dtype, device: str) -> torch.Tensor:
    """Causal mask AND padding mask, composed. Returns [B, 1, q_len, kv_len].

    Two independent reasons a key may be invisible, and they must both apply:
      - causality: a query cannot see a position after itself;
      - padding: a pad slot holds a real K/V vector (a pad token has an
        embedding and gets projected like anything else), and nothing else
        stops a real query from scoring against it.

    `pad_mask` is [B, kv_len], 1 for real tokens and 0 for padding, covering
    the WHOLE cache including tokens generated so far - not just the prompt.
    """
    _, kv_len = pad_mask.shape
    offset = kv_len - q_len
    rows = torch.arange(q_len, device=device)[:, None] + offset
    keys = torch.arange(kv_len, device=device)[None, :]
    causal = (keys <= rows)                                   # [q_len, kv_len]
    visible = causal[None, :, :] & pad_mask[:, None, :].bool()  # [B, q_len, kv_len]
    return torch.where(
        visible.unsqueeze(1),
        torch.zeros((), dtype=dtype, device=device),
        mask_fill_value(dtype).to(device),
    )


def attention(q, k, v, mask, scaling: float) -> torch.Tensor:
    """UPCAST 3/3: softmax is computed in float32, then cast back."""
    weights = torch.matmul(q, k.transpose(2, 3)) * scaling
    if mask is not None:
        weights = weights + mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(weights, v)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class InfernoQwen2:
    def __init__(self, cfg: ModelConfig, layers: list[LayerWeights],
                 embed: torch.Tensor, final_norm_w: torch.Tensor,
                 device: str, dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.layers = layers
        self.embed = embed                  # tied: also the output projection
        self.final_norm_w = final_norm_w
        self.device = device
        self.dtype = dtype
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, device)

    def _layer(self, x, lw: LayerWeights, cos, sin, cache, idx: int,
               start: int, mask) -> torch.Tensor:
        cfg = self.cfg
        b, t, _ = x.shape

        h = rms_norm(x, lw.in_norm_w, cfg.rms_eps)
        q = F.linear(h, lw.q_w, lw.q_b).view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = F.linear(h, lw.k_w, lw.k_b).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = F.linear(h, lw.v_w, lw.v_b).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)

        # RoPE BEFORE the cache write, so the cache holds already-rotated keys.
        # A key's rotation depends on its own absolute position, which never
        # changes once written - so rotating once on write is correct and
        # rotating again on read would double-apply it.
        q, k = apply_rope(q, k, cos, sin)
        k_all, v_all = cache.write(idx, start, k, v)

        a = attention(q, repeat_kv(k_all, cfg.n_rep), repeat_kv(v_all, cfg.n_rep),
                      mask, cfg.scaling)
        a = a.transpose(1, 2).reshape(b, t, cfg.n_heads * cfg.head_dim)
        x = x + F.linear(a, lw.o_w)

        h = rms_norm(x, lw.post_norm_w, cfg.rms_eps)
        h = F.linear(F.silu(F.linear(h, lw.gate_w)) * F.linear(h, lw.up_w), lw.down_w)
        return x + h

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, cache,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """One forward pass. Prefill and decode are the same code, different shapes.

        Prefill: input_ids is the whole prompt, cache.length == 0.
        Decode:  input_ids is one token, cache.length == everything so far.

        Returns logits for the LAST position only - that is all greedy sampling
        needs, and computing the 151936-wide projection at every prefill
        position would be pure waste.
        """
        b, t = input_ids.shape
        start = cache.length
        x = self.embed[input_ids]

        if pad_mask is None:
            # Unbatched. Absolute positions continue from wherever the cache
            # ended. During decode this is a single value, [start], NOT [0] -
            # a token's rotation depends on where it sits in the sequence.
            position_ids = torch.arange(start, start + t,
                                        device=self.device).unsqueeze(0).expand(b, -1)
            mask = build_causal_mask(t, start + t, x.dtype, self.device)
        else:
            # Left-padded batch. Positions are NOT arange: a sequence with
            # leading padding has its first real token at absolute position 0,
            # so position is the count of real tokens seen so far. Pad slots
            # clamp to 0 and are masked out anyway.
            position_ids = (pad_mask.cumsum(-1) - 1).clamp(min=0)[:, -t:]
            mask = build_padded_causal_mask(pad_mask, t, x.dtype, self.device)

        cos, sin = self.rope(position_ids, x.dtype)

        for idx, lw in enumerate(self.layers):
            x = self._layer(x, lw, cos, sin, cache, idx, start, mask)

        cache.advance(t)        # once per forward, after every layer has written

        x = rms_norm(x[:, -1:], self.final_norm_w, self.cfg.rms_eps)
        return F.linear(x, self.embed)      # tie_word_embeddings: no lm_head
