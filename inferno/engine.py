"""The generation loop: weight loading, tokenisation, prefill, decode.

This is the layer the parity test talks to. It replaces
`transformers.GenerationMixin.generate` for the greedy, batch-size-1 case.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from inferno.cache import KVCache
from inferno.model import InfernoQwen2, LayerWeights, ModelConfig


def _load_state_dict(model_id: str) -> dict[str, torch.Tensor]:
    """Weights come from the HuggingFace checkpoint; nothing else does."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = Path(snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"]))
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors in {path}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(str(shard)))
    return state


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class InfernoEngine:
    def __init__(self, model_id: str, device: str = "auto",
                 dtype: str = "auto", attn: str = "eager") -> None:
        if attn != "eager":
            raise NotImplementedError(
                f"attn={attn!r}: Inferno implements the eager path only. "
                "The parity reference is eager (see docs/decisions.md)."
            )
        self.device = _resolve_device(device)
        if dtype == "auto":
            dtype = "float32" if self.device == "cpu" else "float16"
        self.dtype = getattr(torch, dtype)

        hf_cfg = AutoConfig.from_pretrained(model_id)
        rope = getattr(hf_cfg, "rope_parameters", None) or {}
        self.cfg = ModelConfig(
            hidden_size=hf_cfg.hidden_size,
            n_layers=hf_cfg.num_hidden_layers,
            n_heads=hf_cfg.num_attention_heads,
            n_kv_heads=hf_cfg.num_key_value_heads,
            head_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
            rms_eps=hf_cfg.rms_norm_eps,
            # rope_theta is nested under rope_parameters in transformers 5.x
            rope_theta=rope.get("rope_theta", getattr(hf_cfg, "rope_theta", 10000.0)),
            vocab_size=hf_cfg.vocab_size,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        gen = GenerationConfig.from_pretrained(model_id)
        eos = gen.eos_token_id if gen.eos_token_id is not None else self.tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])

        self.model = self._build(_load_state_dict(model_id))

    def _build(self, sd: dict[str, torch.Tensor]) -> InfernoQwen2:
        def g(name: str) -> torch.Tensor:
            return sd[name].to(device=self.device, dtype=self.dtype)

        layers = []
        for i in range(self.cfg.n_layers):
            p = f"model.layers.{i}."
            layers.append(LayerWeights(
                q_w=g(p + "self_attn.q_proj.weight"), q_b=g(p + "self_attn.q_proj.bias"),
                k_w=g(p + "self_attn.k_proj.weight"), k_b=g(p + "self_attn.k_proj.bias"),
                v_w=g(p + "self_attn.v_proj.weight"), v_b=g(p + "self_attn.v_proj.bias"),
                o_w=g(p + "self_attn.o_proj.weight"),      # no bias in Qwen2
                gate_w=g(p + "mlp.gate_proj.weight"),
                up_w=g(p + "mlp.up_proj.weight"),
                down_w=g(p + "mlp.down_proj.weight"),
                in_norm_w=g(p + "input_layernorm.weight"),
                post_norm_w=g(p + "post_attention_layernorm.weight"),
            ))
        return InfernoQwen2(
            cfg=self.cfg, layers=layers,
            embed=g("model.embed_tokens.weight"),
            final_norm_w=g("model.norm.weight"),
            device=self.device, dtype=self.dtype,
        )

    def encode(self, prompt: str) -> torch.Tensor:
        """Same chat template R0 used - see docs/decisions.md."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 128,
                 return_cache: bool = False):
        """Greedy decode. Returns the generated token IDs, EOS included."""
        input_ids = self.encode(prompt)
        n_prompt = input_ids.shape[1]

        # Preallocated for the worst case: every sequence reserves room for
        # max_new_tokens whether it needs them or not. R4 is about this line.
        cache = KVCache(
            n_layers=self.cfg.n_layers, n_kv_heads=self.cfg.n_kv_heads,
            head_dim=self.cfg.head_dim, max_len=n_prompt + max_new_tokens,
            dtype=self.dtype, device=self.device,
        )

        # Prefill: the whole prompt at once, one big compute-bound pass.
        logits = self.model.forward(input_ids, cache)
        token = int(logits[0, -1].argmax())
        out = [token]

        # Decode: one token at a time, each pass re-reading the whole cache.
        while len(out) < max_new_tokens and token not in self.eos_ids:
            nxt = torch.tensor([[token]], device=self.device)
            logits = self.model.forward(nxt, cache)
            token = int(logits[0, -1].argmax())
            out.append(token)

        return (out, cache) if return_cache else out
