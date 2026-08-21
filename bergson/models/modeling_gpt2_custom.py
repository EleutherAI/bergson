"""GPT-2 with optional architecture modifications, for attribution experiments.

One class, three architectures selected by ``config.arch_mod``:

- ``"none"``: exactly stock GPT-2. No extra modules, no extra parameters; stock
  ``gpt2`` checkpoints load with zero missing/unexpected keys, and logits match
  ``GPT2LMHeadModel`` bit-for-bit (tested).
- ``"qk_norm"``: the OLMo 2 convention — RMSNorm applied to the query and key
  projections over their full width, before the head split, one norm pair per
  layer.
- ``"preact_layernorm"``: LayerNorm on the MLP pre-activation (between ``c_fc``
  and the GELU), one per layer.

The modifications attach as forward hooks on the existing ``c_attn`` / ``c_fc``
modules rather than replacing any module or copying any forward method, so the
stock state-dict layout, cache handling, and attention implementation are
inherited unchanged from the installed ``transformers`` — the diff against
stock GPT-2 is exactly the new norm parameters and nothing else. This is what
makes the "effect of the modification" separable from "effect of
reimplementing GPT-2" (metasmoothness DECISIONS.md, D10).
"""

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel

ARCH_MODS = ("none", "qk_norm", "preact_layernorm")


class GPT2CustomConfig(GPT2Config):
    model_type = "gpt2_custom"

    def __init__(self, arch_mod: str = "none", qk_norm_eps: float = 1e-6, **kwargs):
        if arch_mod not in ARCH_MODS:
            raise ValueError(f"arch_mod must be one of {ARCH_MODS}, got {arch_mod!r}")
        self.arch_mod = arch_mod
        self.qk_norm_eps = qk_norm_eps
        super().__init__(**kwargs)


class GPT2CustomLMHeadModel(GPT2LMHeadModel):
    config_class = GPT2CustomConfig

    def __init__(self, config: GPT2CustomConfig):
        super().__init__(config)
        if config.arch_mod == "qk_norm":
            for block in self.transformer.h:
                attn = block.attn
                # OLMo 2 applies the norm to the full projected q/k vectors
                # before the head reshape; c_attn packs [q | k | v] along the
                # last dim, so norming the first two thirds of its output is
                # the same placement.
                attn.q_norm = nn.RMSNorm(attn.embed_dim, eps=config.qk_norm_eps)
                attn.k_norm = nn.RMSNorm(attn.embed_dim, eps=config.qk_norm_eps)
                attn.c_attn.register_forward_hook(_qk_norm_hook(attn))
        elif config.arch_mod == "preact_layernorm":
            inner = config.n_inner if config.n_inner is not None else 4 * config.n_embd
            for block in self.transformer.h:
                block.mlp.preact_ln = nn.LayerNorm(inner, eps=config.layer_norm_epsilon)
                block.mlp.c_fc.register_forward_hook(_preact_ln_hook(block.mlp))
        # Re-run weight init so freshly added norms get their documented init;
        # existing weights are re-initialized identically (init is deterministic
        # per-module) and overwritten anyway when loading a checkpoint.
        self.post_init()


def _qk_norm_hook(attn):
    def hook(module, args, output):
        q, k, v = output.split(attn.split_size, dim=2)
        return torch.cat([attn.q_norm(q), attn.k_norm(k), v], dim=2)

    return hook


def _preact_ln_hook(mlp):
    def hook(module, args, output):
        return mlp.preact_ln(output)

    return hook


AutoConfig.register("gpt2_custom", GPT2CustomConfig)
AutoModelForCausalLM.register(GPT2CustomConfig, GPT2CustomLMHeadModel)
