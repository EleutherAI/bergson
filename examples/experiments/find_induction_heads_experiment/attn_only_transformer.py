"""Attention-only transformer for induction head experiments."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class AttnOnlyConfig(PretrainedConfig):
    model_type = "attn_only"

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2048,
        layer_norm_epsilon=1e-5,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=True,
        layer_norm=False,
        special_pos_embed=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_epsilon = layer_norm_epsilon
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.use_cache = use_cache
        self.layer_norm = layer_norm
        self.special_pos_embed = special_pos_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config: AttnOnlyConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.special_pos_embed = config.special_pos_embed
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.max_position_embeddings, config.max_position_embeddings
                )
            ).view(
                1, 1, config.max_position_embeddings, config.max_position_embeddings
            ),
            persistent=False,
        )

    def _split_heads(self, x):
        B, T, C = x.shape
        x = x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return x

    def _merge_heads(self, x):
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)

    def forward(
        self,
        x,
        pos_emb,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        # add position to q and k only
        if self.special_pos_embed:
            q = q + pos_emb
            k = k + pos_emb

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if layer_past is not None:
            pk, pv = layer_past
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:, :, :T, : k.size(-2)]
        att = att.masked_fill(causal == 0, float("-inf"))
        if attn_mask is not None:
            att = att + attn_mask
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = self._merge_heads(y)
        y = self.resid_drop(self.c_proj(y))

        present = (k, v) if use_cache else None
        return y, present


class AttnOnlyBlock(nn.Module):
    def __init__(self, config: AttnOnlyConfig):
        super().__init__()
        if config.layer_norm:
            self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.ln_1 = None
        self.attn = CausalSelfAttention(config)

    def forward(
        self,
        x,
        pos_emb,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if self.ln_1 is not None:
            x = self.ln_1(x)

        a, present = self.attn(
            x, pos_emb, layer_past=layer_past, use_cache=use_cache, attn_mask=attn_mask
        )
        x = x + a
        return x, present


class AttnOnlyForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = AttnOnlyConfig

    def __init__(self, config: AttnOnlyConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [AttnOnlyBlock(config) for _ in range(config.num_hidden_layers)]
        )
        if config.layer_norm:
            self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.ln_f = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # HF helpers
    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_emb):
        self.wte = new_emb

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_lm_head):
        self.lm_head = new_lm_head

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        B, T = input_ids.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids)

        pos_emb = self.wpe(pos)
        if not self.config.special_pos_embed:
            x = x + pos_emb

        x = self.drop(x)

        presents = []
        for i, block in enumerate(self.h):
            layer_past = None if past_key_values is None else past_key_values[i]
            x, present = block(
                x,
                pos_emb,
                layer_past=layer_past,
                use_cache=self.config.use_cache if use_cache is None else use_cache,
            )
            if present is not None:
                presents.append(present)

        if self.ln_f is not None:
            x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents if presents else None,
            hidden_states=None,
            attentions=None,
        )


AutoConfig.register("attn_only", AttnOnlyConfig)
AutoModelForCausalLM.register(AttnOnlyConfig, AttnOnlyForCausalLM)
