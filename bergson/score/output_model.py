"""Bounded-memory attention and token losses for long sequences.

``install_output_loss_head`` makes a model's output layer return per-token
losses, computed in chunks of tokens, while ``TOKEN_LOSS_REQUEST`` is set, so
the full ``[batch, seq, vocab]`` logits are never stored. The two attention
functions avoid the dense attention mask: ``chunked_attention`` for forward-mode
products and ``efficient_attention`` for the backward pass.
"""

from contextvars import ContextVar

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import AttentionInterface

TOKEN_LOSS_REQUEST = ContextVar("bergson_token_loss_request", default=None)
"""``(labels, label_smoothing)`` while a caller wants per-token losses from the
output layer instead of logits."""


def chunked_attention(
    module,
    query,
    key,
    value,
    attention_mask=None,
    dropout=0.0,
    scaling=None,
    sliding_window=None,
    **kwargs,
):
    """Causal attention over chunks of query positions, building each chunk's
    mask itself; supports forward-mode products."""
    if dropout:
        raise ValueError("Output token influence requires deterministic evaluation")
    groups = query.shape[1] // key.shape[1]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    size = getattr(module, "_output_query_chunk", 256)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    key_positions = torch.arange(key.shape[-2], device=query.device)
    offset = key.shape[-2] - query.shape[-2]
    outputs = []
    for start in range(0, query.shape[-2], size):
        end = min(start + size, query.shape[-2])
        positions = torch.arange(start + offset, end + offset, device=query.device)
        valid = key_positions[None, :] <= positions[:, None]
        if sliding_window is not None:
            valid = valid & (
                key_positions[None, :] > positions[:, None] - sliding_window
            )
        scores = (query[:, :, start:end] @ key.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            mask = attention_mask[:, :, start:end, : key.shape[-2]]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask
        if scores.dtype == torch.bfloat16:
            probabilities = scores.float().softmax(-1).to(scores.dtype)
        else:
            probabilities = scores.softmax(-1)
        outputs.append(probabilities @ value)
    return torch.cat(outputs, dim=-2).transpose(1, 2).contiguous(), None


def register_output_attention():
    """Register ``chunked_attention`` as ``bergson_output_jvp``. HF builds no dense
    mask for an attention function it doesn't know."""
    AttentionInterface.register("bergson_output_jvp", chunked_attention)


def install_output_loss_head(model, chunk_rows: int = 128, checkpoint_reverse=False):
    """Make the output layer return per-token losses while ``TOKEN_LOSS_REQUEST``
    is set. Other calls still return logits.

    The losses are computed for the labelled positions only, ``chunk_rows`` at a
    time, in the head's precision. With ``checkpoint_reverse`` the whole head is
    recomputed in the backward pass, so its saved activations are never stored.
    """
    head = model.get_output_embeddings()
    if getattr(head, "_bergson_output_loss_head", False):
        return model
    original = head.forward

    def token_losses(hidden, labels, smoothing):
        labels = labels[:, 1:].to(hidden.device)
        shape = labels.shape
        positions = (labels.reshape(-1) != -100).nonzero(as_tuple=True)[0]
        targets = labels.reshape(-1).index_select(0, positions)
        selected = (
            hidden[:, :-1].reshape(-1, hidden.shape[-1]).index_select(0, positions)
        )
        chunks = []
        for start in range(0, len(positions), chunk_rows):
            # Keep the vocabulary projection and its loss out of autocast
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                inputs = selected[start : start + chunk_rows].to(head.weight.dtype)
                logits = F.linear(inputs, head.weight, head.bias)
                # Keep at least fp32 losses, as sequence_log_probs does
                if logits.dtype != torch.float64:
                    logits = logits.float()
                chunks.append(
                    F.cross_entropy(
                        logits,
                        targets[start : start + chunk_rows],
                        reduction="none",
                        label_smoothing=smoothing,
                    )
                )
        values = (
            torch.cat(chunks)
            if chunks
            else hidden.new_empty(0, dtype=torch.promote_types(hidden.dtype, torch.float32))
        )
        losses = values.new_zeros(labels.numel()).index_copy(0, positions, values)
        return losses.reshape(shape)

    def forward(hidden):
        request = TOKEN_LOSS_REQUEST.get()
        if request is None:
            return original(hidden)
        labels, smoothing = request
        if checkpoint_reverse and torch.is_grad_enabled():
            return checkpoint(
                lambda value: token_losses(value, labels, smoothing),
                hidden,
                use_reentrant=False,
            )
        return token_losses(hidden, labels, smoothing)

    head.forward = forward
    head._bergson_output_loss_head = True
    model._bergson_output_loss_head = True

    # HF's RMSNorm converts fp64 inputs to fp32; keep fp64 inputs in fp64
    for norm in model.modules():
        if norm.__class__.__name__ != "Olmo3RMSNorm":
            continue

        def norm_forward(hidden, module=norm, original_forward=norm.forward):
            if hidden.dtype != torch.float64:
                return original_forward(hidden)
            variance = hidden.square().mean(-1, keepdim=True)
            return (
                module.weight * hidden * torch.rsqrt(variance + module.variance_epsilon)
            )

        norm.forward = norm_forward
    return model


def efficient_attention(
    module,
    query,
    key,
    value,
    attention_mask=None,
    dropout=0.0,
    scaling=None,
    sliding_window=None,
    **kwargs,
):
    """Memory-efficient SDPA for the backward pass, with grouped key and value
    heads expanded and the sliding window folded into the mask."""
    if query.shape[-2] != key.shape[-2]:
        raise ValueError(
            "efficient_attention requires full sequences without a KV cache"
        )
    groups = query.shape[1] // key.shape[1]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    mask = attention_mask
    if sliding_window is not None:
        positions = torch.arange(query.shape[-2], device=query.device)
        delta = positions[:, None] - positions[None, :]
        causal = (delta >= 0) & (delta < sliding_window)
        if mask is None:
            mask = causal[None, None]
        elif mask.dtype == torch.bool:
            mask = mask & causal
        else:
            mask = mask.masked_fill(~causal, torch.finfo(mask.dtype).min)
    if mask is not None:
        # Convert a boolean mask ourselves: SDPA otherwise forms the bias with
        # unaligned strides for lengths not divisible by eight
        if mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=query.dtype).masked_fill(
                ~mask, float("-inf")
            )
        width = mask.shape[-1]
        mask = F.pad(mask, (0, (-width) % 8))[..., :width]
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        result = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=mask is None,
        )
    return result.transpose(1, 2).contiguous(), None


def register_efficient_attention():
    """Register ``efficient_attention`` as ``bergson_efficient_sdpa``."""
    AttentionInterface.register("bergson_efficient_sdpa", efficient_attention)
