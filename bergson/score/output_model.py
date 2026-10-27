"""Forward-AD-compatible bounded-memory attention and token loss for OLMo3."""

from contextvars import ContextVar

import torch
from torch.nn import functional as F
from transformers import AttentionInterface


TOKEN_LOSS_REQUEST = ContextVar('bergson_output_token_loss', default=None)


def chunked_attention(module, query, key, value, attention_mask=None, dropout=0.0,
                      scaling=None, sliding_window=None, **kwargs):
    if dropout:
        raise ValueError('Output influence requires deterministic evaluation')
    groups = query.shape[1]//key.shape[1]
    if groups != 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    size = getattr(module, '_output_query_chunk', 256)
    scale = scaling if scaling is not None else query.shape[-1]**-0.5
    key_positions = torch.arange(key.shape[-2], device=query.device)
    offset = key.shape[-2]-query.shape[-2]
    outputs = []
    for start in range(0, query.shape[-2], size):
        end = min(start+size, query.shape[-2])
        positions = torch.arange(start+offset, end+offset, device=query.device)
        valid = key_positions[None, :] <= positions[:, None]
        if sliding_window is not None:
            valid = valid & (key_positions[None, :] > positions[:, None]-sliding_window)
        scores = (query[:, :, start:end] @ key.transpose(-1, -2))*scale
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            mask = attention_mask[:, :, start:end, :key.shape[-2]]
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min) if mask.dtype == torch.bool else scores+mask
        probabilities = scores.float().softmax(-1).to(scores.dtype) if scores.dtype == torch.bfloat16 else scores.softmax(-1)
        outputs.append(probabilities @ value)
    return torch.cat(outputs, dim=-2).transpose(1, 2).contiguous(), None


def register_output_attention():
    # An unregistered mask backend makes HF omit its dense SxS mask; the
    # attention function builds the causal/sliding mask for each query chunk.
    AttentionInterface.register('bergson_output_jvp', chunked_attention)


def install_output_loss_head(model, chunk_rows=128, checkpoint_reverse=False):
    """Return token losses only in a scoped JVP call; normal calls keep logits."""
    head = model.get_output_embeddings()
    if getattr(head, '_bergson_output_loss_head', False):
        return model
    original = head.forward

    def token_losses(hidden, labels, smoothing):
        labels = labels[:, 1:].to(hidden.device)
        shape = labels.shape
        mask = labels.reshape(-1) != -100
        positions = mask.nonzero(as_tuple=True)[0]
        targets = labels.reshape(-1).index_select(0, positions)
        selected = hidden[:, :-1].reshape(-1, hidden.shape[-1]).index_select(0, positions)
        chunks = []
        for start in range(0, len(positions), chunk_rows):
            # Keep the vocabulary projection and CE in FP32 in mixed-precision trials.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                inputs = selected[start:start+chunk_rows].to(head.weight.dtype)
                logits = F.linear(inputs, head.weight, head.bias)
                chunks.append(F.cross_entropy(logits, targets[start:start+chunk_rows],
                                               reduction='none', label_smoothing=smoothing))
        values = torch.cat(chunks) if chunks else hidden.new_empty(0)
        return values.new_zeros(labels.numel()).index_copy(0, positions, values).reshape(shape)

    def forward(hidden):
        request = TOKEN_LOSS_REQUEST.get()
        if request is None:
            return original(hidden)
        labels, smoothing = request
        if checkpoint_reverse and torch.is_grad_enabled():
            # One checkpoint for the entire head avoids saving one full frozen
            # vocabulary weight matrix per token chunk in the CPU offloader.
            from torch.utils.checkpoint import checkpoint
            return checkpoint(lambda value: token_losses(value, labels, smoothing),
                              hidden, use_reentrant=False)
        return token_losses(hidden, labels, smoothing)

    head.forward = forward
    head._bergson_output_loss_head = True
    model._bergson_output_loss_head = True
    # HF RMSNorm otherwise explicitly converts FP64 inputs to FP32. Keep
    # existing FP32 behavior, but retain FP64 when the query requires it.
    for norm in model.modules():
        if norm.__class__.__name__ != 'Olmo3RMSNorm':
            continue
        old_forward = norm.forward

        def norm_forward(hidden, module=norm, original_forward=old_forward):
            if hidden.dtype != torch.float64:
                return original_forward(hidden)
            variance = hidden.square().mean(-1, keepdim=True)
            return module.weight*hidden*torch.rsqrt(variance+module.variance_epsilon)

        norm.forward = norm_forward
    return model


def efficient_attention(module, query, key, value, attention_mask=None, dropout=0.0,
                        scaling=None, sliding_window=None, **kwargs):
    """FP32 reverse-mode SDPA with expanded GQA and no quadratic math fallback."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if query.shape[-2] != key.shape[-2]:
        raise ValueError('Attribution efficient attention requires full sequences without KV cache')
    groups=query.shape[1]//key.shape[1]
    if groups != 1:
        key=key.repeat_interleave(groups,dim=1)
        value=value.repeat_interleave(groups,dim=1)
    mask=attention_mask
    if sliding_window is not None:
        positions=torch.arange(query.shape[-2],device=query.device)
        delta=positions[:,None]-positions[None,:]
        causal=(delta>=0)&(delta<sliding_window)
        if mask is None:
            mask=causal[None,None]
        elif mask.dtype==torch.bool:
            mask=mask & causal
        else:
            mask=mask.masked_fill(~causal,torch.finfo(mask.dtype).min)
    if mask is not None:
        # Convert boolean bias ourselves: SDPA otherwise materializes it with
        # unaligned row/head strides for lengths not divisible by four.
        if mask.dtype == torch.bool:
            mask=torch.zeros_like(mask,dtype=query.dtype).masked_fill(~mask,float("-inf"))
        width=mask.shape[-1]
        mask=F.pad(mask,(0,(-width)%8))[..., :width]
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        result=F.scaled_dot_product_attention(query,key,value,attn_mask=mask,
                    dropout_p=dropout,scale=scaling,is_causal=mask is None)
    return result.transpose(1,2).contiguous(),None


def register_efficient_attention():
    AttentionInterface.register('bergson_efficient_sdpa',efficient_attention)
