"""Recompute OLMo3 layers or sublayers in the backward pass without collecting their
forward statistics twice."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps

import torch.nn as nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

IN_RECOMPUTE = ContextVar("bergson_in_recompute", default=False)
"""True while a checkpointed sublayer is being recomputed in the backward pass,
so forward hooks can skip the repeated call."""


@contextmanager
def recompute_context():
    token = IN_RECOMPUTE.set(True)
    try:
        yield
    finally:
        IN_RECOMPUTE.reset(token)


def _checkpoint_modules(model: nn.Module, class_name: str) -> int:
    """Wrap the forward of every ``class_name`` module in an activation
    checkpoint, returning how many were wrapped."""
    count = 0
    for module in model.modules():
        if module.__class__.__name__ != class_name:
            continue
        if getattr(module, "_bergson_checkpointed", False):
            continue
        original = module.forward

        @wraps(original)
        def forward(*args, _original=original, **kwargs):
            if kwargs.get("past_key_values") is not None:
                raise ValueError("Recomputing sublayers requires KV caching disabled")
            # Replay the whole sublayer, so the forward hooks of its last
            # modules also run again
            with set_checkpoint_early_stop(False):
                return checkpoint(
                    _original,
                    *args,
                    use_reentrant=False,
                    context_fn=lambda: (nullcontext(), recompute_context()),
                    **kwargs,
                )

        module.forward = forward
        module._bergson_checkpointed = True
        for submodule in module.modules():
            submodule._bergson_recomputed = True
        count += 1
    if not count:
        raise ValueError(f"No {class_name} modules found to checkpoint")
    return count


def checkpoint_mlps(model: nn.Module) -> int:
    """Recompute each OLMo3 MLP in the backward pass."""
    return _checkpoint_modules(model, "Olmo3MLP")


def checkpoint_attentions(model: nn.Module) -> int:
    """Recompute each OLMo3 attention block in the backward pass."""
    # Replaying attention must never append keys and values to a KV cache
    model.config.use_cache = False
    return _checkpoint_modules(model, "Olmo3Attention")


def checkpoint_layers(model: nn.Module) -> int:
    """Recompute each OLMo3 decoder layer in the backward pass."""
    # Replaying attention must never append keys and values to a KV cache
    model.config.use_cache = False
    return _checkpoint_modules(model, "Olmo3DecoderLayer")
