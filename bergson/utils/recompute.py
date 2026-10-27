"""Recompute MLP activations without collecting their forward statistics twice."""
from contextvars import ContextVar
from contextlib import contextmanager,nullcontext
from functools import wraps
from torch.utils.checkpoint import checkpoint

IN_RECOMPUTE = ContextVar('bergson_in_mlp_recompute',default=False)

@contextmanager
def recompute_context():
    token=IN_RECOMPUTE.set(True)
    try:yield
    finally:IN_RECOMPUTE.reset(token)


def _checkpoint_modules(model, class_name):
    count=0
    for module in model.modules():
        if module.__class__.__name__!=class_name or getattr(module,'_bergson_checkpoint_sublayer',False):continue
        original=module.forward
        @wraps(original)
        def forward(*args,_original=original,**kwargs):
            if kwargs.get("past_key_values") is not None:
                raise ValueError("Attribution recomputation requires KV caching disabled")
            return checkpoint(_original,*args,use_reentrant=False,
                              context_fn=lambda:(nullcontext(),recompute_context()),**kwargs)
        module.forward=forward
        module._bergson_checkpoint_sublayer=True
        count+=1
    if not count:raise ValueError(f'No {class_name} modules found for activation checkpointing')
    return count


def checkpoint_mlps(model):
    return _checkpoint_modules(model,'Olmo3MLP')


def checkpoint_attentions(model):
    # Replaying attention must never append keys/values to a mutable KV cache.
    model.config.use_cache = False
    return _checkpoint_modules(model,'Olmo3Attention')
