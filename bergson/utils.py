from typing import Any, Type, TypeVar, cast

import torch
from peft import PeftModel
from torch import nn
from transformers import PreTrainedModel

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)  # type: ignore[return-value]


def get_layer_list(model: PreTrainedModel | PeftModel) -> nn.ModuleList:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        mod
        for mod in model.base_model.modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


def get_device(rank: int = 0) -> str:
    """Get device string for the given rank.

    Returns "cpu" if CUDA is not available.
    """
    return f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
