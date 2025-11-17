import hashlib
from typing import Any, Literal, Type, TypeVar, cast

import numpy as np
import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)  # type: ignore[return-value]


def get_layer_list(model: PreTrainedModel) -> nn.ModuleList:
    """Get the list of layers to train on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        mod
        for mod in model.base_model.modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


def create_projection_matrix(
    identifier: str,
    m: int,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    projection_type: Literal["normal", "rademacher"] = "normal",
) -> Tensor:
    """Create a projection matrix deterministically based on identifier and side."""
    # Seed the PRNG with the name of the layer and what "side" we are projecting
    message = bytes(identifier, "utf-8")
    digest = hashlib.md5(message).digest()
    seed = int.from_bytes(digest, byteorder="big") % (2**63 - 1)

    if projection_type == "normal":
        prng = torch.Generator(device).manual_seed(seed)
        A = torch.randn(m, n, device=device, dtype=dtype, generator=prng)
    elif projection_type == "rademacher":
        numpy_rng = np.random.Generator(np.random.PCG64(seed))
        random_bytes = numpy_rng.bytes((m * n + 7) // 8)
        random_bytes = np.frombuffer(random_bytes, dtype=np.uint8)
        A = np.unpackbits(random_bytes)[: m * n].reshape((m, n))
        A = torch.from_numpy(A).to(device, dtype=dtype)
        A = A.add_(-0.5).mul_(2)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")
    A /= A.norm(dim=1, keepdim=True)
    return A


def handle_arg_string(arg: str):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def simple_parse_args_string(args_string: str) -> dict[str, Any]:
    """
    Parses something like
        args1=val1,arg2=val2
    into a dictionary.
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        kv[0]: handle_arg_string("=".join(kv[1:]))
        for kv in [arg.split("=") for arg in arg_list]
    }
    return args_dict
