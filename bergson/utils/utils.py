import os
import random
from typing import Any, Literal, Type, TypeVar, cast

import numpy as np
import torch
from ml_dtypes import bfloat16
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


def setup_reproducibility():
    """Setup reproducibility for distributed training"""
    print("WARNING: Running in debug mode, much slower performance expected.")
    seed: int = 42
    # Set all random seeds - same across all ranks for model consistency
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Force deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Environment variables for determinism
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def handle_arg_string(arg: str):
    # Handle lists
    if "|" in arg:
        return [handle_arg_string(v) for v in arg.split("|")]

    # Handle integers
    try:
        return int(arg)
    except ValueError:
        pass

    # Handle floats
    try:
        return float(arg)
    except ValueError:
        pass

    # Handle booleans
    match arg.lower():
        case "true":
            return True
        case "false":
            return False
        case _:
            return arg


def simple_parse_kwargs_string(args_string: str) -> dict:
    """Parses something like `args1=val1,arg2=val2` into a dictionary."""
    args_dict = {}

    for elem in args_string.split(","):
        lvalue, sep, rvalue = elem.partition("=")

        # Ignore whitespace
        lvalue = lvalue.strip()
        rvalue = rvalue.strip()

        if not (lvalue and sep):
            raise ValueError(f"Invalid argument: '{elem}'. Expected format key=value.")

        if not lvalue.isidentifier():
            raise ValueError(f"Invalid key: '{lvalue}'. Must be a valid identifier.")

        args_dict[lvalue] = handle_arg_string(rvalue)

    return args_dict


DTYPE_BY_PRIORITY = {
    torch.float64: 0,
    torch.float32: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.float8_e5m2: 4,
}


def get_gradient_dtype(model) -> torch.dtype:
    """Returns the gradient dtype for a model.
    If multiple dtypes are found, return the first present dtype in
    [torch.float64, torch.float32, torch.float16, torch.bfloat16].
    """
    dtypes = set()
    for p in model.parameters():
        if p.requires_grad and p.dtype.is_floating_point:
            dtypes.add(p.dtype)

    assert len(dtypes) > 0, "No trainable parameters found"

    if len(dtypes) == 1:
        return dtypes.pop()
    else:
        dtypes_by_priority = sorted(dtypes, key=lambda x: dtypes_by_priority[x])
        print(
            f"Multiple gradient dtypes found: {dtypes}. Using {dtypes_by_priority[0]}."
        )
        return dtypes_by_priority[0]


np_bfloat16 = np.dtype(bfloat16)


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    """Convert a torch tensor to numpy array, handling bfloat16.

    PyTorch's .numpy() doesn't support bfloat16, so we view the tensor
    as uint16 (same bit width) and reinterpret as ml_dtypes.bfloat16.
    This preserves the exact bit pattern without lossy float conversion.
    """
    if tensor.dtype != torch.bfloat16:
        return tensor.numpy()

    return tensor.view(torch.uint16).numpy().view(bfloat16)


def numpy_to_tensor(arr: np.ndarray) -> Tensor:
    """Convert a numpy array to torch tensor, handling bfloat16.

    PyTorch's from_numpy() doesn't support ml_dtypes bfloat16, so we view
    the array as uint16 and reinterpret as torch.bfloat16.
    This preserves the exact bit pattern without lossy float conversion.

    Also handles V2 void types from structured arrays, which represent
    bfloat16 values as 2-byte voids.
    """
    if arr.dtype == np.dtype(bfloat16):
        return torch.from_numpy(arr.view(np.uint16).copy()).view(torch.bfloat16)

    # Handle V2 voids (bfloat16 from structured arrays)
    if arr.dtype.str == "|V2":
        return torch.from_numpy(arr.view(np.uint16).copy()).view(torch.bfloat16)

    return torch.from_numpy(arr)


def convert_dtype_to_np(dtype: torch.dtype) -> np.dtype:
    """Convert a torch dtype to the corresponding numpy dtype."""
    match dtype:
        case torch.float16:
            return np.dtype(np.float16)
        case torch.float32:
            return np.dtype(np.float32)
        case torch.float64:
            return np.dtype(np.float64)
        case torch.bfloat16:
            return np.dtype(bfloat16)
        case _:
            raise ValueError(f"Unsupported torch dtype: {dtype}")


def convert_dtype_to_torch(dtype: np.dtype) -> torch.dtype:
    """Convert a numpy dtype to the corresponding torch dtype."""
    match dtype:
        case np.float16:
            return torch.float16
        case np.float32:
            return torch.float32
        case np.float64:
            return torch.float64
        case _:
            raise ValueError(f"Unsupported numpy dtype: {dtype}")


def convert_precision_to_torch(
    precision: Literal["auto", "bf16", "fp16", "fp32"],
) -> torch.dtype:
    """Convert a precision string to the corresponding torch dtype."""
    match precision:
        case "auto":
            raise ValueError(
                "Precision 'auto' is not supported for conversion to torch dtype."
            )
        case "bf16":
            return torch.bfloat16
        case "fp16":
            return torch.float16
        case "fp32":
            return torch.float32


def get_device(rank: int = 0) -> str:
    """Get device string for the given rank.

    Returns "cpu" if CUDA is not available.
    """
    return f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
