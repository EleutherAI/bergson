import hashlib
import os
import random
from typing import TYPE_CHECKING, Any, Literal, Type, TypeVar, cast

import numpy as np
import torch
from ml_dtypes import bfloat16
from torch import Tensor, nn
from transformers import PreTrainedModel

if TYPE_CHECKING:
    from bergson.collector.gradient_collectors import GradientCollector


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


def validate_batch_size(
    model: PreTrainedModel,
    token_batch_size: int | None,
    collector: "GradientCollector",
):
    """Validate that the specified token batch size fits on device."""
    if token_batch_size is None:
        return

    # Check that token_batch_size doesn't exceed model's max sequence length
    max_seq_len = getattr(model.config, "max_position_embeddings", None)
    if max_seq_len is not None and token_batch_size > max_seq_len:
        raise ValueError(
            f"Token batch size {token_batch_size} exceeds model's max sequence length "
            f"({max_seq_len}). Use --token_batch_size {max_seq_len} or smaller."
        )

    random_tokens = torch.randint(
        0, 10, (1, token_batch_size), device=model.device, dtype=torch.long
    )
    try:
        with collector:
            loss = model(random_tokens).logits[0, 0, 0].float()
            loss.backward()
            model.zero_grad()
    except Exception as e:
        raise ValueError(
            f"Token batch size {token_batch_size} is too large for the device. "
            f"Try reducing the batch size or use --fsdp to shard the model."
        ) from e


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
