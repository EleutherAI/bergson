from contextlib import contextmanager
from typing import Generator

import torch

# =========================
# Context manager
# =========================


@contextmanager
def swap_parameters(
    module: torch.nn.Module,
    tensor_dict: dict[str, torch.Tensor],
    buffer_dict: dict[str, torch.Tensor] | None = None,
    *,
    strict: bool = True,
    preserve_graph: bool = False,
) -> Generator[dict[str, torch.Tensor], None, None]:
    """
    Temporarily replace parameter values with tensors from tensor_dict.

    tensor_dict maps parameter names (from named_parameters) to tensors.
    Works even when module parameters are on the meta device.

    When preserve_graph=True, the tensors are placed into the module directly
    (without wrapping in nn.Parameter) so that the autograd graph connecting
    the tensors to prior computations is preserved. This is required for
    correct Hessian-vector products during traced backward passes.
    """

    buffer_dict = {} if buffer_dict is None else buffer_dict.copy()
    tensor_dict = tensor_dict.copy()

    param_wrappers: dict[str, torch.Tensor] = {}
    original_params: dict[str, torch.nn.Parameter] = {}
    original_buffers: dict[str, torch.Tensor] = {}

    try:
        for name, mod in module.named_modules():
            for p_name, param in mod.named_parameters(recurse=False):
                full_name = f"{name}.{p_name}" if name else p_name
                if full_name not in tensor_dict:
                    if strict and param.requires_grad:
                        raise ValueError(
                            f"Parameter {full_name} not found in tensor_dict"
                        )
                    else:
                        continue

                original_params[full_name] = param
                new_tensor = tensor_dict.pop(full_name)

                if preserve_graph:
                    # Place the tensor directly to preserve autograd graph.
                    # This avoids nn.Parameter() which creates a new leaf node.
                    swapped = new_tensor
                    mod._parameters[p_name] = swapped  # type: ignore[assignment]
                else:
                    swapped = torch.nn.Parameter(
                        new_tensor, requires_grad=param.requires_grad
                    )
                    mod.register_parameter(p_name, swapped)

                param_wrappers[full_name] = swapped

        for name, mod in module.named_modules():
            for b_name, buffer in mod.named_buffers(recurse=False):
                full_name = f"{name}.{b_name}" if name else b_name
                if full_name not in buffer_dict:
                    if strict:
                        raise ValueError(f"Buffer {full_name} not found in buffer_dict")
                    else:
                        continue

                original_buffers[full_name] = buffer
                mod.register_buffer(b_name, buffer_dict.pop(full_name))

        if strict:
            if tensor_dict:
                raise ValueError(
                    f"tensor_dict has extra keys: {list(tensor_dict.keys())}"
                )
            if buffer_dict:
                raise ValueError(
                    f"buffer_dict has extra keys: {list(buffer_dict.keys())}"
                )

        yield param_wrappers

    finally:
        for name, original in original_params.items():
            mod, _, key = name.rpartition(".")
            mod = module.get_submodule(mod)
            mod.register_parameter(key, original)

        for name, original in original_buffers.items():
            mod, _, key = name.rpartition(".")
            mod = module.get_submodule(mod)
            mod.register_buffer(key, original)
