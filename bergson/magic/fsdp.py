from collections import defaultdict

import torch
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.nn.utils.parametrize import register_parametrization
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint,
    create_selective_checkpoint_contexts,
)


def fsdp_policy():
    def _fsdp_recomp_policy():
        def _custom_policy(ctx, func, *args, **kwargs):
            to_recompute = func in {
                torch.ops._c10d_functional.all_gather_into_tensor.default,  # type: ignore[attr-defined]
                torch.ops._c10d_functional.wait_tensor.default,  # type: ignore[attr-defined]
            }
            return (
                CheckpointPolicy.MUST_RECOMPUTE
                if to_recompute
                else CheckpointPolicy.MUST_SAVE
            )

        return _custom_policy

    return create_selective_checkpoint_contexts(_fsdp_recomp_policy())


class ReplicateComputation(torch.nn.Module):
    def replicate_compute(self, x):
        return x.redistribute(
            placements=(Replicate(),),
        ).to_local(grad_placements=(Partial(reduce_op="avg"),))

    def forward(self, x):
        return checkpoint(
            self.replicate_compute, x, use_reentrant=False, context_fn=fsdp_policy
        )


def simple_fsdp(model: torch.nn.Module) -> torch.nn.Module:
    """SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.compile"""
    # For each unique parameter, construct a list of the places in the model where it
    # appears. This is a bit wonky, but it is the best way to handle tied weights.
    param_to_paths = defaultdict(list)
    for path, param in model.named_parameters(remove_duplicate=False):
        param_to_paths[param].append(path)

    # Use a while loop to avoid modifying the dict while iterating over it. We don't
    # want to hold onto both the original and distributed versions of each parameter.
    while param_to_paths:
        param, paths = param_to_paths.popitem()

        # Create a new distributed version of this param
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, placements=(Shard(0),)),
            requires_grad=param.requires_grad,
        )

        # Update all occurrences of this parameter in the model
        for path in paths:
            # Find the module that has a reference to this parameter
            mod_name, _, p_name = path.rpartition(".")
            mod = model.get_submodule(mod_name)

            # Re-register the parameter with sharding and replication
            mod.register_parameter(p_name, dist_param)
            register_parametrization(
                mod,
                p_name,
                ReplicateComputation(),
                unsafe=True,
            )

    return model


def shallow_copy(tensor_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create a shallow copy of a dict of tensors, handling tied weights.

    Preserves the original key order. All paths that shared the same tensor
    (tied weights) will point to the same copied tensor in the output.
    """
    seen: dict[int, torch.Tensor] = {}  # id(original) -> copied tensor
    result: dict[str, torch.Tensor] = {}

    for path, t in tensor_dict.items():
        tid = id(t)
        if tid not in seen:
            if isinstance(t, DTensor):
                t2 = DTensor.from_local(
                    t.to_local(),
                    t.device_mesh,
                    t.placements,
                    shape=t.shape,
                    stride=t.stride(),
                )
            else:
                t2 = torch.Tensor(t.data)
            t2.requires_grad_(t.requires_grad)
            seen[tid] = t2

        result[path] = seen[tid]

    return result
