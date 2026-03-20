import os
import socket
from collections import defaultdict
from contextlib import nullcontext, redirect_stdout
from typing import Any, Callable, Concatenate, Mapping, ParamSpec

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
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

from bergson.config import DistributedConfig


def grad_tree(
    outputs: torch.Tensor,
    inputs: Mapping[str, torch.Tensor],
    grad_outputs: dict[str, torch.Tensor] | None = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Compute grads of loss wrt inputs dict, returning a dict with the same keys.

    Args:
        outputs: The output tensor to compute gradients for.
        inputs: A dict of input tensors to compute gradients with respect to.
        grad_outputs: Optional dict of gradient outputs for each output tensor.
        **kwargs: Additional keyword arguments to pass to torch.autograd.grad.
    """
    if grad_outputs is not None:
        kwargs["grad_outputs"] = list(grad_outputs.values())

    grads = torch.autograd.grad(
        outputs,
        list(inputs.values()),
        **kwargs,
        allow_unused=True,
    )
    return dict(zip(inputs, grads))


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
                t2 = DTensor.from_local(t.to_local(), t.device_mesh, t.placements)
            else:
                t2 = torch.Tensor(t.data)
            t2.requires_grad_(t.requires_grad)
            seen[tid] = t2

        result[path] = seen[tid]

    return result


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
            distribute_tensor(param, placements=(Shard(0),))
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


def dist_worker(
    worker: Callable,
    *worker_args,
):
    try:
        rank = int(os.environ.get("RANK", 0))
        with nullcontext() if rank == 0 else redirect_stdout(None):
            worker(*worker_args)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                print(f"Barrier failed during cleanup: {e}")
                pass

            dist.destroy_process_group()


def launch_distributed_run(
    process_name: str,
    worker,
    const_worker_args: list[Any],
    dist_config: DistributedConfig | None = None,
):
    if dist_config is None:
        dist_config = DistributedConfig()

    local_world_size = dist_config.nproc_per_node
    world_size = dist_config.world_size
    start_rank = dist_config.start_rank

    # Multi-node environment
    if dist_config.nnode > 1:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
    else:
        master_addr = "localhost"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, master_port = s.getsockname()
        master_port = str(master_port)

    if world_size <= 1:
        worker(0, 0, 1, *const_worker_args)
    else:
        mp.set_sharing_strategy("file_system")

        ctx = None
        try:
            ctx = start_processes(
                process_name,
                dist_worker,
                args={
                    i: (worker, start_rank + i, i, world_size, *const_worker_args)
                    for i in range(local_world_size)
                },
                envs={
                    i: {
                        "LOCAL_RANK": str(i),
                        "RANK": str(start_rank + i),
                        "WORLD_SIZE": str(world_size),
                        "MASTER_ADDR": master_addr,
                        "MASTER_PORT": master_port,
                    }
                    for i in range(local_world_size)
                },
                logs_specs=DefaultLogsSpecs(),
            )
            result = ctx.wait()

            if result is not None and hasattr(result, "failures") and result.failures:
                newline = "\n"
                raise RuntimeError(
                    f"{process_name} failed with {len(result.failures)} process "
                    f"failure(s): {newline.join([str(f) for f in result.failures])}"
                )
        finally:
            if ctx is not None:
                ctx.close()  # Kill any processes that are still running


Args = ParamSpec("Args")
Worker = Callable[Concatenate[int, int, int, Args], None]
"""A worker function for distributed training."""


def simple_dist_worker(rank: int, world_size: int, dataset, worker: Worker):
    try:
        worker(rank, world_size, dataset)
    finally:
        dist.destroy_process_group()


def dist_main(dataset, worker: Worker):
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # Run the worker directly if no distributed training is needed. This is great
        # for debugging purposes.
        worker(0, 1, dataset)
    else:
        # Set up multiprocessing and distributed training
        mp.set_sharing_strategy("file_system")

        # Find an available port for distributed training
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, port = s.getsockname()

        ctx = start_processes(
            "train",
            simple_dist_worker,
            args={i: (i, world_size, dataset, worker) for i in range(world_size)},
            envs={
                i: {
                    "LOCAL_RANK": str(i),
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": str(port),
                }
                for i in range(world_size)
            },
            logs_specs=DefaultLogsSpecs(),
        )
        ctx.wait()
