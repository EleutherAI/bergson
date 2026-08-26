import io
import os
import socket
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
from datetime import timedelta
from typing import Any, Callable, Concatenate, Mapping, ParamSpec

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes

from bergson.config.config import DistributedConfig
from bergson.utils.utils import dist_backend, dist_device_id, get_device_index


# Collective timeout for every process group bergson creates.
#
# NCCL's 10-minute default assumes ranks reach each collective at roughly the
# same time. bergson has sections where they do not: after the gradient loop in
# run_with_collector_hooks, rank 0 runs process_autocorrelation_matrices and
# writes the Hessians, while every other rank is already blocked in the
# following dist.all_reduce(total_processed). That rank-0 work grows with the
# dataset, so past the timeout the watchdog aborts a run that is making normal
# progress -- observed at 64k documents where 32k with the same config passes.
#
# torch has no environment override for this (it is an init_process_group
# argument), so it is set here and exposed via BERGSON_DIST_TIMEOUT_MIN.
DIST_TIMEOUT = timedelta(minutes=float(os.environ.get("BERGSON_DIST_TIMEOUT_MIN", 60)))


def init_dist(rank: int, local_rank: int, world_size: int) -> None:
    """Pin CUDA device and (if multi-rank) join the NCCL group set up by
    ``launch_distributed_run`` via MASTER_ADDR/MASTER_PORT env vars."""
    if torch.cuda.is_available():
        torch.cuda.set_device(get_device_index(local_rank))
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(
            dist_backend(),
            init_method=f"tcp://{addr}:{port}",
            device_id=dist_device_id(local_rank),
            rank=rank,
            timeout=DIST_TIMEOUT,
            world_size=world_size,
        )


def cap_world_size_to_dataset(
    cfg: DistributedConfig, dataset_size: int
) -> DistributedConfig:
    """Return a single node DistributedConfig for small datasets."""
    if dataset_size >= cfg.world_size:
        return cfg

    capped_cfg = deepcopy(cfg)
    capped_cfg.nnode = 1
    capped_cfg.nproc_per_node = max(1, min(dataset_size, cfg.nproc_per_node))
    return capped_cfg


def parent_barrier(dist_config: DistributedConfig) -> None:
    if dist_config.nnode <= 1:
        return
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=dist_config._node_rank,
        timeout=DIST_TIMEOUT,
        world_size=dist_config.nnode,
    )
    dist.barrier()
    dist.destroy_process_group()


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


def dist_worker(
    worker: Callable,
    rank: int,  # global
    local_rank: int,  # local
    world_size: int,
    master_addr: str,
    master_port: str,
    *worker_args,
):
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    try:
        with nullcontext() if rank == 0 else redirect_stdout(io.StringIO()):
            worker(rank, local_rank, world_size, *worker_args)
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

    if world_size <= 1:
        # Handle multi-node pipelines with single-process steps
        # Other nodes proceed to next step's NCCL rendezvous
        if dist_config._node_rank == 0:
            worker(0, 0, 1, *const_worker_args)
    else:
        mp.set_sharing_strategy("file_system")

        # Pin CUDA_VISIBLE_DEVICES per child so each only sees its assigned
        # GPU. If the parent already had a CUDA_VISIBLE_DEVICES slice, index
        # into that slice instead of overwriting it.
        parent_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        parent_cvd = (
            [d.strip() for d in parent_cuda_visible_devices.split(",") if d.strip()]
            if parent_cuda_visible_devices
            else [str(j) for j in range(local_world_size)]
        )
        assert len(parent_cvd) >= local_world_size, (
            f"CUDA_VISIBLE_DEVICES has {len(parent_cvd)} entries "
            f"({parent_cuda_visible_devices!r}) but nproc_per_node={local_world_size}"
        )

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

        # Mutate CUDA_VISIBLE_DEVICES in the parent before each spawn so
        # the child inherits it via execve, before any torch import. The
        # other distributed env vars are set inside dist_worker once the
        # child has started, since they're only read by Python code.
        spawn_ctx = mp.get_context("spawn")
        saved_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        children = []
        try:
            for i in range(local_world_size):
                os.environ["CUDA_VISIBLE_DEVICES"] = parent_cvd[i]
                p = spawn_ctx.Process(
                    target=dist_worker,
                    args=(
                        worker,
                        start_rank + i,
                        i,
                        world_size,
                        master_addr,
                        master_port,
                        *const_worker_args,
                    ),
                )
                p.start()
                children.append(p)
        finally:
            if saved_cvd is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved_cvd

        for p in children:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(
                    f"{process_name} child exited with code {p.exitcode}"
                )


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
                    "CUDA_VISIBLE_DEVICES": (
                        os.environ["CUDA_VISIBLE_DEVICES"].split(",")[i].strip()
                        if os.environ.get("CUDA_VISIBLE_DEVICES")
                        else str(i)
                    ),
                }
                for i in range(world_size)
            },
            logs_specs=DefaultLogsSpecs(),
        )
        ctx.wait()
