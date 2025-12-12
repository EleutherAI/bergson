import socket
from typing import Any, Callable
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes


def dist_worker(
    worker: Callable,
    *worker_args,
):
    try:
        worker(*worker_args)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as e:
                print(f"Barrier failed during cleanup: {e}")
                pass

            dist.destroy_process_group()


def launch_distributed_run(process_name: str, worker, const_worker_args: list[Any]):
    local_world_size = torch.cuda.device_count()
    
    # Check for multi-node environment
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        node_rank = int(os.environ.get("RANK", os.environ.get("NODE_RANK", 0)))
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ.get("MASTER_PORT", "29500")
    else:
        world_size = local_world_size
        node_rank = 0
        master_addr = "localhost"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            _, master_port = s.getsockname()
        master_port = str(master_port)

    if world_size <= 1:
        worker(0, 1, *const_worker_args)
    else:
        mp.set_sharing_strategy("file_system")

        ctx = None
        try:
            ctx = start_processes(
                process_name,
                dist_worker,
                args={
                    i: (worker, node_rank * local_world_size + i, world_size, *const_worker_args)
                    for i in range(local_world_size)
                },
                envs={
                    i: {
                        "LOCAL_RANK": str(i),
                        "RANK": str(node_rank * local_world_size + i),
                        "WORLD_SIZE": str(world_size),
                        "MASTER_ADDR": master_addr,
                        "MASTER_PORT": master_port,
                    }
                    for i in range(local_world_size)
                },
                logs_specs=DefaultLogsSpecs(),
            )
            ctx.wait()
        finally:
            if ctx is not None:
                ctx.close()