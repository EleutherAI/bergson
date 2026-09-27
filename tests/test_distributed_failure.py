"""A rank that raises must fail the launch quickly, not leave the other ranks
waiting in a collective until the process group times out."""

import os
import time

import pytest
import torch
import torch.distributed as dist

from bergson.config import DistributedConfig
from bergson.distributed import init_dist, launch_distributed_run


def fail_on_rank_one(rank: int, local_rank: int, world_size: int, wait: str):
    init_dist(rank, local_rank, world_size)
    if rank == 1:
        raise ValueError("rank 1 failed")
    if wait == "sleep":
        # Nothing tells rank 0 that rank 1 died, as with NCCL, so only the parent
        # can stop it. Gloo would notice the dropped connection in a collective.
        time.sleep(600)
    else:
        # Rank 1 never sends this, so rank 0 waits here.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dist.broadcast(torch.zeros(1, device=device), src=1)


@pytest.mark.parametrize(
    "backend, wait",
    [
        ("gloo", "collective"),
        ("gloo", "sleep"),
        pytest.param(
            "nccl",
            "collective",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Needs two GPUs"
            ),
        ),
    ],
)
def test_failed_rank_stops_the_launch(monkeypatch, backend: str, wait: str):
    if backend == "gloo":
        # Hide the GPUs so both ranks use Gloo on the CPU.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1,-1")
    # Bound the test if the launch waits for the timeout after all. The spawned
    # ranks read this when they import bergson.distributed.
    monkeypatch.setenv("BERGSON_DIST_TIMEOUT_MIN", "1")

    start = time.monotonic()
    with pytest.raises(RuntimeError, match=r"rank 1\) exited with code 1"):
        launch_distributed_run(
            "test", fail_on_rank_one, [wait], DistributedConfig(nproc_per_node=2)
        )
    assert time.monotonic() - start < 30


def fork_a_sleeper(rank: int, local_rank: int, world_size: int):
    init_dist(rank, local_rank, world_size)
    # The grandchild inherits the child's end of the pipe the parent waits on.
    if os.fork() == 0:
        time.sleep(30)
        os._exit(0)


def test_forked_process_does_not_hold_the_launch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1,-1")
    start = time.monotonic()
    launch_distributed_run(
        "test", fork_a_sleeper, [], DistributedConfig(nproc_per_node=2)
    )
    assert time.monotonic() - start < 20
