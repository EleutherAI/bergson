"""Covariances summed per NUMA node and added at teardown must match one
process's."""

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import bergson.hessians.kfac as kfac
from bergson.gradients import GradientProcessor
from bergson.hessians.kfac import CovarianceCollector
from bergson.hessians.sharded_computation import numa_copies
from tests.ekfac_tests.test_module_ownership import CPU_RANKS, full_matrices
from tests.ekfac_tests.test_utils import rank_batches, run_batches, small_network

# Rank 0 alone on one node, ranks 1 and 2 on the other.
GROUPS = [[0], [1, 2]]


def fit(rank: int, world_size: int, port: int, root: str, numa_bytes: float):
    """Covariances over ``world_size`` gloo ranks on the NUMA nodes in
    ``GROUPS``, or over every rank's batches in one process."""
    if world_size > 1:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        kfac.numa_groups = lambda: GROUPS
        batches = rank_batches(rank)
    else:
        batches = [b for r in range(CPU_RANKS) for b in rank_batches(r)]
    out = os.path.join(root, f"fit{world_size}")
    collector = CovarianceCollector(
        model=small_network(),
        path=out,
        dtype=torch.float64,
        processor=GradientProcessor(include_bias=True),
        numa_bytes=numa_bytes,
    )
    run_batches(collector, batches)
    if world_size > 1:
        if rank == 0:
            torch.save(collector.copies, os.path.join(out, "copies.pt"))
        dist.destroy_process_group()


@pytest.mark.parametrize("numa_bytes", [500, float("inf")])
def test_numa_copies_match_one_process(tmp_path: Path, monkeypatch, numa_bytes: float):
    # gloo can't gather CUDA tensors.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(fit, args=(1, port, str(tmp_path), numa_bytes), nprocs=1)
    mp.spawn(fit, args=(CPU_RANKS, port, str(tmp_path), numa_bytes), nprocs=CPU_RANKS)

    copies = torch.load(tmp_path / f"fit{CPU_RANKS}" / "copies.pt")
    assert copies
    if numa_bytes == float("inf"):
        assert len(copies) == 2 * 3  # A and S of each of the three Linear layers
    for sub in ("activation_sharded", "gradient_sharded"):
        expected = full_matrices(str(tmp_path / "fit1" / sub), 1)
        actual = full_matrices(str(tmp_path / f"fit{CPU_RANKS}" / sub), CPU_RANKS)
        assert expected.keys() == actual.keys()
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])


def test_numa_copies_fit_each_rank_budget():
    sizes = {"a": 10, "b": 20, "c": 30}
    owners = {"a": 0, "b": 1, "c": 2}
    copies = numa_copies(sizes, owners, [[0, 1], [2, 3]], budget=30)
    # Each owner's copy goes to the rank at its position in the other group.
    assert copies == {"a": [0, 2], "b": [1, 3], "c": [0, 2]}
    # c's copy (30) doesn't fit in rank 0's budget of 25.
    assert numa_copies(sizes, owners, [[0, 1], [2, 3]], budget=25) == {
        "a": [0, 2],
        "b": [1, 3],
    }
