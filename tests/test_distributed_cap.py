"""Tests for capping world_size to dataset size in build/score paths.

When a dataset has fewer docs than the configured world_size, the build
path used to pad the dataset and truncate the resulting index. The new
behavior reduces world_size to ``len(dataset)`` instead — no padding,
no truncation, no duplicate gradient computation across ranks.
"""

import os
import socket
import time

import torch.multiprocessing as mp

from bergson.config import DistributedConfig
from bergson.distributed import cap_world_size_to_dataset, parent_barrier


def test_cap_to_intermediate_dataset_size():
    """Dataset between 1 and original world_size → single node, fitting nproc."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    capped = cap_world_size_to_dataset(cfg, dataset_size=3)
    assert capped.world_size == 3
    assert capped.nnode == 1
    assert capped.nproc_per_node == 3


def test_caps_to_one_node_when_dataset_exceeds_nproc_per_node():
    """Dataset between nproc_per_node and world_size → 1 node, full nproc_per_node."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    # 5 docs: too many for 1 GPU, too few for the full 16-GPU world.
    capped = cap_world_size_to_dataset(cfg, dataset_size=5)
    assert capped.world_size == 4
    assert capped.nnode == 1
    assert capped.nproc_per_node == 4


def test_unchanged_when_dataset_exactly_world_size():
    """Boundary: dataset == world_size leaves cfg unchanged (cap on strict-less)."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    capped = cap_world_size_to_dataset(cfg, dataset_size=16)
    assert capped.world_size == 16
    assert capped.nnode == 4
    assert capped.nproc_per_node == 4


def test_does_not_mutate_input():
    """Capping returns a new cfg; the original is untouched."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    cap_world_size_to_dataset(cfg, dataset_size=1)
    assert cfg.nnode == 4
    assert cfg.nproc_per_node == 4
    assert cfg.world_size == 16


def test_zero_dataset_size_caps_to_one_worker():
    """Dataset with zero docs caps to a single worker (downstream raises)."""
    cfg = DistributedConfig(nnode=4, nproc_per_node=4)
    capped = cap_world_size_to_dataset(cfg, dataset_size=0)
    assert capped.world_size == 1
    assert capped.nnode == 1
    assert capped.nproc_per_node == 1


def _capped_build_simulator(
    rank: int, nnode: int, port: int, partial_path, final_path
) -> None:
    """Mimic build()'s capped path: rank-0 worker writes, rank-0 renames
    partial → final, then parent_barrier across all srun parents."""
    os.environ["SLURM_NODEID"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    orig = DistributedConfig(nnode=nnode, nproc_per_node=4)
    capped = cap_world_size_to_dataset(orig, dataset_size=1)

    if capped.world_size <= 1 and capped._node_rank == 0:
        time.sleep(0.3)
        partial_path.mkdir(parents=True, exist_ok=True)
        (partial_path / "config.yaml").write_text("data")

    if capped.rank == 0:
        time.sleep(0.3)
        partial_path.rename(final_path)

    if capped.world_size < orig.world_size:
        parent_barrier(orig)

    assert (
        final_path / "config.yaml"
    ).exists(), f"rank {rank}: file not visible after parent_barrier"


def test_parent_barrier_serializes_capped_build_writes(tmp_path):
    """After build()'s capped rank-0 shutil.move, parent_barrier must keep
    non-rank-0 srun parents from sprinting ahead. Spawns 2 simulated
    srun parents and verifies all see the final path post-barrier.

    Without the barrier (or with the barrier placed before the rename),
    the non-rank-0 process would assert before rank-0 finishes the
    rename, since rank-0 sleeps to mimic a slow NFS move.
    """
    nnode = 2
    procs: list[mp.Process] = []

    # parent_barrier rendezvous on port+1, and anything on the machine can
    # grab it between the bind probe below and the TCPStore listen. Retry
    # with a fresh port on failure; a genuine barrier regression is
    # deterministic and still fails every attempt.
    for attempt in range(3):
        partial_path = tmp_path / f"out{attempt}.part"
        final_path = tmp_path / f"out{attempt}"

        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        procs = [
            mp.Process(
                target=_capped_build_simulator,
                args=(r, nnode, port, partial_path, final_path),
            )
            for r in range(nnode)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
        for p in procs:
            # If one rank crashed, its peer waits in parent_barrier forever;
            # reap it or it outlives the test and pins the pytest process.
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        if all(p.exitcode == 0 for p in procs) or attempt == 2:
            break

    for p in procs:
        assert p.exitcode == 0, f"proc exited with {p.exitcode}"
