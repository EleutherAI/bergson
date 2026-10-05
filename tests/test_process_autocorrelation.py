"""The eigendecomposition saved across ranks must be that of the summed Gram."""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from bergson.gradients import GradientProcessor
from bergson.process_autocorrelation import process_autocorrelation_matrices

SIZES = {"a": 5, "b": 3, "c": 4}
NUM_ROWS = 12


def rank_rows(rank: int, name: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(100 * rank + ord(name))
    return torch.randn(NUM_ROWS // 2, SIZES[name], generator=gen, dtype=torch.float64)


def full_gram(name: str, world_size: int) -> torch.Tensor:
    rows = torch.cat([rank_rows(r, name) for r in range(world_size)])
    return rows.mT @ rows / NUM_ROWS


def run(rank: int, world_size: int, port: int, owned: bool, out: dict):
    dist.init_process_group(
        "gloo", init_method=f"tcp://localhost:{port}", rank=rank, world_size=world_size
    )
    if owned:
        # Each rank holds the full Gram of the modules it owns.
        names = [n for i, n in enumerate(SIZES) if i % world_size == rank]
        hessians = {n: full_gram(n, world_size) * NUM_ROWS for n in names}
    else:
        # Each rank holds a partial Gram over its own rows for every module.
        hessians = {n: rank_rows(rank, n).mT @ rank_rows(rank, n) for n in SIZES}

    processor = GradientProcessor()
    process_autocorrelation_matrices(processor, hessians, NUM_ROWS, SIZES, rank)
    if rank == 0:
        out.update(
            hessians=dict(processor.hessians), eigen=dict(processor.hessians_eigen)
        )
    dist.destroy_process_group()


@pytest.mark.parametrize("owned", [False, True])
def test_eigen_matches_summed_gram(owned: bool):
    world_size = 2
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    out = mp.Manager().dict()
    mp.spawn(run, args=(world_size, port, owned, out), nprocs=world_size)

    for name in SIZES:
        expected = full_gram(name, world_size)
        torch.testing.assert_close(out["hessians"][name], expected)
        eigvals, eigvecs = out["eigen"][name]
        torch.testing.assert_close(eigvals, torch.linalg.eigvalsh(expected))
        torch.testing.assert_close(eigvecs @ torch.diag(eigvals) @ eigvecs.mT, expected)
