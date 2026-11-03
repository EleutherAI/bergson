"""Modules called on the same input, as LoRA's q, k and v adapters are, must get
the activation covariance they would get alone."""

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from bergson.gradients import GradientProcessor
from bergson.hessians.kfac import CovarianceCollector
from tests.ekfac_tests.test_module_ownership import CPU_RANKS, full_matrices
from tests.ekfac_tests.test_utils import WIDTHS, rank_batches, run_batches


class SharedInput(nn.Module):
    """Three projections of one input, and a fourth of a copy of it, which
    sums its own covariance."""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, 3)
        self.k = nn.Linear(dim, 2)
        self.v = nn.Linear(dim, 4)
        self.copy = nn.Linear(dim, 2)

    def forward(self, x):
        out = torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)
        return torch.cat([out, self.copy(x.clone())], dim=-1)


def network() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(WIDTHS[0], 6), nn.Tanh(), SharedInput(6))


def fit(rank: int, world_size: int, port: int, root: str):
    """Covariances over ``world_size`` gloo ranks, or over every rank's batches
    in one process when ``world_size`` is 1."""
    if world_size > 1:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        batches = rank_batches(rank)
    else:
        batches = [b for r in range(CPU_RANKS) for b in rank_batches(r)]
    out = os.path.join(root, f"fit{world_size}")
    collector = CovarianceCollector(
        model=network(),
        path=out,
        dtype=torch.float64,
        processor=GradientProcessor(include_bias=True),
    )
    run_batches(collector, batches)
    if rank == 0:
        torch.save(
            (collector.aliases, collector.num_positions),
            os.path.join(out, "aliases.pt"),
        )
        torch.save(collector.owners, os.path.join(out, "owners.pt"))
    if world_size > 1:
        dist.destroy_process_group()


def test_shared_inputs_match_their_own_covariances(tmp_path: Path, monkeypatch):
    # gloo can't send CUDA tensors.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(fit, args=(1, port, str(tmp_path)), nprocs=1)
    mp.spawn(fit, args=(CPU_RANKS, port, str(tmp_path)), nprocs=CPU_RANKS)

    aliases, num_positions = torch.load(tmp_path / "fit1" / "aliases.pt")
    assert aliases == {"2.k": "2.q", "2.v": "2.q"}
    assert torch.load(tmp_path / f"fit{CPU_RANKS}" / "aliases.pt") == (
        aliases,
        num_positions,
    )

    # The shared input, with the bias's ones column, summed over every batch.
    model = network()
    rows = []
    for x, mask in (b for r in range(CPU_RANKS) for b in rank_batches(r)):
        with torch.no_grad():
            hidden = model[1](model[0](x))[mask].double()
        rows.append(torch.cat([hidden, hidden.new_ones(len(hidden), 1)], dim=1))
    stacked = torch.cat(rows)
    expected = stacked.mT @ stacked

    single = full_matrices(str(tmp_path / "fit1" / "activation_sharded"), 1)
    distributed = full_matrices(
        str(tmp_path / f"fit{CPU_RANKS}" / "activation_sharded"), CPU_RANKS
    )
    for name in ("2.q", "2.k", "2.v"):
        torch.testing.assert_close(single[name], expected)
        torch.testing.assert_close(distributed[name], expected)
        assert num_positions[name] == len(stacked)
    torch.testing.assert_close(single["2.copy"], expected)
    torch.testing.assert_close(distributed["2.copy"], expected)

    # Some alias belongs to another rank than its earlier module, so its
    # covariance went between ranks.
    owners = torch.load(tmp_path / f"fit{CPU_RANKS}" / "owners.pt")
    assert any(owners[name] != owners[earlier] for name, earlier in aliases.items())
