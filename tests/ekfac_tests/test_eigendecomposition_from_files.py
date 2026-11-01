"""Eigendecomposing saved covariances, each owned by one rank, must give the
eigenvectors and eigenvalues of the joined covariances in every rank's rows."""

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file

from bergson.gradients import GradientProcessor
from bergson.hessians.eigenvectors import compute_eigendecomposition
from bergson.hessians.kfac import CovarianceCollector
from tests.ekfac_tests.test_utils import rank_batches, run_batches, small_network

TOTAL_PROCESSED = 7
SUBDIRS = ("activation_sharded", "gradient_sharded")


def fit(rank: int, world_size: int, port: int, root: str):
    if world_size > 1:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
    run_batches(
        CovarianceCollector(
            model=small_network(),
            path=root,
            dtype=torch.float64,
            processor=GradientProcessor(include_bias=True),
        ),
        rank_batches(rank),
    )
    for sub in SUBDIRS:
        values = compute_eigendecomposition(os.path.join(root, sub), TOTAL_PROCESSED)
        save_file(values, os.path.join(root, f"values_{sub}_{rank}.safetensors"))
    if world_size > 1:
        dist.destroy_process_group()


def joined(directory: str, world_size: int) -> dict[str, torch.Tensor]:
    shards = [
        load_file(os.path.join(directory, f"shard_{r}.safetensors"))
        for r in range(world_size)
    ]
    return {key: torch.cat([s[key] for s in shards]) for key in shards[0]}


@pytest.mark.parametrize("world_size", [1, 3])
def test_eigendecomposition_from_files(tmp_path: Path, monkeypatch, world_size: int):
    # Collectors place their factors on the GPU when one is visible, and gloo
    # can't scatter CUDA tensors.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(fit, args=(world_size, port, str(tmp_path)), nprocs=world_size)

    for sub in SUBDIRS:
        covariances = joined(str(tmp_path / sub), world_size)
        vectors = joined(str(tmp_path / f"eigen_{sub}"), world_size)
        values = {
            key: torch.cat(
                [
                    load_file(tmp_path / f"values_{sub}_{r}.safetensors")[key]
                    for r in range(world_size)
                ]
            )
            for key in covariances
        }
        assert vectors.keys() == covariances.keys()
        for key, covariance in covariances.items():
            expected_values, expected_vectors = torch.linalg.eigh(
                (covariance + covariance.T) / (2 * TOTAL_PROCESSED)
            )
            torch.testing.assert_close(values[key], expected_values)
            # Eigenvectors are sign-ambiguous per column.
            torch.testing.assert_close(vectors[key].abs(), expected_vectors.abs())
