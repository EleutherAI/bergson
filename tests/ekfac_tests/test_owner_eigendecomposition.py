"""Eigendecomposing each rank's owned covariances from memory must give the
files and eigenvalue corrections the file-based path gives."""

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file

from bergson.gradients import GradientProcessor
from bergson.hessians.eigenvectors import (
    LambdaCollector,
    compute_eigendecomposition,
    eigendecompose_owned,
)
from bergson.hessians.kfac import CovarianceCollector
from tests.ekfac_tests.test_module_ownership import rank_batches, small_network

TOTAL_PROCESSED = 7


def run_batches(collector, batches):
    network = collector.model
    for x, mask in batches:
        with collector.with_batch(mask):
            (network(x) ** 2 * mask[..., None]).sum().backward()
        network.zero_grad()
    collector.teardown()


def fit(rank: int, world_size: int, port: int, root: str):
    if world_size > 1:
        dist.init_process_group(
            "gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
    batches = rank_batches(rank)
    root_path = Path(root)
    processor = GradientProcessor(include_bias=True)

    covariances = CovarianceCollector(
        model=small_network(),
        path=str(root_path / "files"),
        dtype=torch.float64,
        processor=processor,
    )
    run_batches(covariances, batches)

    eigenvectors, eigenvalues = {}, {}
    for sub, owned, shapes in (
        ("activation_sharded", covariances.A_cov_dict, covariances.A_shapes),
        ("gradient_sharded", covariances.S_cov_dict, covariances.S_shapes),
    ):
        eigenvectors[sub], eigenvalues[sub] = eigendecompose_owned(
            owned,
            shapes,
            covariances.owners,
            TOTAL_PROCESSED,
            str(root_path / "owned" / f"eigen_{sub}"),
            torch.float64,
        )
        expected = compute_eigendecomposition(
            str(root_path / "files" / sub), TOTAL_PROCESSED
        )
        assert eigenvalues[sub].keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(eigenvalues[sub][key], expected[key])

    for name, extra in (
        (
            "owned",
            {
                "eigenvectors": tuple(eigenvectors.values()),
                "eigen_path": str(root_path / "missing"),
            },
        ),
        ("files", {}),
    ):
        run_batches(
            LambdaCollector(
                model=small_network(),
                path=str(root_path / name),
                dtype=torch.float64,
                processor=processor,
                **extra,
            ),
            batches,
        )

    if world_size > 1:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 3])
def test_owned_eigendecomposition_matches_files(
    tmp_path: Path, monkeypatch, world_size: int
):
    # Collectors place their factors on the GPU when one is visible, and gloo
    # can't scatter CUDA tensors.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(fit, args=(world_size, port, str(tmp_path)), nprocs=world_size)

    for sub in (
        "eigen_activation_sharded",
        "eigen_gradient_sharded",
        "eigenvalue_correction_sharded",
    ):
        for rank in range(world_size):
            shard = f"shard_{rank}.safetensors"
            expected = load_file(os.path.join(tmp_path, "files", sub, shard))
            actual = load_file(os.path.join(tmp_path, "owned", sub, shard))
            assert expected.keys() == actual.keys()
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])
