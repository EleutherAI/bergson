"""Regression tests for K-FAC + include_bias shape compatibility.

See https://github.com/EleutherAI/bergson/issues/277: with include_bias=True,
`bergson build` stores per-layer gradients of shape [O, I+1] (the bias gradient
is an extra "activation" column), but the K-FAC covariance collectors used to
compute A^T A on the raw activation, giving an [I, I] activation covariance
that `apply_hessian`'s `.view(-1, O, I)` could not reconcile with the stored
flat size N*O*(I+1).
"""

import math
import os

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from bergson.gradients import GradientProcessor
from bergson.hessians.eigenvectors import LambdaCollector
from bergson.hessians.kfac import CovarianceCollector
from bergson.hessians.sharded_computation import shard_bounds
from bergson.utils.utils import get_device

IN_DIM = 4
OUT_DIM = 6


class TinyBiasModel(nn.Module):
    """Minimal model mixing a biased and an unbiased linear layer."""

    def __init__(self):
        super().__init__()
        self.biased = nn.Linear(IN_DIM, OUT_DIM, bias=True)
        self.unbiased = nn.Linear(OUT_DIM, IN_DIM, bias=False)

    def forward(self, x):
        return self.unbiased(self.biased(x))


def test_covariance_collector_include_bias(tmp_path):
    """A_cov must use the augmented activation [a; 1] when bias is collected."""
    device = get_device(0)
    model = TinyBiasModel().to(device)
    collector = CovarianceCollector(
        model=model,
        dtype=torch.float32,
        path=str(tmp_path),
        processor=GradientProcessor(include_bias=True),
    )

    # Augmented [I+1, I+1] for the biased layer, raw [O, O] for the unbiased one
    assert collector.A_cov_dict["biased"].shape == (IN_DIM + 1, IN_DIM + 1)
    assert collector.S_cov_dict["biased"].shape == (OUT_DIM, OUT_DIM)
    assert collector.A_cov_dict["unbiased"].shape == (OUT_DIM, OUT_DIM)

    n, s = 2, 3
    x = torch.randn(n, s, IN_DIM, device=device)
    mask = torch.ones(n, s, dtype=torch.bool, device=device)

    with collector.with_batch(mask):
        out = model(x)
        out.sum().backward()

    # Forward hook accumulated A^T A over the augmented activation
    a = x[mask]
    a_aug = torch.cat([a, a.new_ones(a.shape[0], 1)], dim=1)
    torch.testing.assert_close(collector.A_cov_dict["biased"], a_aug.mT @ a_aug)
    # Bias-bias corner counts the number of valid positions
    torch.testing.assert_close(
        collector.A_cov_dict["biased"][-1, -1],
        torch.tensor(float(n * s), device=device),
    )

    # The covariance dims must factorize the stored gradient size [O, I+1],
    # so apply_hessian's view(-1, O, I+1) succeeds (issue #277).
    grad_shape = collector.shapes()["biased"]
    assert collector.S_cov_dict["biased"].shape[1] * collector.A_cov_dict[
        "biased"
    ].shape[1] == math.prod(grad_shape)


def test_lambda_collector_include_bias(tmp_path):
    """LambdaCollector must transform the augmented activation [a; 1]."""
    device = get_device(0)
    model = TinyBiasModel().to(device)

    # Save identity eigenvectors with the augmented activation dimension
    eigen_a = {
        "biased": torch.eye(IN_DIM + 1, dtype=torch.float32),
        "unbiased": torch.eye(OUT_DIM, dtype=torch.float32),
    }
    eigen_g = {
        "biased": torch.eye(OUT_DIM, dtype=torch.float32),
        "unbiased": torch.eye(IN_DIM, dtype=torch.float32),
    }
    os.makedirs(tmp_path / "eigen_activation_sharded")
    os.makedirs(tmp_path / "eigen_gradient_sharded")
    save_file(eigen_a, str(tmp_path / "eigen_activation_sharded/shard_0.safetensors"))
    save_file(eigen_g, str(tmp_path / "eigen_gradient_sharded/shard_0.safetensors"))

    collector = LambdaCollector(
        model=model,
        path=str(tmp_path),
        processor=GradientProcessor(include_bias=True),
    )

    n, s = 2, 3
    x = torch.randn(n, s, IN_DIM, device=device)
    mask = torch.ones(n, s, dtype=torch.bool, device=device)

    with collector.with_batch(mask):
        out = model(x)
        out.sum().backward()

    # Eigenvalue corrections match the stored [O, I+1] gradient layout
    assert collector.eigenvalue_corrections["biased"].shape == (OUT_DIM, IN_DIM + 1)
    assert collector.eigenvalue_corrections["unbiased"].shape == (IN_DIM, OUT_DIM)


@pytest.mark.parametrize("dim, world_size", [(1, 1), (7, 3), (64, 4)])
def test_shard_bounds_partitions_dim(dim, world_size):
    """Shards tile [0, dim) contiguously; rank 0 takes the remainder rows."""
    base, remainder = divmod(dim, world_size)
    prev_end = 0
    for rank in range(world_size):
        start, end = shard_bounds(dim, rank, world_size)
        assert start == prev_end
        assert end - start == base + (remainder if rank == 0 else 0)
        prev_end = end
    assert prev_end == dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
