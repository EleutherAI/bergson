"""hessian_dtype must reach BOTH Hessian collectors.

The covariance collectors always took a ``dtype``, but the EK-FAC
eigenvalue-correction (lambda) pass silently ignored ``hessian_dtype`` and
accumulated in the activations' dtype. These tests pin the dtype through both
collectors, including the fp64 setting used to match kronfluence's
eigendecomposition precision.
"""

import os

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from bergson.gradients import GradientProcessor
from bergson.hessians.eigenvectors import LambdaCollector
from bergson.hessians.kfac import CovarianceCollector
from bergson.utils.utils import convert_precision_to_torch, get_device

IN_DIM = 4
OUT_DIM = 6


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(IN_DIM, OUT_DIM, bias=False)

    def forward(self, x):
        return self.lin(x)


def _run_batch(model, collector, device):
    n, s = 2, 3
    x = torch.randn(n, s, IN_DIM, device=device)
    mask = torch.ones(n, s, dtype=torch.bool, device=device)
    with collector.with_batch(mask):
        model(x).sum().backward()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16])
def test_covariance_collector_accumulates_in_dtype(tmp_path, dtype):
    device = get_device(0)
    model = TinyModel().to(device)
    collector = CovarianceCollector(
        model=model,
        dtype=dtype,
        path=str(tmp_path),
        processor=GradientProcessor(),
    )
    _run_batch(model, collector, device)
    assert collector.A_cov_dict["lin"].dtype == dtype
    assert collector.S_cov_dict["lin"].dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_lambda_collector_accumulates_in_dtype(tmp_path, dtype):
    device = get_device(0)
    model = TinyModel().to(device)

    eigen_a = {"lin": torch.eye(IN_DIM, dtype=torch.float32)}
    eigen_g = {"lin": torch.eye(OUT_DIM, dtype=torch.float32)}
    os.makedirs(tmp_path / "eigen_activation_sharded")
    os.makedirs(tmp_path / "eigen_gradient_sharded")
    save_file(eigen_a, str(tmp_path / "eigen_activation_sharded/shard_0.safetensors"))
    save_file(eigen_g, str(tmp_path / "eigen_gradient_sharded/shard_0.safetensors"))

    collector = LambdaCollector(
        model=model,
        path=str(tmp_path),
        processor=GradientProcessor(),
        dtype=dtype,
    )
    _run_batch(model, collector, device)
    assert collector.eigenvalue_corrections["lin"].dtype == dtype

    collector.teardown()
    from safetensors.torch import load_file

    shard = load_file(
        str(tmp_path / "eigenvalue_correction_sharded/shard_0.safetensors")
    )
    assert shard["lin"].dtype == dtype


def test_fp64_precision_string_converts():
    assert convert_precision_to_torch("fp64") is torch.float64
