import pytest
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D as HFConv1D

from bergson.hessians.kfac import CovarianceCollector
from bergson.utils.utils import get_device

IN_DIM = 4
OUT_DIM = 6


class TinyConv1DModel(nn.Module):
    """Minimal model mixing HFConv1D and nn.Linear with the same in/out dims."""

    def __init__(self):
        super().__init__()
        self.conv = HFConv1D(OUT_DIM, IN_DIM)  # HFConv1D(nf=out, nx=in)
        self.linear = nn.Linear(OUT_DIM, IN_DIM)

    def forward(self, x):
        return self.linear(self.conv(x))


def test_covariance_collector_on_conv1d(tmp_path):
    """End-to-end hook pass over a Conv1D layer."""
    # CovarianceCollector accumulates on get_device(rank); keep everything there
    device = get_device(0)
    model = TinyConv1DModel().to(device)
    collector = CovarianceCollector(
        model=model, dtype=torch.float32, path=str(tmp_path)
    )

    # A_cov is [in, in], S_cov is [out, out] for the Conv1D layer
    assert collector.A_cov_dict["conv"].shape == (IN_DIM, IN_DIM)
    assert collector.S_cov_dict["conv"].shape == (OUT_DIM, OUT_DIM)

    n, s = 2, 3
    x = torch.randn(n, s, IN_DIM, device=device)
    mask = torch.ones(n, s, dtype=torch.bool, device=device)

    with collector.with_batch(mask):
        out = model(x)
        out.sum().backward()

    # Forward hook accumulated A^T A over valid positions
    a = x[mask]
    torch.testing.assert_close(collector.A_cov_dict["conv"], a.mT @ a)

    # Backward hook accumulated G^T G with the right (out) dimension
    assert collector.S_cov_dict["conv"].shape == (OUT_DIM, OUT_DIM)
    assert collector.S_cov_dict["conv"].abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
