"""CovarianceCollector's running sums survive a save and reload."""

import pytest
import torch
import torch.nn as nn

from bergson.gradients import GradientProcessor
from bergson.hessians.kfac import CovarianceCollector
from bergson.utils.utils import get_device


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 6, bias=False)

    def forward(self, x):
        return self.lin(x)


def _collector(model, tmp_path):
    return CovarianceCollector(
        model=model,
        dtype=torch.float64,
        path=str(tmp_path),
        processor=GradientProcessor(),
    )


def _run_batch(model, collector, x):
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    with collector.with_batch(mask):
        model(x).sum().backward()


def test_resumed_sums_match_uninterrupted(tmp_path):
    device = get_device(0)
    model = TinyModel().to(device)
    batches = [torch.randn(2, 3, 4, device=device) for _ in range(2)]

    full = _collector(model, tmp_path / "full")
    for x in batches:
        _run_batch(model, full, x)

    first = _collector(model, tmp_path / "first")
    _run_batch(model, first, batches[0])
    state = first.state_dict()

    resumed = _collector(model, tmp_path / "resumed")
    resumed.load_state_dict(state)
    _run_batch(model, resumed, batches[1])

    torch.testing.assert_close(resumed.A_cov_dict, full.A_cov_dict, rtol=0, atol=0)
    torch.testing.assert_close(resumed.S_cov_dict, full.S_cov_dict, rtol=0, atol=0)
    assert resumed.num_positions == full.num_positions


def test_load_rejects_other_modules(tmp_path):
    model = TinyModel().to(get_device(0))
    collector = _collector(model, tmp_path)
    state = collector.state_dict()
    state["A_cov"] = {"other": state["A_cov"]["lin"]}

    with pytest.raises(ValueError, match="A_cov"):
        collector.load_state_dict(state)


def test_load_rejects_other_shapes(tmp_path):
    model = TinyModel().to(get_device(0))
    collector = _collector(model, tmp_path)
    state = collector.state_dict()
    # One row of the sum broadcasts into the full matrix, so check shapes.
    state["S_cov"] = {"lin": state["S_cov"]["lin"][:1].clone()}

    with pytest.raises(ValueError, match="shape"):
        collector.load_state_dict(state)
