"""The autocorrelation collectors' running Grams survive a save and reload."""

import pytest
import torch
import torch.nn as nn
from datasets import Dataset

from bergson.gradients import GradientProcessor
from bergson.hessians.autocorrelation import (
    AutocorrelationCollector,
    JointAutocorrelationCollector,
)
from bergson.utils.utils import get_device

JOINT_DIM = 8


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 6, bias=False)

    def forward(self, x):
        return self.lin(x)


def _collector(model, tmp_path, joint: bool):
    data = Dataset.from_dict({"input_ids": [[0]] * 4})
    if joint:
        return JointAutocorrelationCollector(
            model=model,
            data=data,
            path=str(tmp_path),
            processor=GradientProcessor(
                projection_dim=JOINT_DIM, projection_target="global"
            ),
        )
    return AutocorrelationCollector(
        model=model, data=data, path=str(tmp_path), processor=GradientProcessor()
    )


def _run_batch(model, collector, x):
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    with collector.with_batch(mask):
        model(x).sum().backward()
    collector.process_batch(list(range(len(x))))


def _saved_state(collector, tmp_path) -> dict:
    """The collector's state as a checkpointed run reads it back."""
    file = tmp_path / "state.pt"
    torch.save(collector.state_dict(), file)
    return torch.load(file, map_location="cpu", weights_only=True)


@pytest.mark.parametrize("joint", [False, True])
def test_resumed_grams_match_uninterrupted(tmp_path, joint):
    device = get_device(0)
    model = TinyModel().to(device)
    batches = [torch.randn(2, 3, 4, device=device) for _ in range(2)]

    full = _collector(model, tmp_path / "full", joint)
    for x in batches:
        _run_batch(model, full, x)

    first = _collector(model, tmp_path / "first", joint)
    _run_batch(model, first, batches[0])
    state = _saved_state(first, tmp_path)

    resumed = _collector(model, tmp_path / "resumed", joint)
    resumed.load_state_dict(state)
    _run_batch(model, resumed, batches[1])

    assert set(full.processor.hessians) == {"joint" if joint else "lin"}
    torch.testing.assert_close(
        resumed.processor.hessians, full.processor.hessians, rtol=0, atol=0
    )


@pytest.mark.parametrize("joint", [False, True])
def test_load_rejects_other_modules(tmp_path, joint):
    model = TinyModel().to(get_device(0))
    collector = _collector(model, tmp_path, joint)
    _run_batch(model, collector, torch.randn(2, 3, 4, device=get_device(0)))
    (gram,) = _saved_state(collector, tmp_path)["hessians"].values()

    with pytest.raises(ValueError, match="hessians"):
        collector.load_state_dict({"hessians": {"other": gram}})


@pytest.mark.parametrize("joint", [False, True])
def test_load_rejects_other_shapes(tmp_path, joint):
    model = TinyModel().to(get_device(0))
    collector = _collector(model, tmp_path, joint)
    _run_batch(model, collector, torch.randn(2, 3, 4, device=get_device(0)))
    state = _saved_state(collector, tmp_path)
    state["hessians"] = {name: g[:1] for name, g in state["hessians"].items()}

    with pytest.raises(ValueError, match="shape"):
        collector.load_state_dict(state)
