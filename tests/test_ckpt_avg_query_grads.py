import os

import pytest

import bergson.magic.cli as cli


class FakeState:
    def __init__(self):
        self.step = "final"
        self.log = []

    def load(self, path):
        self.step = os.path.basename(path)

    def to(self, device):
        return "snapshot"

    def copy_(self, other):
        self.log.append(other)
        self.step = "final"


def test_averages_last_k_and_restores_state(tmp_path, monkeypatch):
    for n in (2, 10, 40, 125):
        os.makedirs(tmp_path / f"step_{n}.ckpt")
    state = FakeState()
    compute = cli.compute_query_gradients
    steps = []

    def fake(fwd_state, *args):
        steps.append(fwd_state.step)
        return {"w": len(steps)}, float(len(steps))

    monkeypatch.setattr(cli, "compute_query_gradients", fake)
    grads, loss = compute(state, None, None, ckpts_path=str(tmp_path), ckpt_avg_k=2)

    assert steps == ["step_40.ckpt", "step_125.ckpt"]
    assert grads == {"w": 1.5} and loss == 1.5
    assert state.step == "final" and state.log == ["snapshot"]


def test_too_few_checkpoints_raises(tmp_path):
    os.makedirs(tmp_path / "step_1.ckpt")
    with pytest.raises(ValueError, match="only 1 checkpoints"):
        cli.compute_query_gradients(
            FakeState(), None, None, ckpts_path=str(tmp_path), ckpt_avg_k=2
        )
