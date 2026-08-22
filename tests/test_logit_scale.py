import pytest
import torch

from bergson.utils.worker_utils import apply_logit_scale


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 6, bias=False)

    def forward(self, x):
        return self.linear(x)


class _TinyCausalLM(torch.nn.Module):
    """Minimal stand-in exposing the one method apply_logit_scale relies on."""

    def __init__(self):
        super().__init__()
        self.head = _Head()

    def get_output_embeddings(self):
        return self.head

    def forward(self, x):
        return self.head(x)


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(3, 4)


@pytest.mark.parametrize("scale", [0.25, 0.5, 2.0])
def test_logits_are_scaled(x, scale):
    model = _TinyCausalLM()
    before = model(x).clone()
    apply_logit_scale(model, scale)
    torch.testing.assert_close(model(x), before * scale)
