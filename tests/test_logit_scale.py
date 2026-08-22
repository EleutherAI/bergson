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


class _NotALanguageModel(torch.nn.Module):
    def get_output_embeddings(self):
        return None


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(3, 4)


def test_scale_one_is_a_no_op(x):
    """The default must not even register a hook."""
    model = _TinyCausalLM()
    before = model(x).clone()
    apply_logit_scale(model, 1.0)
    torch.testing.assert_close(model(x), before)
    assert not model.head._forward_hooks


@pytest.mark.parametrize("scale", [0.25, 0.5, 2.0])
def test_logits_are_scaled(x, scale):
    model = _TinyCausalLM()
    before = model(x).clone()
    apply_logit_scale(model, scale)
    torch.testing.assert_close(model(x), before * scale)


def test_scaling_flattens_the_softmax(x):
    """The point of the axis: a scale below 1 lowers the peak probability."""
    model = _TinyCausalLM()
    sharp = torch.softmax(model(x), dim=-1).max(dim=-1).values
    apply_logit_scale(model, 0.25)
    flat = torch.softmax(model(x), dim=-1).max(dim=-1).values
    assert (flat < sharp).all()


def test_scale_applies_to_every_call(x):
    """Not just the first forward -- the bank evaluates many times."""
    model = _TinyCausalLM()
    before = model(x).clone()
    apply_logit_scale(model, 0.5)
    for _ in range(3):
        torch.testing.assert_close(model(x), before * 0.5)


def test_gradients_scale_with_logits(x):
    """Training must see the scale, not only evaluation."""
    plain = _TinyCausalLM()
    scaled = _TinyCausalLM()
    scaled.load_state_dict(plain.state_dict())
    apply_logit_scale(scaled, 0.5)

    plain(x).sum().backward()
    scaled(x).sum().backward()

    torch.testing.assert_close(
        scaled.head.linear.weight.grad, plain.head.linear.weight.grad * 0.5
    )


def test_model_without_output_embeddings_is_rejected():
    with pytest.raises(ValueError, match="no output embeddings"):
        apply_logit_scale(_NotALanguageModel(), 0.5)


def test_rejection_does_not_fire_at_scale_one():
    """A non-LM at the default scale is fine -- nothing is hooked."""
    model = _NotALanguageModel()
    assert apply_logit_scale(model, 1.0) is model


def test_config_exposes_logit_scale():
    from bergson.config.config import ModelConfig

    assert ModelConfig(run_path="/tmp/x", model="gpt2").logit_scale == 1.0
    scaled = ModelConfig(run_path="/tmp/x", model="gpt2", logit_scale=0.25)
    assert scaled.logit_scale == 0.25
