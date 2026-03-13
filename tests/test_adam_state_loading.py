import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from bergson.gradients import AdamNormalizer
from bergson.utils.load_adam_state import load_from_optimizer


def _create_model():
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    return AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)


def _create_fake_optimizer_state(model, lr=1e-3):
    """Create a fake optimizer state dict matching the model's parameters."""
    state = {}
    param_groups = [{"lr": lr, "params": []}]

    for idx, (name, param) in enumerate(model.named_parameters()):
        param_groups[0]["params"].append(idx)
        # Only create exp_avg_sq for 2D (weight) params
        if param.ndim == 2:
            state[idx] = {
                "step": torch.tensor(100),
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.rand_like(param).abs() * 0.01,
            }
        else:
            state[idx] = {
                "step": torch.tensor(100),
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.rand_like(param).abs() * 0.01,
            }

    return {"state": state, "param_groups": param_groups}


def test_load_from_optimizer_file(tmp_path):
    """Load normalizers from a bare optimizer.pt file."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path))

    # Should have normalizers for all linear layers with 2D weights
    assert len(normalizers) > 0
    for name, norm in normalizers.items():
        assert isinstance(norm, AdamNormalizer)
        assert norm.weight_avg_sq.ndim == 2


def test_load_from_checkpoint_dir(tmp_path):
    """Load normalizers from a checkpoint directory containing optimizer.pt."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    checkpoint_dir = tmp_path / "checkpoint-100"
    checkpoint_dir.mkdir()
    torch.save(opt_state, checkpoint_dir / "optimizer.pt")

    normalizers = load_from_optimizer(model, str(checkpoint_dir))
    assert len(normalizers) > 0


def test_target_modules_filter(tmp_path):
    """Only layers in target_modules are loaded."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    # Get all linear layer names
    all_linear = {
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    # Pick a subset
    subset = set(list(all_linear)[:2])

    normalizers = load_from_optimizer(model, str(opt_path), target_modules=subset)
    assert set(normalizers.keys()) == subset


def test_missing_optimizer_file(tmp_path):
    """Error when directory has no optimizer.pt."""
    model = _create_model()

    with pytest.raises(FileNotFoundError):
        load_from_optimizer(model, str(tmp_path))
