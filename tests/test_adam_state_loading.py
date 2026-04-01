import tempfile

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from bergson.gradients import AdafactorNormalizer, AdamNormalizer
from bergson.utils.load_from_optimizer import load_from_optimizer


def _create_model():
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM")
    return AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)


def _create_fake_optimizer_state(model, lr=1e-3):
    """Create a fake optimizer state dict matching the model's parameters."""
    state = {}
    param_groups = [{"lr": lr, "params": []}]

    for idx, (name, param) in enumerate(model.named_parameters()):
        param_groups[0]["params"].append(idx)
        state[idx] = {
            "step": torch.tensor(100),
            "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.rand_like(param) * 0.01,
        }

    return {"state": state, "param_groups": param_groups}


def _train_checkpoint(optim_name: str) -> tuple:
    """Train a tiny model for a few steps and return (checkpoint_path, model)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    dummy_data = Dataset.from_dict({"text": ["hello world"] * 20, "label": [0, 1] * 10})
    dummy_data = dummy_data.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True, max_length=32
        ),
        batched=True,
    )

    tmpdir = tempfile.mkdtemp()
    args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=3,
        save_steps=3,
        per_device_train_batch_size=4,
        optim=optim_name,
        learning_rate=1e-4,
    )
    trainer = Trainer(model=model, args=args, train_dataset=dummy_data)
    trainer.train()

    import os

    ckpt = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint")][0]
    return os.path.join(tmpdir, ckpt), model


# ---------------------------------------------------------------------------
# Unit tests with fake state
# ---------------------------------------------------------------------------


def test_load_from_optimizer_file(tmp_path):
    """Load normalizers from a bare optimizer.pt file."""
    model = _create_model()
    opt_state = _create_fake_optimizer_state(model)

    opt_path = tmp_path / "optimizer.pt"
    torch.save(opt_state, opt_path)

    normalizers = load_from_optimizer(model, str(opt_path))

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

    all_linear = {
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    subset = set(list(all_linear)[:2])

    normalizers = load_from_optimizer(model, str(opt_path), target_modules=subset)
    assert set(normalizers.keys()) == subset


def test_missing_optimizer_file(tmp_path):
    """Error when directory has no optimizer.pt."""
    model = _create_model()

    with pytest.raises(FileNotFoundError):
        load_from_optimizer(model, str(tmp_path))


# ---------------------------------------------------------------------------
# Integration tests with real training checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_adam_checkpoint():
    """Load from a real AdamW training checkpoint and verify values match."""
    ckpt_path, model = _train_checkpoint("adamw_torch")
    opt_state = torch.load(
        f"{ckpt_path}/optimizer.pt", map_location="cpu", weights_only=False
    )

    normalizers = load_from_optimizer(model, ckpt_path)

    assert len(normalizers) > 0
    for norm in normalizers.values():
        assert isinstance(norm, AdamNormalizer)

    # Verify loaded values match the raw checkpoint
    params = list(model.named_parameters())
    for idx, (name, _param) in enumerate(params):
        if not name.endswith(".weight"):
            continue
        module_name = name.removesuffix(".weight")
        if module_name not in normalizers:
            continue

        raw = opt_state["state"][idx]["exp_avg_sq"]
        loaded = normalizers[module_name].weight_avg_sq.cpu()
        torch.testing.assert_close(loaded, raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_adafactor_checkpoint():
    """Load from a real Adafactor training checkpoint and verify values match."""
    ckpt_path, model = _train_checkpoint("adafactor")
    opt_state = torch.load(
        f"{ckpt_path}/optimizer.pt", map_location="cpu", weights_only=False
    )

    normalizers = load_from_optimizer(model, ckpt_path)

    assert len(normalizers) > 0
    for norm in normalizers.values():
        assert isinstance(norm, AdafactorNormalizer)

    # Verify loaded values match the raw checkpoint
    params = list(model.named_parameters())
    for idx, (name, _param) in enumerate(params):
        if not name.endswith(".weight"):
            continue
        module_name = name.removesuffix(".weight")
        if module_name not in normalizers:
            continue

        raw_row = opt_state["state"][idx]["exp_avg_sq_row"]
        raw_col = opt_state["state"][idx]["exp_avg_sq_col"]
        norm = normalizers[module_name]
        assert isinstance(norm, AdafactorNormalizer)
        torch.testing.assert_close(norm.row.cpu(), raw_row)
        torch.testing.assert_close(norm.col.cpu(), raw_col)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_load_8bit_adam_checkpoint():
    """Load from a real 8-bit Adam (bitsandbytes) training checkpoint."""
    ckpt_path, model = _train_checkpoint("adamw_bnb_8bit")
    opt_state = torch.load(
        f"{ckpt_path}/optimizer.pt", map_location="cpu", weights_only=False
    )

    normalizers = load_from_optimizer(model, ckpt_path)

    assert len(normalizers) > 0
    for norm in normalizers.values():
        assert isinstance(norm, AdamNormalizer)

    # Verify loaded values match the raw checkpoint
    params = list(model.named_parameters())
    for idx, (name, _param) in enumerate(params):
        if not name.endswith(".weight"):
            continue
        module_name = name.removesuffix(".weight")
        if module_name not in normalizers:
            continue

        raw = opt_state["state"][idx]["__bnb_optimizer_quant_state__"]["state2"]
        loaded = normalizers[module_name].weight_avg_sq.cpu()
        torch.testing.assert_close(loaded, raw)
