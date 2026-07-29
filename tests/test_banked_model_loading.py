"""Loading banked ``save_models`` checkpoints in ``evaluate_retrained``.

``save_models`` banks whatever ``model.save_pretrained`` produces:
a full HF checkpoint for plain fine-tunes, or an adapter-only directory when
``peft_init_kwargs`` is set. ``AutoModelForCausalLM.from_pretrained`` pointed
straight at an adapter-only directory silently returns a randomly-initialised
base model with the LoRA weights ignored, so ``_load_banked_model`` must apply
the adapter explicitly on top of the base model.
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from bergson.magic.config import MagicConfig
from bergson.validate import _load_banked_model

MODEL = "trl-internal-testing/tiny-Phi3ForCausalLM"


def _logits(model: torch.nn.Module) -> torch.Tensor:
    x = torch.arange(8)[None]
    with torch.no_grad():
        return model(input_ids=x).logits


def test_load_banked_model_peft_adapter(tmp_path):
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    peft_model = get_peft_model(base, LoraConfig(target_modules=["qkv_proj"], r=4))
    # Non-zero adapter weights, so an ignored adapter can't pass by accident.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.normal_()
    out_dir = tmp_path / "subset_0"
    peft_model.save_pretrained(out_dir)
    assert (out_dir / "adapter_config.json").exists()

    run_cfg = MagicConfig(run_path=str(tmp_path), model=MODEL)
    loaded = _load_banked_model(run_cfg, str(out_dir), "cpu")

    torch.testing.assert_close(_logits(loaded), _logits(peft_model))


def test_load_banked_model_full_checkpoint(tmp_path):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    # Perturb so a fresh random or pretrained base can't pass by accident.
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    out_dir = tmp_path / "subset_0"
    model.save_pretrained(out_dir)

    run_cfg = MagicConfig(run_path=str(tmp_path), model=MODEL)
    loaded = _load_banked_model(run_cfg, str(out_dir), "cpu")

    torch.testing.assert_close(_logits(loaded), _logits(model))
