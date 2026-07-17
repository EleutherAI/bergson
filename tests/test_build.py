from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM

from bergson import GradientProcessor, collect_gradients
from bergson.config import AttentionConfig, IndexConfig
from bergson.data import load_gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_consistency(tmp_path: Path, model, dataset):
    model = model.float()

    cfg = IndexConfig(
        run_path=str(tmp_path),
        token_batch_size=1024,
        loss_reduction="mean",
    )
    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(projection_dim=cfg.projection_dim),
        cfg=cfg,
    )

    index = load_gradients(cfg.partial_run_path)

    cache_path = Path("runs/test_build_cache.npy")

    assert cache_path.exists()
    # if not cache_path.exists():
    #   Regenerate cache
    #   np.save(cache_path, index[index.dtype.names[0]][0])

    cached_item_grad = np.load(cache_path)
    first_module_grad = index[index.dtype.names[0]][0]

    assert np.allclose(first_module_grad, cached_item_grad, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_split_attention_build(tmp_path: Path, model, dataset):
    # tiny-Phi3 o_proj has shape [8, 8] -> split into 2 heads of size 4
    attention_cfgs = {
        "layers.0.self_attn.o_proj": AttentionConfig(
            num_heads=2, head_size=4, head_dim=2
        ),
    }

    cfg = IndexConfig(run_path=str(tmp_path), token_batch_size=1024)

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(projection_dim=16),
        cfg=cfg,
        attention_cfgs=attention_cfgs,
    )

    assert any(
        Path(cfg.partial_run_path).iterdir()
    ), "Expected artifacts in the temp run_path"

    # Verify that per-head gradient columns exist and have non-zero values
    index = load_gradients(cfg.partial_run_path)
    module_names = index.dtype.names
    head_modules = [n for n in module_names if "head_" in n]
    assert (
        len(head_modules) == 2
    ), f"Expected 2 per-head gradient columns, got {len(head_modules)}: {head_modules}"
    for head_name in head_modules:
        assert (
            index[head_name][0].sum().item() != 0.0
        ), f"Gradient for {head_name} is all zeros"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_conv1d_build(tmp_path: Path, dataset):
    model_name = "openai-community/gpt2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, use_safetensors=True
    )

    cfg = IndexConfig(
        run_path=str(tmp_path),
        # GPT-2 model_max_length is 1024
        token_batch_size=1024,
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(projection_dim=16),
        cfg=cfg,
    )

    assert any(
        Path(cfg.partial_run_path).iterdir()
    ), "Expected artifacts in the run path"

    index = load_gradients(cfg.partial_run_path)

    assert len(modules := index.dtype.names) != 0
    assert len(index[modules[0]]) == len(dataset)
    assert index[modules[0]][0].sum().item() != 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tokenizer_build(tmp_path: Path, model, dataset):
    cfg = IndexConfig(
        run_path=str(tmp_path),
        # Use a different tokenizer than the model
        tokenizer="openai-community/gpt2",
        token_batch_size=1024,
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(projection_dim=16),
        cfg=cfg,
    )
    assert any(
        Path(cfg.partial_run_path).iterdir()
    ), "Expected artifacts in the run path"
