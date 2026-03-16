import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM

from bergson import (
    CollectorComputer,
    GradientProcessor,
    InMemoryCollector,
    collect_gradients,
)
from bergson.config import AttentionConfig, IndexConfig
from bergson.data import load_gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_e2e(tmp_path: Path):
    result = subprocess.run(
        [
            "python",
            "-m",
            "bergson",
            "build",
            "test_e2e",
            "--model",
            "EleutherAI/pythia-14m",
            "--dataset",
            "NeelNanda/pile-10k",
            "--split",
            "train[:100]",
            "--truncation",
            "--projection_dim",
            "4",
            "--token_batch_size",
            "1024",
            "--precision",
            "bf16",
        ],
        cwd=tmp_path,
        capture_output=True,  # Add this
        text=True,  # Add this to get strings instead of bytes
    )

    assert "Error" not in result.stderr, f"Error found in stderr:\n{result.stderr}"

    processor = GradientProcessor.load(tmp_path / "test_e2e")

    assert processor.preconditioners is not None
    assert processor.preconditioners_eigen is not None

    assert len(processor.preconditioners) > 0
    assert len(processor.preconditioners_eigen) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_consistency(tmp_path: Path, model, dataset):
    model = model.float()

    cfg = IndexConfig(
        run_path=str(tmp_path),
        skip_preconditioners=True,
        token_batch_size=1024,
    )
    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(projection_dim=cfg.projection_dim),
        cfg=cfg,
    )

    index = load_gradients(cfg.partial_run_path)

    cache_path = Path("runs/test_build_cache.npy")
    if not cache_path.exists():
        # Regenerate cache, TODO: We shouldn't do this, maybe use dvc
        np.save(cache_path, index[index.dtype.names[0]][0])

    cached_item_grad = np.load(cache_path)
    first_module_grad = index[index.dtype.names[0]][0]

    assert np.allclose(first_module_grad, cached_item_grad, atol=1e-6)
    assert np.allclose(first_module_grad, cached_item_grad, atol=1e-6)
    assert np.allclose(first_module_grad, cached_item_grad, atol=1e-3)

    # Test BF16 consistency
    model = model.to(torch.bfloat16)
    cfg = IndexConfig(
        run_path=str(tmp_path),
        skip_preconditioners=True,
        token_batch_size=1024,
        precision="bf16",
    )

    collector = InMemoryCollector(
        model.base_model,
        processor=GradientProcessor(projection_dim=cfg.projection_dim),
        data=dataset,
        cfg=cfg,
    )
    computer = CollectorComputer(
        model=model,  # type: ignore
        data=dataset,
        collector=collector,
        batches=[[idx] for idx in range(len(dataset))],
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="New worker - Collecting gradients")

    print(list(collector.gradients.keys()))
    first_bf16_mod_grad = collector.gradients[index.dtype.names[0]][0]

    # Assert BF16 gradients are relatively close to FP32 ones
    assert (
        torch.nn.functional.cosine_similarity(
            first_bf16_mod_grad, torch.from_numpy(cached_item_grad), dim=0
        )
        > 0.99
    )
    assert np.allclose(first_bf16_mod_grad.float().numpy(), cached_item_grad, atol=1e-3)


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
        # This build hangs in pytest with preconditioners enabled.
        # It works when run directly so it may be a pytest issue.
        skip_preconditioners=True,
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
