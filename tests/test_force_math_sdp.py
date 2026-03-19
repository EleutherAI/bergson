"""Test that --force_math_sdp produces padding-invariant gradients."""

from pathlib import Path

import pytest
import torch

from bergson import GradientProcessor, collect_gradients
from bergson.config import IndexConfig
from bergson.data import load_gradients
from bergson.utils.worker_utils import apply_force_math_sdp


def test_apply_force_math_sdp_sets_backends():
    """apply_force_math_sdp disables flash and mem-efficient SDPA."""
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    cfg = IndexConfig(run_path="unused", force_math_sdp=True)
    apply_force_math_sdp(cfg)

    assert not torch.backends.cuda.flash_sdp_enabled()
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def test_apply_force_math_sdp_noop_when_false():
    """apply_force_math_sdp is a no-op when force_math_sdp=False."""
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    cfg = IndexConfig(run_path="unused", force_math_sdp=False)
    apply_force_math_sdp(cfg)

    assert torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.mem_efficient_sdp_enabled()


@pytest.fixture
def short_and_long_dataset():
    """Dataset with two documents of very different lengths.

    The length difference triggers padding divergence in models that
    use flash/mem-efficient SDPA with lower precision.
    """
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "input_ids": [
                list(range(1, 8)),  # short doc (7 tokens)
                list(range(1, 50)),  # long doc (49 tokens)
            ],
            "length": [7, 49],
        }
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_force_math_sdp_persists_through_collect(
    tmp_path: Path, model, short_and_long_dataset
):
    """force_math_sdp stays active after running collect_gradients,
    and the collected gradients for the short doc are consistent
    whether it is batched alone or with a longer doc."""
    model = model.float()

    cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        skip_preconditioners=True,
        token_batch_size=1024,
        force_math_sdp=True,
        projection_dim=0,
    )
    apply_force_math_sdp(cfg)

    # Collect gradients with both docs in one batch
    collect_gradients(
        model=model,
        data=short_and_long_dataset,
        processor=GradientProcessor(),
        cfg=cfg,
    )

    # Verify SDPA backends are still disabled after collection
    assert not torch.backends.cuda.flash_sdp_enabled(), (
        "flash SDP was re-enabled during collect_gradients"
    )
    assert not torch.backends.cuda.mem_efficient_sdp_enabled(), (
        "mem-efficient SDP was re-enabled during collect_gradients"
    )

    mixed_index = load_gradients(cfg.partial_run_path)
    mixed_grad_short = torch.from_numpy(
        mixed_index[mixed_index.dtype.names[0]][0].copy()
    ).float()

    # Now collect with only the short doc
    cfg_alone = IndexConfig(
        run_path=str(tmp_path / "run_alone"),
        skip_preconditioners=True,
        token_batch_size=1024,
        force_math_sdp=True,
        projection_dim=0,
    )

    alone_dataset = short_and_long_dataset.select([0])
    collect_gradients(
        model=model,
        data=alone_dataset,
        processor=GradientProcessor(),
        cfg=cfg_alone,
    )

    alone_index = load_gradients(cfg_alone.partial_run_path)
    alone_grad_short = torch.from_numpy(
        alone_index[alone_index.dtype.names[0]][0].copy()
    ).float()

    cos_sim = torch.nn.functional.cosine_similarity(
        mixed_grad_short.unsqueeze(0), alone_grad_short.unsqueeze(0)
    ).item()

    assert cos_sim > 0.999, (
        f"Gradient cosine similarity {cos_sim:.6f} is too low."
        " force_math_sdp + fp32 should produce near-identical gradients"
        " regardless of batch composition."
    )

    # Restore
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
