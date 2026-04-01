from pathlib import Path
from typing import Any

import pytest
import torch

from bergson import Attributor, FaissConfig, GradientProcessor, collect_gradients
from bergson.config import IndexConfig


def _has_faiss_gpu() -> bool:
    """Check if faiss-gpu is installed and functional."""
    try:
        import faiss  # type: ignore[import]

        return hasattr(faiss, "GpuMultipleClonerOptions")
    except ImportError:
        return False


requires_faiss_gpu = pytest.mark.skipif(
    not _has_faiss_gpu(), reason="faiss-gpu not available"
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attributor(tmp_path: Path, model, dataset):
    cfg = IndexConfig(run_path=str(tmp_path), token_batch_size=1024)
    cfg.skip_preconditioners = True

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(),
        cfg=cfg,
    )

    attr = Attributor(cfg.partial_run_path, device="cpu", unit_norm=True)

    x = torch.tensor(dataset[0]["input_ids"]).unsqueeze(0)

    with attr.trace(model.base_model, 5) as result:
        model(x, labels=x).loss.backward()
        model.zero_grad()

    assert result.scores[0, 0].item() > 0.99  # Same item
    assert result.scores[0, 1].item() < 0.50  # Different item


@requires_faiss_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_faiss(tmp_path: Path, model, dataset):
    dtype: Any = model.dtype
    model.to("cuda")
    dtype = torch.float32 if model.dtype == torch.float32 else torch.float16

    cfg = IndexConfig(run_path=str(tmp_path), token_batch_size=1024)

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(projection_dim=16),
        cfg=cfg,
    )

    attr = Attributor(
        cfg.partial_run_path,
        device="cuda",
        unit_norm=True,
        faiss_cfg=FaissConfig(),
        dtype=dtype,
    )

    x = torch.tensor(dataset[0]["input_ids"]).unsqueeze(0).cuda()

    with attr.trace(model.base_model, 2) as result:
        model(x, labels=x).loss.backward()

        model.zero_grad()

    assert result.scores[0, 0].item() > 0.99  # Same item
    assert result.scores[0, 1].item() < 0.50  # Different item


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attributor_precondition_split(tmp_path: Path, model, dataset):
    """Test split preconditioning (unit_norm=True): H^(-1/2) on both query and index."""
    cfg = IndexConfig(run_path=str(tmp_path), token_batch_size=1024)

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(),
        cfg=cfg,
    )

    attr = Attributor(
        cfg.partial_run_path, device="cpu", unit_norm=True, precondition=True
    )

    x = torch.tensor(dataset[0]["input_ids"]).unsqueeze(0)

    with attr.trace(model.base_model, 5) as result:
        model(x, labels=x).loss.backward()
        model.zero_grad()

    assert result.scores[0, 0].item() > 0.90  # Same item, top match
    assert result.indices[0, 0].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attributor_precondition_one_sided(tmp_path: Path, model, dataset):
    """Test one-sided preconditioning (unit_norm=False): H^(-1) on query only."""
    cfg = IndexConfig(run_path=str(tmp_path), token_batch_size=1024)

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(),
        cfg=cfg,
    )

    attr = Attributor(
        cfg.partial_run_path, device="cpu", unit_norm=False, precondition=True
    )

    x = torch.tensor(dataset[0]["input_ids"]).unsqueeze(0)

    with attr.trace(model.base_model, 5) as result:
        model(x, labels=x).loss.backward()
        model.zero_grad()

    # Same item should still be the top match
    assert result.indices[0, 0].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_attributor_reverse(tmp_path: Path, model, dataset):
    """Test that reverse mode returns lowest influence examples."""
    cfg = IndexConfig(run_path=str(tmp_path), token_batch_size=1024)
    cfg.skip_preconditioners = True

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(),
        cfg=cfg,
    )

    attr = Attributor(cfg.partial_run_path, device="cpu", unit_norm=True)

    x = torch.tensor(dataset[0]["input_ids"]).unsqueeze(0)

    # Get normal results (highest influence)
    with attr.trace(model.base_model, 5) as normal_result:
        model(x, labels=x).loss.backward()
        model.zero_grad()

    # Get reverse results (lowest influence)
    with attr.trace(model.base_model, 5, reverse=True) as reverse_result:
        model(x, labels=x).loss.backward()
        model.zero_grad()

    # In reverse mode, the first result should have lower score than normal mode
    assert reverse_result.scores[0, 0].item() < normal_result.scores[0, 0].item()

    # The same item (index 0) should be first in normal mode but not in reverse
    assert normal_result.indices[0, 0].item() == 0  # Same item is top match
    assert reverse_result.indices[0, 0].item() != 0  # Same item is NOT lowest match


@requires_faiss_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_faiss_reverse(tmp_path: Path, model, dataset):
    """Test that reverse mode works with FAISS index."""
    dtype: Any = model.dtype
    model.to("cuda")
    dtype = torch.float32 if model.dtype == torch.float32 else torch.float16

    cfg = IndexConfig(run_path=str(tmp_path), token_batch_size=1024)

    collect_gradients(
        model=model,
        data=dataset,
        processor=GradientProcessor(projection_dim=16),
        cfg=cfg,
    )

    attr = Attributor(
        cfg.partial_run_path,
        device="cuda",
        unit_norm=True,
        faiss_cfg=FaissConfig(),
        dtype=dtype,
    )

    x = torch.tensor(dataset[0]["input_ids"]).unsqueeze(0).cuda()

    # Get normal results (highest influence)
    with attr.trace(model.base_model, 2) as normal_result:
        model(x, labels=x).loss.backward()
        model.zero_grad()

    # Get reverse results (lowest influence)
    with attr.trace(model.base_model, 2, reverse=True) as reverse_result:
        model(x, labels=x).loss.backward()
        model.zero_grad()

    # In reverse mode, the first result should have lower score than normal mode
    assert reverse_result.scores[0, 0].item() < normal_result.scores[0, 0].item()

    # The same item (index 0) should be first in normal mode but not in reverse
    assert normal_result.indices[0, 0].item() == 0  # Same item is top match
    assert reverse_result.indices[0, 0].item() != 0  # Same item is NOT lowest match
