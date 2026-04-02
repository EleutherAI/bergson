"""Tests for builder device handling and distributed correctness."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset

from bergson.builder import Builder
from bergson.config import PreprocessConfig

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def small_dataset():
    """Dataset with 4 examples, each length 5, all labels valid."""
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5]] * 4,
            "labels": [[1, 2, 3, 4, 5]] * 4,
            "attention_mask": [[1, 1, 1, 1, 1]] * 4,
        }
    )


@pytest.fixture
def grad_sizes():
    return {"module_a": 4, "module_b": 4}


def _make_mod_grads(grad_sizes, batch_size, device="cpu", dtype=torch.float32):
    """Create fake per-example gradients."""
    return {
        name: torch.randn(batch_size, dim, device=device, dtype=dtype)
        for name, dim in grad_sizes.items()
    }


def _no_dist():
    """Patch context: dist not initialized, rank 0."""
    mock = MagicMock()
    mock.is_initialized.return_value = False
    mock.get_rank.return_value = 0
    return patch("bergson.builder.dist", mock)


def _fake_dist(rank):
    """Patch context: dist initialized at given rank."""
    mock = MagicMock()
    mock.is_initialized.return_value = True
    mock.get_rank.return_value = rank
    mock.ReduceOp.SUM = MagicMock()
    return patch("bergson.builder.dist", mock)


def _inject_identity_preconditioner(builder, grad_sizes, device="cuda:0"):
    """Set h_inv to identity matrices on the given device."""
    builder.h_inv = {
        name: torch.eye(dim, device=device, dtype=torch.float32)
        for name, dim in grad_sizes.items()
    }


def _make_builder(dataset, grad_sizes, dtype, cfg, **kwargs):
    """Build with dist mocked out."""
    with _no_dist():
        return Builder(dataset, grad_sizes, dtype, cfg, **kwargs)


# ── Device: uses current_device, not global rank ─────────────────────────


@requires_cuda
def test_builder_multinode_rank(small_dataset, grad_sizes, tmp_path):
    """With global rank=99 (multi-node), cuda:99 doesn't exist.
    Should use torch.cuda.current_device() instead."""
    cfg = PreprocessConfig(aggregation="mean")
    with _fake_dist(rank=99):
        Builder(small_dataset, grad_sizes, torch.float32, cfg, path=tmp_path)


# ── Preconditioner + non-aggregation path ────────────────────────────────


@requires_cuda
def test_builder_no_agg_with_preconditioner(small_dataset, grad_sizes, tmp_path):
    """Non-aggregation path should work when a preconditioner is active."""
    cfg = PreprocessConfig(aggregation="none")
    builder = _make_builder(
        small_dataset,
        grad_sizes,
        torch.float32,
        cfg,
        path=tmp_path,
    )
    _inject_identity_preconditioner(builder, grad_sizes)

    mod_grads = _make_mod_grads(grad_sizes, batch_size=2, device="cuda:0")
    builder([0, 1], mod_grads)


# ── Rank-0 guard in teardown ─────────────────────────────────────────────


@requires_cuda
def test_builder_teardown_rank0_guard(small_dataset, grad_sizes):
    """After dist.reduce(dst=0), only rank 0 has the correct result.
    Non-zero ranks should NOT overwrite grad_buffer with stale local data."""
    cfg = PreprocessConfig(aggregation="mean")

    builder = Builder.__new__(Builder)
    builder.grad_sizes = grad_sizes
    builder.num_items = len(small_dataset)
    builder.preprocess_cfg = cfg
    builder.h_inv = {}
    builder.in_memory_grad_buffer = torch.ones(1, 8, device="cuda:0")
    builder.grad_buffer = np.zeros((1, 8), dtype=np.float32)

    with _fake_dist(rank=1):
        builder.teardown()

    assert np.allclose(
        builder.grad_buffer, 0.0
    ), "Non-zero rank wrote stale data to grad_buffer — missing `if rank == 0` guard"


@requires_cuda
def test_builder_teardown_rank0_guard_disk(small_dataset, grad_sizes, tmp_path):
    """Rank-0 guard works for disk-backed builder too."""
    cfg = PreprocessConfig(aggregation="mean")
    builder = _make_builder(
        small_dataset,
        grad_sizes,
        torch.float32,
        cfg,
        path=tmp_path,
    )

    builder.in_memory_grad_buffer = torch.ones(1, 8, device="cuda:0")

    with _fake_dist(rank=1):
        builder.teardown()

    total_dim = sum(grad_sizes.values())
    row0 = np.frombuffer(
        builder.grad_buffer[0].tobytes(),
        dtype=np.float32,
        count=total_dim,
    )
    np.testing.assert_allclose(row0, 0.0, atol=1e-7)


# ── In-memory all-reduce in teardown ──────────────────────────────────────


@requires_cuda
def test_inmemory_teardown_allreduces(small_dataset, grad_sizes):
    """In-memory non-agg teardown should all-reduce so every rank has full data."""
    cfg = PreprocessConfig(aggregation="none")
    builder = _make_builder(small_dataset, grad_sizes, torch.float32, cfg)

    # Simulate rank 0 writing indices [0, 1]
    builder.grad_buffer[0] = 1.0
    builder.grad_buffer[1] = 2.0

    with _fake_dist(rank=0) as mock_dist:
        builder.teardown()

    # all_reduce should have been called with the buffer tensor
    mock_dist.all_reduce.assert_called_once()


@requires_cuda
def test_disk_teardown_skips_allreduce(small_dataset, grad_sizes, tmp_path):
    """Disk (memmap) non-agg teardown should NOT all-reduce — memmap is shared."""
    cfg = PreprocessConfig(aggregation="none")
    builder = _make_builder(
        small_dataset,
        grad_sizes,
        torch.float32,
        cfg,
        path=tmp_path,
    )

    with _fake_dist(rank=0) as mock_dist:
        builder.teardown()

    mock_dist.all_reduce.assert_not_called()


# ── Construction ─────────────────────────────────────────────────────────


@requires_cuda
def test_builder_construction_no_agg(small_dataset, grad_sizes):
    """In-memory sequence builder can be constructed."""
    cfg = PreprocessConfig(aggregation="none")
    builder = _make_builder(small_dataset, grad_sizes, torch.float32, cfg)
    assert builder.grad_buffer.shape == (4, 8)


@requires_cuda
def test_builder_construction_with_aggregation(small_dataset, grad_sizes):
    """In-memory builder with aggregation creates a 1-row buffer."""
    cfg = PreprocessConfig(aggregation="mean")
    builder = _make_builder(small_dataset, grad_sizes, torch.float32, cfg)
    assert builder.in_memory_grad_buffer is not None
    assert builder.grad_buffer.shape == (1, 8)


# ── Correctness: disk sequence ───────────────────────────────────────────


@requires_cuda
def test_disk_sequence_writes_cuda_grads(small_dataset, grad_sizes, tmp_path):
    """Disk builder writes CUDA grads correctly (no aggregation, no precond)."""
    cfg = PreprocessConfig(aggregation="none")
    builder = _make_builder(
        small_dataset,
        grad_sizes,
        torch.float32,
        cfg,
        path=tmp_path,
    )

    mod_grads = {
        name: torch.ones(2, dim, device="cuda:0") * 5.0
        for name, dim in grad_sizes.items()
    }
    builder([0, 1], mod_grads)

    total_dim = sum(grad_sizes.values())
    row0 = np.frombuffer(
        builder.grad_buffer[0].tobytes(),
        dtype=np.float32,
        count=total_dim,
    )
    np.testing.assert_allclose(row0, 5.0, atol=1e-6)


@requires_cuda
def test_disk_sequence_writes_cpu_grads(small_dataset, grad_sizes, tmp_path):
    """Disk builder works with CPU grads too."""
    cfg = PreprocessConfig(aggregation="none")
    builder = _make_builder(
        small_dataset,
        grad_sizes,
        torch.float32,
        cfg,
        path=tmp_path,
    )

    mod_grads = {name: torch.ones(2, dim) * 5.0 for name, dim in grad_sizes.items()}
    builder([0, 1], mod_grads)
    builder.teardown()

    total_dim = sum(grad_sizes.values())
    row0 = np.frombuffer(
        builder.grad_buffer[0].tobytes(),
        dtype=np.float32,
        count=total_dim,
    )
    np.testing.assert_allclose(row0, 5.0, atol=1e-6)


@requires_cuda
def test_disk_sequence_aggregation_teardown(small_dataset, grad_sizes, tmp_path):
    """Aggregation='mean': accumulate + teardown on rank 0."""
    cfg = PreprocessConfig(aggregation="mean")
    builder = _make_builder(
        small_dataset,
        grad_sizes,
        torch.float32,
        cfg,
        path=tmp_path,
    )

    for _ in range(4):
        mod_grads = {
            name: torch.ones(1, dim, device="cuda:0")
            for name, dim in grad_sizes.items()
        }
        builder([0], mod_grads)

    with _no_dist():
        builder.teardown()

    total_dim = sum(grad_sizes.values())
    row0 = np.frombuffer(
        builder.grad_buffer[0].tobytes(),
        dtype=np.float32,
        count=total_dim,
    )
    np.testing.assert_allclose(row0, 1.0, atol=1e-5)


# ── Correctness: in-memory sequence ──────────────────────────────────────


@requires_cuda
def test_inmemory_sequence_no_agg_no_precond(small_dataset, grad_sizes):
    """In-memory sequence builder writes correct values."""
    cfg = PreprocessConfig(aggregation="none")
    builder = _make_builder(small_dataset, grad_sizes, torch.float32, cfg)

    torch.manual_seed(0)
    mod_grads = _make_mod_grads(grad_sizes, batch_size=2, device="cpu")
    builder([0, 2], mod_grads)

    expected = torch.cat([mod_grads[k] for k in grad_sizes], dim=-1).numpy()
    np.testing.assert_allclose(builder.grad_buffer[0], expected[0], atol=1e-6)
    np.testing.assert_allclose(builder.grad_buffer[2], expected[1], atol=1e-6)
    np.testing.assert_allclose(builder.grad_buffer[1], 0.0)
    np.testing.assert_allclose(builder.grad_buffer[3], 0.0)


# ── Correctness: in-memory token ─────────────────────────────────────────


@requires_cuda
def test_inmemory_token_writes_correct_values(small_dataset, grad_sizes):
    """Per-token gradients land at the right offsets."""
    cfg = PreprocessConfig(aggregation="none")
    builder = _make_builder(
        small_dataset,
        grad_sizes,
        torch.float32,
        cfg,
        attribute_tokens=True,
    )

    assert builder.num_token_grads[0] == 4

    mod_grads = {name: torch.ones(8, dim) * 0.5 for name, dim in grad_sizes.items()}
    builder([0, 1], mod_grads)

    np.testing.assert_allclose(builder.grad_buffer[0:4], 0.5, atol=1e-6)
    np.testing.assert_allclose(builder.grad_buffer[4:8], 0.5, atol=1e-6)
    np.testing.assert_allclose(builder.grad_buffer[8:], 0.0)


@requires_cuda
def test_inmemory_token_noncontiguous_indices(small_dataset, grad_sizes):
    """Writing to non-contiguous example indices."""
    cfg = PreprocessConfig(aggregation="none")
    builder = _make_builder(
        small_dataset,
        grad_sizes,
        torch.float32,
        cfg,
        attribute_tokens=True,
    )

    mod_grads = {name: torch.ones(8, dim) * 2.0 for name, dim in grad_sizes.items()}
    builder([0, 3], mod_grads)

    np.testing.assert_allclose(builder.grad_buffer[0:4], 2.0, atol=1e-6)
    np.testing.assert_allclose(builder.grad_buffer[12:16], 2.0, atol=1e-6)
    np.testing.assert_allclose(builder.grad_buffer[4:12], 0.0)


# ── Correctness: disk token ──────────────────────────────────────────────


@requires_cuda
def test_unit_normalize_no_aggregation(small_dataset, grad_sizes):
    """unit_normalize=True with aggregation='none' should produce unit-norm rows."""
    cfg = PreprocessConfig(aggregation="none", unit_normalize=True)
    builder = _make_builder(small_dataset, grad_sizes, torch.float32, cfg)
    _inject_identity_preconditioner(builder, grad_sizes)

    # Use non-uniform grads so norms aren't already 1
    mod_grads = {
        "module_a": torch.tensor([[3.0, 0.0, 0.0, 4.0]], device="cuda:0"),
        "module_b": torch.tensor([[0.0, 0.0, 0.0, 0.0]], device="cuda:0"),
    }
    builder([0], mod_grads)

    row = builder.grad_buffer[0]
    norm = np.linalg.norm(row)
    assert norm > 0, "Row is all zeros — normalization not applied"
    np.testing.assert_allclose(
        norm,
        1.0,
        atol=1e-5,
        err_msg="Gradients should be unit normalized when unit_normalize=True",
    )


@requires_cuda
def test_disk_token_writes_and_flushes(small_dataset, grad_sizes, tmp_path):
    """Disk token builder writes per-token grads to memmap."""
    cfg = PreprocessConfig(aggregation="none")
    builder = _make_builder(
        small_dataset,
        grad_sizes,
        torch.float32,
        cfg,
        attribute_tokens=True,
        path=tmp_path,
    )

    mod_grads = {name: torch.ones(8, dim) * 3.0 for name, dim in grad_sizes.items()}
    builder([0, 1], mod_grads)
    builder.teardown()

    np.testing.assert_allclose(builder.grad_buffer[0:4], 3.0, atol=1e-6)
    np.testing.assert_allclose(builder.grad_buffer[4:8], 3.0, atol=1e-6)
    assert isinstance(builder.grad_buffer, np.memmap)
