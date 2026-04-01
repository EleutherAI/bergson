import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from datasets import Dataset

from bergson import (
    CollectorComputer,
    GradientProcessor,
    InMemoryCollector,
    TokenGradients,
    collect_gradients,
    fit_normalizers,
    load_token_gradients,
)
from bergson.builder import Builder
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig, PreprocessConfig
from bergson.data import compute_num_token_grads, create_token_index
from bergson.score.score_writer import MemmapTokenScoreWriter
from bergson.score.scorer import Scorer
from bergson.utils.utils import convert_dtype_to_np, get_gradient_dtype

# ---------------------------------------------------------------------------
# compute_num_token_grads
# ---------------------------------------------------------------------------


def test_compute_num_token_grads_no_labels():
    """Without labels, every position except the last is valid."""
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3], [4, 5, 6, 7]], "length": [3, 4]})
    sl = compute_num_token_grads(ds)
    np.testing.assert_array_equal(sl, [2, 3])


def test_compute_num_token_grads_with_labels():
    """Only positions where labels[t+1] != -100 produce gradients."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            "labels": [[-100, -100, 3, 4, 5], [-100, 7, -100, 9, 10]],
            "length": [5, 5],
        }
    )
    sl = compute_num_token_grads(ds)
    # first:  labels[1:] = [-100, 3, 4, 5]  → 3 valid
    # second: labels[1:] = [7, -100, 9, 10] → 3 valid
    np.testing.assert_array_equal(sl, [3, 3])


def test_compute_num_token_grads_all_masked():
    """All labels -100 → zero valid positions."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "labels": [[-100, -100, -100]],
            "length": [3],
        }
    )
    sl = compute_num_token_grads(ds)
    np.testing.assert_array_equal(sl, [0])


# ---------------------------------------------------------------------------
# create_token_index / load_token_gradients / TokenGradients
# ---------------------------------------------------------------------------


def test_create_and_load_token_index(tmp_path: Path):
    num_token_grads = np.array([3, 5, 2], dtype=np.int64)
    grad_sizes = {"mod_a": 4, "mod_b": 6}
    dtype = np.float32

    mmap, offsets = create_token_index(tmp_path, num_token_grads, grad_sizes, dtype)

    assert mmap.shape == (10, 10)  # 3+5+2=10 tokens, 4+6=10 grad_dim
    np.testing.assert_array_equal(offsets, [0, 3, 8, 10])

    # Verify metadata
    with (tmp_path / "info.json").open() as f:
        info = json.load(f)
    assert info["attribute_tokens"] is True
    assert info["total_tokens"] == 10
    assert info["total_grad_dim"] == 10

    # Write some data and reload
    mmap[:] = np.arange(100, dtype=np.float32).reshape(10, 10)
    mmap.flush()

    loaded_mmap, loaded_ntg, loaded_off = load_token_gradients(tmp_path)
    np.testing.assert_array_equal(loaded_ntg, num_token_grads)
    np.testing.assert_array_equal(loaded_off, offsets)

    # Example 1 (indices 3..7)
    ex1 = loaded_mmap[loaded_off[1] : loaded_off[2]]
    assert ex1.shape == (5, 10)
    np.testing.assert_array_equal(ex1, mmap[3:8])


def test_token_gradients_wrapper(tmp_path: Path):
    num_token_grads = np.array([2, 4], dtype=np.int64)
    grad_sizes = {"m": 3}
    mmap, _ = create_token_index(tmp_path, num_token_grads, grad_sizes, np.float32)

    # Fill with identifiable values
    mmap[0] = [1, 2, 3]
    mmap[1] = [4, 5, 6]
    mmap[2] = [7, 8, 9]
    mmap[3] = [10, 11, 12]
    mmap[4] = [13, 14, 15]
    mmap[5] = [16, 17, 18]
    mmap.flush()

    tg = TokenGradients(tmp_path)
    assert len(tg) == 2
    np.testing.assert_array_equal(tg.num_token_grads, [2, 4])
    np.testing.assert_array_equal(tg[0], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(
        tg[1], [[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]
    )


# ---------------------------------------------------------------------------
# Token Builder
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_token_builder_write(tmp_path: Path):
    """Token builder correctly writes non-contiguous batches."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8, 9]],
            "length": [3, 4, 2],
        }
    )

    # [2, 3, 1]
    grad_sizes = {"m": 2}
    cfg = PreprocessConfig(aggregation="none")

    with patch("bergson.builder.dist") as mock_dist:
        mock_dist.is_initialized.return_value = False
        mock_dist.get_rank.return_value = 0
        builder = Builder(
            ds,
            grad_sizes,
            torch.float32,
            cfg,
            attribute_tokens=True,
            path=tmp_path,
        )

    # Write examples 0 and 2 (non-contiguous!)
    mod_grads = {
        "m": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    }  # 2 + 1 = 3 rows
    builder([0, 2], mod_grads)

    # Write example 1
    mod_grads = {"m": torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])}  # 3 rows
    builder([1], mod_grads)
    builder.flush()

    # Verify
    tg = TokenGradients(tmp_path)
    np.testing.assert_array_equal(tg[0], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(tg[1], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    np.testing.assert_array_equal(tg[2], [[5.0, 6.0]])


# ---------------------------------------------------------------------------
# MemmapTokenScoreWriter
# ---------------------------------------------------------------------------


def test_token_score_writer(tmp_path: Path):
    # lengths [4, 3] → num_token_grads [3, 2]
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4], [5, 6, 7]], "length": [4, 3]})

    writer = MemmapTokenScoreWriter(
        tmp_path,
        data=ds,
        num_scores=2,
        dtype=torch.float32,
    )

    # Write example 1 first (non-contiguous)
    scores_ex1 = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    writer([1], scores_ex1)

    # Write example 0
    scores_ex0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    writer([0], scores_ex0)
    writer.flush()

    # Read back
    assert (tmp_path / "token_scores.bin").exists()
    assert (tmp_path / "info.json").exists()

    with (tmp_path / "info.json").open() as f:
        info = json.load(f)
    assert info["attribute_tokens"] is True
    assert info["total_tokens"] == 5
    assert info["num_scores"] == 2

    offsets = np.load(tmp_path / "offsets.npy")
    scores = np.memmap(
        tmp_path / "token_scores.bin",
        dtype=np.float32,
        mode="r",
        shape=(5, 2),
    )

    # Example 0 at offsets[0]:offsets[1] = 0:3
    np.testing.assert_array_equal(
        scores[offsets[0] : offsets[1]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )
    # Example 1 at offsets[1]:offsets[2] = 3:5
    np.testing.assert_array_equal(
        scores[offsets[1] : offsets[2]],
        [[10.0, 20.0], [30.0, 40.0]],
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_attribute_tokens_adam_allowed():
    """Adam normalizer is now compatible with attribute_tokens."""
    cfg = IndexConfig(run_path="test", attribute_tokens=True, normalizer="adam")
    assert cfg.attribute_tokens is True
    assert cfg.normalizer == "adam"


def test_attribute_tokens_adafactor_allowed():
    cfg = IndexConfig(run_path="test", attribute_tokens=True, normalizer="adafactor")
    assert cfg.attribute_tokens is True


# ---------------------------------------------------------------------------
# End-to-end: build with attribute_tokens
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_token_build_e2e(tmp_path: Path, model, dataset):
    """Build a token-attribution index and verify output shapes."""
    model = model.float()
    cfg = IndexConfig(
        run_path=str(tmp_path),
        skip_preconditioners=True,
        token_batch_size=1024,
        attribute_tokens=True,
    )
    processor = GradientProcessor(projection_dim=16)

    collect_gradients(
        model=model,
        data=dataset,
        processor=processor,
        cfg=cfg,
    )

    # Verify artifacts exist
    assert (cfg.partial_run_path / "token_gradients.bin").exists()
    assert (cfg.partial_run_path / "num_token_grads.npy").exists()
    assert (cfg.partial_run_path / "offsets.npy").exists()
    assert (cfg.partial_run_path / "info.json").exists()

    # Load and verify shapes
    tg = TokenGradients(cfg.partial_run_path)
    assert len(tg) == len(dataset)

    # Each example has 5 tokens, all labels valid → 4 token grads
    for i in range(len(dataset)):
        assert tg.num_token_grads[i] == 4
        assert tg[i].shape == (4, tg.mmap.shape[1])
        # Gradients should be non-zero
        assert np.linalg.norm(tg[i].astype(np.float32)) > 0

    # Verify dataset saved
    ds = Dataset.load_from_disk(str(cfg.partial_run_path / "data.hf"))
    assert "loss" in ds.column_names


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_token_build_with_labels(tmp_path: Path, model):
    """Build with partial labels — only assistant tokens get gradients."""
    model = model.float()
    dataset = Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ],
            "labels": [
                [-100, -100, 3, 4, 5],
                [-100, 7, -100, 9, 10],
            ],
            "length": [5, 5],
        }
    )

    cfg = IndexConfig(
        run_path=str(tmp_path),
        skip_preconditioners=True,
        token_batch_size=1024,
        attribute_tokens=True,
    )
    processor = GradientProcessor(projection_dim=16)

    collect_gradients(
        model=model,
        data=dataset,
        processor=processor,
        cfg=cfg,
    )

    tg = TokenGradients(cfg.partial_run_path)

    # first example: labels[1:] = [-100, 3, 4, 5] → 3 valid
    assert tg.num_token_grads[0] == 3
    assert tg[0].shape[0] == 3

    # second example: labels[1:] = [7, -100, 9, 10] → 3 valid
    assert tg.num_token_grads[1] == 3
    assert tg[1].shape[0] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_token_score_e2e(tmp_path: Path, model, dataset):
    """Build token index then score against a query."""
    model = model.float()
    processor = GradientProcessor(projection_dim=16)

    collector = GradientCollector(
        model.base_model,
        data=dataset,
        cfg=IndexConfig(
            run_path=str(tmp_path / "dummy"),
            attribute_tokens=True,
        ),
        processor=processor,
    )
    shapes = collector.shapes()
    modules = list(shapes.keys())
    # Fake query gradient (1 query)
    query_grads = {m: torch.randn(1, math.prod(shapes[m])) for m in modules}

    score_dtype = get_gradient_dtype(model)
    writer = MemmapTokenScoreWriter(
        tmp_path / "scores",
        data=dataset,
        num_scores=1,
        dtype=score_dtype,
    )

    scorer = Scorer(
        query_grads=query_grads,
        modules=modules,
        writer=writer,
        device=torch.device("cuda:0"),
        dtype=score_dtype,
        attribute_tokens=True,
    )

    cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        skip_preconditioners=True,
        token_batch_size=1024,
        attribute_tokens=True,
        skip_index=True,
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=processor,
        cfg=cfg,
        scorer=scorer,
    )

    writer.flush()

    # Verify scores
    offsets = writer.offsets
    total_tokens = int(offsets[-1])
    scores = np.memmap(
        tmp_path / "scores" / "token_scores.bin",
        dtype=convert_dtype_to_np(score_dtype),
        mode="r",
        shape=(total_tokens, 1),
    )

    # All examples should have 4 valid tokens (length 5, all labels valid)
    for i in range(len(dataset)):
        ex_scores = scores[offsets[i] : offsets[i + 1]]
        assert ex_scores.shape == (4, 1)
        # Scores should be non-zero
        assert np.abs(ex_scores.astype(np.float32)).sum() > 0


# ---------------------------------------------------------------------------
# End-to-end: build with attribute_tokens + Adam normalizer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_token_build_adam_e2e(tmp_path: Path, model, dataset):
    """Build a token-attribution index with Adam normalizer."""
    model = model.float()
    dataset = dataset.repeat(10)

    cfg = IndexConfig(
        run_path=str(tmp_path),
        skip_preconditioners=True,
        token_batch_size=1024,
        attribute_tokens=True,
        normalizer="adam",
    )

    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    normalizers = fit_normalizers(
        model,
        dataset,
        cfg=cfg,
        batches=[[idx] for idx in range(len(dataset))],
        target_modules=target_modules,
    )
    processor = GradientProcessor(
        projection_dim=16,
        normalizers=normalizers,
    )

    collect_gradients(
        model=model,
        data=dataset,
        processor=processor,
        cfg=cfg,
        target_modules=target_modules,
    )

    # Verify artifacts exist
    assert (cfg.partial_run_path / "token_gradients.bin").exists()
    assert (cfg.partial_run_path / "num_token_grads.npy").exists()
    assert (cfg.partial_run_path / "offsets.npy").exists()

    # Load and verify shapes
    tg = TokenGradients(cfg.partial_run_path)
    assert len(tg) == len(dataset)

    # Each example has 5 tokens, all labels valid -> 4 token grads
    for i in range(len(dataset)):
        assert tg.num_token_grads[i] == 4
        assert tg[i].shape == (4, tg.mmap.shape[1])
        assert np.linalg.norm(tg[i].astype(np.float32)) > 0


# ---------------------------------------------------------------------------
# Correctness: sum of token grads == sequence grad (sum reduction)
# ---------------------------------------------------------------------------


def _collect_in_memory(
    model,
    dataset,
    processor,
    target_modules,
    attribute_tokens,
    run_path,
    include_bias=False,
):
    """Run InMemoryCollector and return the collector for inspection."""
    cfg = IndexConfig(
        run_path=run_path,
        skip_preconditioners=True,
        token_batch_size=1024,
        attribute_tokens=attribute_tokens,
        loss_reduction="sum",
        skip_index=True,
        include_bias=include_bias,
    )
    cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    collector = InMemoryCollector(
        model=model.base_model,
        data=dataset,
        cfg=cfg,
        processor=processor,
        target_modules=target_modules,
        attention_cfgs={},
    )
    computer = CollectorComputer(
        model=model,
        data=dataset,
        collector=collector,
        cfg=cfg,
    )
    computer.run_with_collector_hooks(desc="Collecting")
    return collector


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("normalizer", ["none", "adam", "adafactor"])
@pytest.mark.parametrize("include_bias", [False, True])
@pytest.mark.parametrize("projection_dim", [None, 8])
def test_token_sum_equals_sequence(
    tmp_path, model, dataset, normalizer, include_bias, projection_dim
):
    """Sum of per-token grads must equal the per-example sequence grad.

    With loss_reduction='sum' the sequence path computes g.mT @ a which
    is exactly sum_s g_s (x) a_s. Since normalize_weight() and
    normalize_bias() are element-wise, they commute with the sum, so
    both paths must agree for all normalizers.
    """
    model = model.float()
    dataset = dataset.repeat(10)

    # tiny-Phi3 has no bias on Linear layers; add zero bias when testing bias
    if include_bias:
        for m in model.base_model.modules():
            if isinstance(m, torch.nn.Linear) and m.bias is None:
                m.bias = torch.nn.Parameter(
                    torch.zeros(m.out_features, device=m.weight.device)
                )

    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    # Fit normalizers if needed
    if normalizer == "none":
        normalizers = {}
    else:
        fit_cfg = IndexConfig(
            run_path=str(tmp_path / "fit"),
            skip_preconditioners=True,
            normalizer=normalizer,
            include_bias=include_bias,
        )
        normalizers = fit_normalizers(
            model,
            dataset,
            cfg=fit_cfg,
            batches=[[idx] for idx in range(len(dataset))],
            target_modules=target_modules,
        )

    processor = GradientProcessor(
        normalizers=normalizers,
        include_bias=include_bias,
        projection_dim=projection_dim,
    )

    # --- Sequence grads (attribute_tokens=False) ---
    seq_collector = _collect_in_memory(
        model,
        dataset,
        processor,
        target_modules,
        attribute_tokens=False,
        run_path=str(tmp_path / "seq"),
        include_bias=include_bias,
    )
    # seq_collector.gradients: {module_name: [N, grad_dim]}

    # --- Token grads (attribute_tokens=True) ---
    tok_collector = _collect_in_memory(
        model,
        dataset,
        processor,
        target_modules,
        attribute_tokens=True,
        run_path=str(tmp_path / "tok"),
        include_bias=include_bias,
    )
    # tok_collector.builder.grad_buffer: [total_tokens, total_grad_dim]

    assert tok_collector.builder is not None
    offsets = tok_collector.builder.offsets

    # Sum token grads per example and compare to sequence grads
    for name, seq_grads in seq_collector.gradients.items():
        tok_grads = tok_collector.gradients[name]  # [total_tokens, grad_dim]
        for i in range(len(dataset)):
            start, end = int(offsets[i]), int(offsets[i + 1])
            tok_sum = tok_grads[start:end].sum(dim=0).float()
            seq_grad = seq_grads[i].float()
            torch.testing.assert_close(
                tok_sum,
                seq_grad,
                atol=1e-2,
                rtol=1e-2,
                msg=f"Module {name}, example {i}: "
                f"token sum and sequence grad diverge",
            )
