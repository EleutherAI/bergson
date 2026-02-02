import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from bergson import (
    GradientProcessor,
    TokenGradients,
    collect_gradients,
    load_token_gradients,
)
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig
from bergson.data import TokenBuilder, compute_num_seq_grads, create_token_index
from bergson.score.score_writer import TokenScoreWriter
from bergson.score.scorer import Scorer
from bergson.utils.utils import convert_dtype_to_np, get_gradient_dtype

# ---------------------------------------------------------------------------
# compute_num_seq_grads
# ---------------------------------------------------------------------------


def test_compute_num_seq_grads_no_labels():
    """Without labels, every position except the last is valid."""
    ds = Dataset.from_dict(
        {"input_ids": [[1, 2, 3], [4, 5, 6, 7]], "length": [3, 4]}
    )
    sl = compute_num_seq_grads(ds)
    np.testing.assert_array_equal(sl, [2, 3])


def test_compute_num_seq_grads_with_labels():
    """Only positions where labels[t+1] != -100 produce gradients."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            "labels": [[-100, -100, 3, 4, 5], [-100, 7, -100, 9, 10]],
            "length": [5, 5],
        }
    )
    sl = compute_num_seq_grads(ds)
    # first:  labels[1:] = [-100, 3, 4, 5]  → 3 valid
    # second: labels[1:] = [7, -100, 9, 10] → 3 valid
    np.testing.assert_array_equal(sl, [3, 3])


def test_compute_num_seq_grads_all_masked():
    """All labels -100 → zero valid positions."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "labels": [[-100, -100, -100]],
            "length": [3],
        }
    )
    sl = compute_num_seq_grads(ds)
    np.testing.assert_array_equal(sl, [0])


# ---------------------------------------------------------------------------
# create_token_index / load_token_gradients / TokenGradients
# ---------------------------------------------------------------------------


def test_create_and_load_token_index(tmp_path: Path):
    seq_lengths = np.array([3, 5, 2], dtype=np.int64)
    grad_sizes = {"mod_a": 4, "mod_b": 6}
    dtype = np.float32

    mmap, offsets = create_token_index(tmp_path, seq_lengths, grad_sizes, dtype)

    assert mmap.shape == (10, 10)  # 3+5+2=10 tokens, 4+6=10 grad_dim
    np.testing.assert_array_equal(offsets, [0, 3, 8, 10])

    # Verify metadata
    with (tmp_path / "info.json").open() as f:
        info = json.load(f)
    assert info["token_attributions"] is True
    assert info["total_tokens"] == 10
    assert info["total_grad_dim"] == 10

    # Write some data and reload
    mmap[:] = np.arange(100, dtype=np.float32).reshape(10, 10)
    mmap.flush()

    loaded_mmap, loaded_sl, loaded_off = load_token_gradients(tmp_path)
    np.testing.assert_array_equal(loaded_sl, seq_lengths)
    np.testing.assert_array_equal(loaded_off, offsets)

    # Example 1 (indices 3..7)
    ex1 = loaded_mmap[loaded_off[1] : loaded_off[2]]
    assert ex1.shape == (5, 10)
    np.testing.assert_array_equal(ex1, mmap[3:8])


def test_token_gradients_wrapper(tmp_path: Path):
    seq_lengths = np.array([2, 4], dtype=np.int64)
    grad_sizes = {"m": 3}
    mmap, _ = create_token_index(tmp_path, seq_lengths, grad_sizes, np.float32)

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
    np.testing.assert_array_equal(tg.seq_lengths, [2, 4])
    np.testing.assert_array_equal(tg[0], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(
        tg[1], [[7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]
    )


# ---------------------------------------------------------------------------
# TokenBuilder
# ---------------------------------------------------------------------------


def test_token_builder_write(tmp_path: Path):
    """TokenBuilder correctly writes non-contiguous batches."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8, 9]],
            "length": [3, 4, 2],
        }
    )
    seq_lengths = compute_num_seq_grads(ds)
    # [2, 3, 1]
    grad_sizes = {"m": 2}

    builder = TokenBuilder(
        tmp_path, ds, grad_sizes, torch.float32, seq_lengths
    )

    # Write examples 0 and 2 (non-contiguous!)
    mod_grads = {
        "m": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    }  # 2 + 1 = 3 rows
    builder([0, 2], mod_grads)

    # Write example 1
    mod_grads = {
        "m": torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    }  # 3 rows
    builder([1], mod_grads)
    builder.flush()

    # Verify
    tg = TokenGradients(tmp_path)
    np.testing.assert_array_equal(tg[0], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(
        tg[1], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    )
    np.testing.assert_array_equal(tg[2], [[5.0, 6.0]])


# ---------------------------------------------------------------------------
# TokenScoreWriter
# ---------------------------------------------------------------------------


def test_token_score_writer(tmp_path: Path):
    seq_lengths = np.array([3, 2], dtype=np.int64)

    writer = TokenScoreWriter(
        tmp_path, num_items=2, num_scores=2, seq_lengths=seq_lengths,
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
    assert info["token_attributions"] is True
    assert info["total_tokens"] == 5
    assert info["num_scores"] == 2

    offsets = np.load(tmp_path / "offsets.npy")
    scores = np.memmap(
        tmp_path / "token_scores.bin", dtype=np.float32, mode="r",
        shape=(5, 2),
    )

    # Example 0 at offsets[0]:offsets[1] = 0:3
    np.testing.assert_array_equal(
        scores[offsets[0]:offsets[1]],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )
    # Example 1 at offsets[1]:offsets[2] = 3:5
    np.testing.assert_array_equal(
        scores[offsets[1]:offsets[2]],
        [[10.0, 20.0], [30.0, 40.0]],
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_token_attributions_adam_raises():
    with pytest.raises(ValueError, match="Adam normalizer"):
        IndexConfig(
            run_path="test", token_attributions=True, normalizer="adam"
        )


def test_token_attributions_adafactor_allowed():
    cfg = IndexConfig(
        run_path="test", token_attributions=True, normalizer="adafactor"
    )
    assert cfg.token_attributions is True


# ---------------------------------------------------------------------------
# End-to-end: build with token_attributions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_token_build_e2e(tmp_path: Path, model, dataset):
    """Build a token-attribution index and verify output shapes."""
    model = model.float()
    cfg = IndexConfig(
        run_path=str(tmp_path),
        skip_preconditioners=True,
        token_batch_size=1024,
        token_attributions=True,
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
    assert (cfg.partial_run_path / "seq_lengths.npy").exists()
    assert (cfg.partial_run_path / "offsets.npy").exists()
    assert (cfg.partial_run_path / "info.json").exists()

    # Load and verify shapes
    tg = TokenGradients(cfg.partial_run_path)
    assert len(tg) == len(dataset)

    # Each example has 5 tokens, all labels valid → seq_length = 4
    for i in range(len(dataset)):
        assert tg.seq_lengths[i] == 4
        assert tg[i].shape == (4, tg.mmap.shape[1])
        # Gradients should be non-zero
        assert np.linalg.norm(tg[i].astype(np.float32)) > 0

    # Verify dataset saved with seq_length column
    ds = Dataset.load_from_disk(str(cfg.partial_run_path / "data.hf"))
    assert "seq_length" in ds.column_names
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
        token_attributions=True,
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
    assert tg.seq_lengths[0] == 3
    assert tg[0].shape[0] == 3

    # second example: labels[1:] = [7, -100, 9, 10] → 3 valid
    assert tg.seq_lengths[1] == 3
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
            token_attributions=True,
        ),
        processor=processor,
    )
    shapes = collector.shapes()
    modules = list(shapes.keys())
    # Fake query gradient (1 query)
    query_grads = {
        m: torch.randn(1, math.prod(shapes[m])) for m in modules
    }

    # Compute seq_lengths for the dataset
    seq_lengths = compute_num_seq_grads(dataset)

    score_dtype = get_gradient_dtype(model)
    writer = TokenScoreWriter(
        tmp_path / "scores",
        num_items=len(dataset),
        num_scores=1,
        seq_lengths=seq_lengths,
        dtype=score_dtype,
    )

    scorer = Scorer(
        query_grads=query_grads,
        modules=modules,
        writer=writer,
        device=torch.device("cuda:0"),
        dtype=score_dtype,
        token_attributions=True,
    )

    cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        skip_preconditioners=True,
        token_batch_size=1024,
        token_attributions=True,
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
        ex_scores = scores[offsets[i]:offsets[i + 1]]
        assert ex_scores.shape == (4, 1)
        # Scores should be non-zero
        assert np.abs(ex_scores.astype(np.float32)).sum() > 0
