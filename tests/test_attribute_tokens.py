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
    load_token_gradients,
)
from bergson.builder import Builder
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import IndexConfig, PreprocessConfig
from bergson.data import compute_num_token_grads, create_token_index, load_scores
from bergson.score.score_writer import MemmapTokenScoreWriter
from bergson.score.scorer import Scorer
from bergson.utils.utils import get_gradient_dtype

# ---------------------------------------------------------------------------
# compute_num_token_grads
# ---------------------------------------------------------------------------


def test_compute_num_token_grads_no_labels():
    """Without labels, every position except the last is valid."""
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3], [4, 5, 6, 7]], "length": [3, 4]})
    sl = compute_num_token_grads(ds)
    np.testing.assert_array_equal(sl, [2, 3])


def test_compute_num_token_grads_with_labels():
    """Every real position produces a gradient row (length - 1), regardless of
    the label mask: g_t is nonzero even at prompt positions, so per-token rows
    cover all positions and sum to the per-doc gradient."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            "labels": [[-100, -100, 3, 4, 5], [-100, 7, -100, 9, 10]],
            "length": [5, 5],
        }
    )
    sl = compute_num_token_grads(ds)
    # length - 1, independent of where the completion mask falls
    np.testing.assert_array_equal(sl, [4, 4])


def test_compute_num_token_grads_all_masked():
    """Even with all labels -100 we store length - 1 rows (they carry zero
    gradient, but the row count stays position-based, not label-based)."""
    ds = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3]],
            "labels": [[-100, -100, -100]],
            "length": [3],
        }
    )
    sl = compute_num_token_grads(ds)
    np.testing.assert_array_equal(sl, [2])


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
    assert info["num_grads"] == 10
    assert sum(info["grad_sizes"].values()) == 10

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
    writer.flush()

    # Example 0's cells aren't written yet.
    assert not load_scores(tmp_path).is_written()

    # Write example 0
    scores_ex0 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    writer([0], scores_ex0)
    writer.flush()

    # Read back
    assert (tmp_path / "scores.bin").exists()
    assert (tmp_path / "info.json").exists()

    with (tmp_path / "info.json").open() as f:
        info = json.load(f)
    assert info["attribute_tokens"] is True
    assert info["num_rows"] == 5
    assert info["num_scores"] == 2

    scores = load_scores(tmp_path)
    assert scores.is_written()
    offsets = scores.offsets

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


# ---------------------------------------------------------------------------
# End-to-end: build with attribute_tokens
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_token_build_e2e(tmp_path: Path, model, dataset):
    """Build a token-attribution index and verify output shapes."""
    model = model.float()
    cfg = IndexConfig(
        run_path=str(tmp_path),
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
    assert (cfg.partial_run_path / "gradients.bin").exists()
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
    """Build with partial labels — per-token rows cover every real position
    (length - 1), not just the completion, so they sum to the per-doc grad."""
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

    # length - 1 rows per example, independent of the completion mask
    assert tg.num_token_grads[0] == 4
    assert tg[0].shape[0] == 4

    assert tg.num_token_grads[1] == 4
    assert tg[1].shape[0] == 4


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
        token_batch_size=1024,
        attribute_tokens=True,
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
    scores = load_scores(tmp_path / "scores")
    assert scores.is_written()
    offsets = scores.offsets

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
        token_batch_size=1024,
        attribute_tokens=True,
    )

    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    # Create AdamNormalizer instances with dummy second moments
    from bergson.gradients import AdamNormalizer

    normalizers = {}
    for name, module in model.base_model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in target_modules:
            normalizers[name] = AdamNormalizer(
                weight_avg_sq=torch.ones_like(module.weight),
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
    assert (cfg.partial_run_path / "gradients.bin").exists()
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
        token_batch_size=1024,
        attribute_tokens=attribute_tokens,
        loss_reduction="sum",
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

    # Create normalizers if needed
    from bergson.gradients import AdafactorNormalizer, AdamNormalizer

    if normalizer == "none":
        normalizers = {}
    else:
        normalizers = {}
        for name, module in model.base_model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in target_modules:
                bias_sq = (
                    torch.ones(module.out_features, device=module.weight.device)
                    if include_bias
                    else None
                )
                if normalizer == "adam":
                    normalizers[name] = AdamNormalizer(
                        weight_avg_sq=torch.ones_like(module.weight),
                        bias_avg_sq=bias_sq,
                    )
                else:
                    normalizers[name] = AdafactorNormalizer(
                        row=torch.ones(
                            module.out_features, device=module.weight.device
                        ),
                        col=torch.ones(module.in_features, device=module.weight.device),
                        bias_avg_sq=bias_sq,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("projection_dim", [None, 8])
def test_trackstar_token_scores_sum_to_sequence_scores(
    tmp_path, model, dataset, projection_dim
):
    """Per-token TrackStar scores summed within a doc equal the per-doc score.

    TrackStar's score is ``s(d) = <q, H^-1, grad(d)>`` — a linear function of
    the index gradient. With ``loss_reduction='sum'`` the per-token gradients
    ``g_{d,t}`` satisfy ``sum_t g_{d,t} = grad(d)`` (see
    ``test_token_sum_equals_sequence``); composing with the linear scoring
    operator gives ``sum_t s(d, t) = s(d)`` for every doc d.

    This is the score-level analog of the gradient-level test above, and
    the trackstar analog of ``test_magic_per_token_sums_to_per_doc``.
    """
    from bergson.score.score_writer import InMemorySequenceScoreWriter
    from bergson.score.scorer import Scorer

    model = model.float()
    dataset = dataset.repeat(10)

    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    processor = GradientProcessor(projection_dim=projection_dim)

    seq_collector = _collect_in_memory(
        model,
        dataset,
        processor,
        target_modules,
        attribute_tokens=False,
        run_path=str(tmp_path / "seq"),
    )
    tok_collector = _collect_in_memory(
        model,
        dataset,
        processor,
        target_modules,
        attribute_tokens=True,
        run_path=str(tmp_path / "tok"),
    )

    sorted_modules = sorted(seq_collector.gradients.keys())
    torch.manual_seed(0)
    query_grads = {
        m: torch.randn(1, seq_collector.gradients[m].shape[-1]) for m in sorted_modules
    }

    device = torch.device("cpu")
    dtype = torch.float32

    seq_scorer = Scorer(
        query_grads=query_grads,
        modules=sorted_modules,
        writer=InMemorySequenceScoreWriter(len(dataset), 1, dtype=dtype),
        device=device,
        dtype=dtype,
    )
    seq_scores = seq_scorer.score(seq_collector.gradients).float().cpu().squeeze(-1)

    n_tokens = tok_collector.gradients[sorted_modules[0]].shape[0]
    tok_scorer = Scorer(
        query_grads=query_grads,
        modules=sorted_modules,
        writer=InMemorySequenceScoreWriter(n_tokens, 1, dtype=dtype),
        device=device,
        dtype=dtype,
    )
    tok_scores = tok_scorer.score(tok_collector.gradients).float().cpu().squeeze(-1)

    assert tok_collector.builder is not None
    offsets = tok_collector.builder.offsets

    for i in range(len(dataset)):
        start, end = int(offsets[i]), int(offsets[i + 1])
        tok_sum = tok_scores[start:end].sum()
        torch.testing.assert_close(
            tok_sum,
            seq_scores[i],
            atol=1e-2,
            rtol=1e-2,
            msg=(
                f"Example {i}: per-token score sum {tok_sum:.6e} != "
                f"per-doc score {seq_scores[i]:.6e}"
            ),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("projection_dim", [None, 8])
def test_trackstar_token_scores_sum_to_sequence_scores_on_disk(
    tmp_path, model, dataset, projection_dim
):
    """On-disk per-token scores summed within a doc equal on-disk per-doc scores.

    Same property as ``test_trackstar_token_scores_sum_to_sequence_scores``,
    routed through ``MemmapSequenceScoreWriter`` /
    ``MemmapTokenScoreWriter`` so the disk write + read-back paths
    (info.json, structured scores.bin, offsets.npy) are exercised
    end-to-end.
    """
    from bergson.score.score_writer import (
        MemmapSequenceScoreWriter,
        MemmapTokenScoreWriter,
    )
    from bergson.score.scorer import Scorer

    model = model.float()
    dataset = dataset.repeat(10)

    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    processor = GradientProcessor(projection_dim=projection_dim)

    seq_collector = _collect_in_memory(
        model,
        dataset,
        processor,
        target_modules,
        attribute_tokens=False,
        run_path=str(tmp_path / "seq"),
    )
    tok_collector = _collect_in_memory(
        model,
        dataset,
        processor,
        target_modules,
        attribute_tokens=True,
        run_path=str(tmp_path / "tok"),
    )

    sorted_modules = sorted(seq_collector.gradients.keys())
    torch.manual_seed(0)
    query_grads = {
        m: torch.randn(1, seq_collector.gradients[m].shape[-1]) for m in sorted_modules
    }

    device = torch.device("cpu")
    dtype = torch.float32
    indices = list(range(len(dataset)))

    # --- Per-doc scores via MemmapSequenceScoreWriter ---
    seq_path = tmp_path / "seq_scores"
    seq_writer = MemmapSequenceScoreWriter(seq_path, len(dataset), 1, dtype=dtype)
    seq_scorer = Scorer(
        query_grads=query_grads,
        modules=sorted_modules,
        writer=seq_writer,
        device=device,
        dtype=dtype,
    )
    seq_scorer(indices, seq_collector.gradients)
    seq_writer.flush()

    # --- Per-token scores via MemmapTokenScoreWriter ---
    tok_path = tmp_path / "tok_scores"
    tok_writer = MemmapTokenScoreWriter(tok_path, dataset, 1, dtype=dtype)
    tok_scorer = Scorer(
        query_grads=query_grads,
        modules=sorted_modules,
        writer=tok_writer,
        device=device,
        dtype=dtype,
        attribute_tokens=True,
    )
    # Per-token Scorer expects already-flat per-token gradients of shape
    # [total_valid_tokens, grad_dim] per module, which is exactly what the
    # InMemoryCollector populated with attribute_tokens=True produces.
    tok_scorer(indices, tok_collector.gradients)
    tok_writer.flush()

    # --- Read back from disk, through the same reader for both formats ---
    seq_scores = torch.from_numpy(load_scores(seq_path)[:][:, 0].copy())

    tok_store = load_scores(tok_path)
    assert tok_store.is_written()
    tok_scores = torch.from_numpy(tok_store[:][:, 0].copy())
    offsets = tok_store.offsets

    for i in range(len(dataset)):
        start, end = int(offsets[i]), int(offsets[i + 1])
        tok_sum = tok_scores[start:end].sum()
        torch.testing.assert_close(
            tok_sum,
            seq_scores[i],
            atol=1e-2,
            rtol=1e-2,
            msg=(
                f"Example {i}: on-disk token sum {tok_sum:.6e} != "
                f"on-disk per-doc score {seq_scores[i]:.6e}"
            ),
        )


# ---------------------------------------------------------------------------
# Masked-prompt semantics: per-token rows cover ALL positions and sum to the
# per-document gradient (== the autograd gradient of the completion-masked loss)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_masked_prompt_token_grads_cover_all_positions(tmp_path, model):
    """With a prompt mask there is ONE mask: the loss mask. The per-doc gradient
    is the true autograd gradient of the completion-masked loss, which includes
    prompt-position contributions (completion losses backprop through the prompt
    via causal attention).

    Per-token gradients cover EVERY real position (prompt + completion), so:
      * there are ``length - 1`` rows per example, and
      * the rows sum to the per-doc gradient == the autograd gradient.

    The test also checks that the prompt genuinely contributes (per-token sum !=
    completion-only sum), so the all-positions behavior is materially different
    from a completion-only decomposition -- i.e. masked positions really are
    included.
    """
    model = model.float()

    # Prompt/completion mask: the leading 6 positions are prompt (labels == -100),
    # the trailing 4 are the completion.
    masked = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            "labels": [[-100, -100, -100, -100, -100, -100, 7, 8, 9, 10]],
            "length": [10],
        }
    )

    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    processor = GradientProcessor(projection_dim=None)  # raw grads, no projection

    seq_collector = _collect_in_memory(
        model,
        masked,
        processor,
        target_modules,
        attribute_tokens=False,
        run_path=str(tmp_path / "seq"),
    )
    tok_collector = _collect_in_memory(
        model,
        masked,
        processor,
        target_modules,
        attribute_tokens=True,
        run_path=str(tmp_path / "tok"),
    )
    offsets = tok_collector.builder.offsets
    names = sorted(seq_collector.gradients.keys())
    device = next(model.parameters()).device

    for ex in range(len(masked)):
        length = masked[ex]["length"]
        start, end = int(offsets[ex]), int(offsets[ex + 1])
        # one gradient row per real position except the last (length - 1)
        assert end - start == length - 1

        x = torch.tensor([masked[ex]["input_ids"]], device=device)
        y = torch.tensor([masked[ex]["labels"]], device=device)
        # completion positions only (for the "prompt really contributes" check)
        vmask = torch.zeros(x.size(1), dtype=torch.bool, device=device)
        vmask[:-1] = y[0, 1:] != -100

        # Independent autograd reference: capture g (output grad) and a (input
        # activation) for every target module in a single backward.
        cap: dict[str, dict] = {}
        handles = []
        for n in names:
            m = model.base_model.get_submodule(n)
            handles.append(
                m.register_forward_hook(
                    lambda mod, inp, out, n=n: cap.setdefault(n, {}).update(
                        a=inp[0].detach()
                    )
                )
            )
            handles.append(
                m.register_full_backward_hook(
                    lambda mod, gi, go, n=n: cap.setdefault(n, {}).update(
                        g=go[0].detach()
                    )
                )
            )
        model.zero_grad(set_to_none=True)
        logits = model(x).logits[:, :-1]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y[:, 1:].reshape(-1),
            reduction="sum",  # matches _collect_in_memory(loss_reduction="sum")
            ignore_index=-100,
        )
        loss.backward()
        for h in handles:
            h.remove()

        max_prompt_frac = 0.0
        for n in names:
            module = model.base_model.get_submodule(n)
            o, i_dim = module.weight.shape
            a = cap[n]["a"][0].double()  # [S, I]
            g = cap[n]["g"][0].double()  # [S, O]

            full = g.mT @ a  # sum over ALL positions == autograd weight grad
            comp = (g * vmask.unsqueeze(-1)).mT @ a  # completion positions only

            seq_grad = seq_collector.gradients[n][ex].reshape(o, i_dim).double().cpu()
            tok_sum = (
                tok_collector.gradients[n][start:end]
                .sum(0)
                .reshape(o, i_dim)
                .double()
                .cpu()
            )

            # doc/sequence gradient == true masked-loss gradient (incl. prompt)
            torch.testing.assert_close(
                seq_grad,
                full.cpu(),
                atol=1e-3,
                rtol=1e-3,
                msg=f"ex {ex} module {n}: per-doc grad must equal the autograd "
                f"gradient of the masked loss (no separate gradient mask)",
            )
            # per-token rows now cover ALL positions, so they sum to the doc grad
            torch.testing.assert_close(
                tok_sum,
                full.cpu(),
                atol=1e-3,
                rtol=1e-3,
                msg=f"ex {ex} module {n}: per-token rows must sum to the per-doc "
                f"gradient (all positions included)",
            )
            max_prompt_frac = max(
                max_prompt_frac,
                ((full - comp).norm() / full.norm().clamp_min(1e-12)).item(),
            )

        # Prompt positions genuinely contribute, so including them is materially
        # different from a completion-only decomposition (which would omit them).
        assert max_prompt_frac > 0.1, (
            f"ex {ex}: expected a substantial prompt-position contribution, got "
            f"max fraction {max_prompt_frac:.4f}"
        )
