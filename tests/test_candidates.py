"""Scoring only the rows an earlier run ranked highest."""

from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from bergson.build import build
from bergson.cli.commands import Score
from bergson.config import (
    CandidateConfig,
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
)
from bergson.config.config_io import save_run_config
from bergson.data import load_scores
from bergson.score.candidates import (
    candidate_count,
    merge_scores,
    select_candidates,
    write_merged_scores,
)
from bergson.score.score import score_dataset
from bergson.score.score_writer import (
    MemmapSequenceScoreWriter,
    save_sequence_scores,
)

MODEL = "trl-internal-testing/tiny-Phi3ForCausalLM"


def _write_earlier(path: Path, scores: np.ndarray, higher_is_better: bool) -> str:
    save_sequence_scores(path, scores.astype(np.float32))
    save_run_config(
        Score(
            ScoreConfig(higher_is_better=higher_is_better),
            IndexConfig(run_path=str(path)),
            PreprocessConfig(),
        ),
        path,
    )
    return str(path)


def test_candidate_count():
    assert candidate_count(CandidateConfig(top_k=3), 10) == 3
    assert candidate_count(CandidateConfig(top_k=30), 10) == 10
    assert candidate_count(CandidateConfig(fraction=0.25), 10) == 2
    assert candidate_count(CandidateConfig(fraction=0.01), 10) == 1

    with pytest.raises(ValueError, match="exactly one"):
        candidate_count(CandidateConfig(), 10)
    with pytest.raises(ValueError, match="exactly one"):
        candidate_count(CandidateConfig(top_k=2, fraction=0.5), 10)
    with pytest.raises(ValueError, match="fraction"):
        candidate_count(CandidateConfig(fraction=1.5), 10)


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_select_candidates_unions_each_column(tmp_path: Path, higher_is_better):
    scores = np.array(
        [
            [0.9, 0.1],
            [0.5, 0.5],
            [0.1, 0.9],
            [0.7, 0.2],
            [0.2, 0.7],
        ]
    )
    path = _write_earlier(tmp_path / "earlier", scores, higher_is_better)

    # Either end of either column keeps two of rows 0, 2, 3 and 4.
    for direction in ("proponents", "detractors"):
        cfg = CandidateConfig(scores=path, top_k=2, direction=direction)
        np.testing.assert_array_equal(select_candidates(cfg, 5), [0, 2, 3, 4])

    cfg = CandidateConfig(scores=path, top_k=1)
    top = [0, 2] if higher_is_better else [2, 0]
    np.testing.assert_array_equal(select_candidates(cfg, 5), sorted(top))
    cfg.direction = "detractors"
    np.testing.assert_array_equal(select_candidates(cfg, 5), sorted(top))
    cfg.higher_is_better = not higher_is_better
    np.testing.assert_array_equal(select_candidates(cfg, 5), sorted(top))

    cfg = CandidateConfig(scores=path, top_k=9)
    np.testing.assert_array_equal(select_candidates(cfg, 5), np.arange(5))
    with pytest.raises(ValueError, match="rows"):
        select_candidates(cfg, 6)


@pytest.mark.parametrize("keep_high", [True, False])
@pytest.mark.parametrize("flip", [True, False])
def test_merge_scores(keep_high, flip):
    rng = np.random.default_rng(0)
    num_rows, num_scores = 40, 3
    earlier = rng.normal(size=(num_rows, num_scores))
    candidates = np.array([1, 4, 9, 16, 25, 36])
    # Far larger than the earlier scores, as Shampoo is next to projected KFAC.
    own = rng.normal(size=(len(candidates), num_scores)) * 1e7

    merged = merge_scores(own, earlier, candidates, keep_high=keep_high, flip=flip)
    merged32 = merged.astype(np.float32)

    np.testing.assert_array_equal(merged[candidates], own)

    rest = np.setdiff1d(np.arange(num_rows), candidates)
    oriented = -earlier if flip else earlier
    for col in range(num_scores):
        if keep_high:
            assert merged32[rest, col].max() < merged32[candidates, col].min()
        else:
            assert merged32[rest, col].min() > merged32[candidates, col].max()
        # The order survives float32, with no ties introduced.
        np.testing.assert_array_equal(
            np.argsort(merged32[rest, col], kind="stable"),
            np.argsort(oriented[rest, col], kind="stable"),
        )
        assert len(np.unique(merged32[rest, col])) == len(rest)


def test_merge_scores_broadcasts_one_earlier_column():
    earlier = np.array([[3.0], [1.0], [2.0], [0.0]])
    candidates = np.array([0])
    own = np.array([[5.0, -5.0]])

    merged = merge_scores(own, earlier, candidates, keep_high=True, flip=False)

    assert merged.shape == (4, 2)
    np.testing.assert_array_equal(merged[0], own[0])
    for col in range(2):
        assert merged[1:, col].max() < own[0, col]
        assert merged[2, col] > merged[1, col] > merged[3, col]

    with pytest.raises(ValueError, match="score columns"):
        merge_scores(
            np.zeros((1, 3)), np.zeros((4, 2)), candidates, keep_high=True, flip=False
        )


def test_write_merged_scores(tmp_path: Path):
    rng = np.random.default_rng(1)
    num_rows = 12
    earlier = rng.normal(size=(num_rows, 2))
    cfg = CandidateConfig(scores=_write_earlier(tmp_path / "earlier", earlier, True))
    cfg.top_k = 2
    candidates = select_candidates(cfg, num_rows)
    own = rng.normal(size=(len(candidates), 2))

    store = tmp_path / "scores"
    writer = MemmapSequenceScoreWriter(
        store, num_rows, 2, distributed=False, rows=candidates
    )
    writer(list(range(len(candidates))), torch.from_numpy(own).float())
    writer.flush()
    assert not load_scores(store).is_written()

    write_merged_scores(store, cfg, candidates, higher_is_better=True)

    loaded = load_scores(store)
    assert loaded.is_written()
    merged = np.asarray(loaded[:])
    np.testing.assert_allclose(merged[candidates], own, rtol=1e-6)
    rest = np.setdiff1d(np.arange(num_rows), candidates)
    assert (merged[rest].max(axis=0) < merged[candidates].min(axis=0)).all()
    np.testing.assert_array_equal(np.load(store / "candidates.npy"), candidates)


def _token_dataset(num_rows: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(4, 9, size=num_rows)
    ids = [rng.integers(2, 100, size=int(n)).tolist() for n in lengths]
    return Dataset.from_dict({"input_ids": ids, "labels": ids})


def _index_cfg(run_path: Path, data_path: Path) -> IndexConfig:
    cfg = IndexConfig(run_path=str(run_path), model=MODEL, token_batch_size=64)
    cfg.data.dataset = str(data_path)
    cfg.distributed.nproc_per_node = 1
    return cfg


def test_score_dataset_with_candidates(tmp_path: Path):
    """Rescoring with the same method leaves each column's top rows in place."""
    num_rows = 12
    data_path = tmp_path / "train.hf"
    _token_dataset(num_rows, seed=0).save_to_disk(str(data_path))
    query_data_path = tmp_path / "query.hf"
    _token_dataset(3, seed=1).save_to_disk(str(query_data_path))

    query_cfg = _index_cfg(tmp_path / "query", query_data_path)
    build(query_cfg, PreprocessConfig())

    full_cfg = _index_cfg(tmp_path / "full", data_path)
    full_score_cfg = ScoreConfig(query_path=str(query_cfg.run_path))
    score_dataset(full_cfg, full_score_cfg, PreprocessConfig())
    save_run_config(
        Score(full_score_cfg, full_cfg, PreprocessConfig()), full_cfg.run_path
    )
    full = np.asarray(load_scores(Path(full_cfg.run_path))[:])

    subset_cfg = _index_cfg(tmp_path / "subset", data_path)
    score_cfg = ScoreConfig(
        query_path=str(query_cfg.run_path),
        candidates=CandidateConfig(scores=full_cfg.run_path, top_k=2),
    )
    score_dataset(subset_cfg, score_cfg, PreprocessConfig())

    loaded = load_scores(Path(subset_cfg.run_path))
    assert len(loaded) == num_rows and loaded.is_written()
    merged = np.asarray(loaded[:])
    candidates = np.load(Path(subset_cfg.run_path) / "candidates.npy")

    assert 2 <= len(candidates) <= 6
    np.testing.assert_allclose(merged[candidates], full[candidates], rtol=1e-5)
    rest = np.setdiff1d(np.arange(num_rows), candidates)
    for col in range(full.shape[1]):
        assert set(np.argsort(-merged[:, col])[:2]) == set(
            np.argsort(-full[:, col])[:2]
        )
        assert merged[rest, col].max() < merged[candidates, col].min()
        np.testing.assert_array_equal(
            np.argsort(-merged[rest, col]), np.argsort(-full[rest, col])
        )
