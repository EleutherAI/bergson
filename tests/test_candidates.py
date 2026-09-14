"""Scoring only the rows an earlier run ranked highest."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset, load_from_disk

from bergson.cli.commands import Score
from bergson.collector.gradient_collectors import GradientCollector
from bergson.config import (
    CandidateConfig,
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
)
from bergson.config.config_io import save_run_config
from bergson.data import load_scores
from bergson.query.query_index import csv_recorder
from bergson.score.candidates import candidate_count, select_candidates
from bergson.score.score import score_dataset
from bergson.score.score_writer import save_sequence_scores

from .test_score import _write_query_index

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


def test_select_candidates_from_query_record(tmp_path: Path):
    path = str(tmp_path / "record.csv")
    with csv_recorder(path) as record:
        assert record is not None
        for query, direction, rows in [
            ("q1", "Top", [7, 2, 9]),
            ("q1", "Bottom", [0, 4, 1]),
            ("q2", "Top", [2, 5, 3]),
            ("q2", "Bottom", [8, 0, 6]),
        ]:
            for rank, row in enumerate(rows):
                record([query, direction, f"text {row}", row, -rank])

    cfg = CandidateConfig(scores=path)
    np.testing.assert_array_equal(select_candidates(cfg, 10), [2, 3, 5, 7, 9])
    cfg.top_k = 2
    np.testing.assert_array_equal(select_candidates(cfg, 10), [2, 5, 7])
    cfg.direction = "detractors"
    np.testing.assert_array_equal(select_candidates(cfg, 10), [0, 4, 8])

    with pytest.raises(ValueError, match="rows"):
        select_candidates(cfg, 8)
    cfg.fraction = 0.5
    with pytest.raises(ValueError, match="fraction"):
        select_candidates(cfg, 10)


def _token_dataset(num_rows: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(4, 9, size=num_rows)
    ids = [rng.integers(2, 100, size=int(n)).tolist() for n in lengths]
    return Dataset.from_dict({"input_ids": ids, "labels": ids})


def test_score_dataset_with_candidates(tmp_path: Path, model):
    num_rows, num_queries = 12, 3
    train = _token_dataset(num_rows, seed=0)
    train.save_to_disk(str(tmp_path / "train.hf"))

    shapes = GradientCollector(
        model.base_model, data=train, cfg=IndexConfig(run_path=str(tmp_path))
    ).shapes()
    rng = torch.Generator().manual_seed(0)
    grads = {
        name: torch.randn(num_queries, math.prod(shape), generator=rng)
        for name, shape in shapes.items()
    }
    query_path = _write_query_index(
        tmp_path / "query", grads, PreprocessConfig(), num_queries
    )
    earlier = np.random.default_rng(1).normal(size=(num_rows, num_queries))
    earlier_path = _write_earlier(tmp_path / "earlier", earlier, True)

    cfg = IndexConfig(run_path=str(tmp_path / "scores"), model=MODEL)
    cfg.data.dataset = str(tmp_path / "train.hf")
    cfg.distributed.nproc_per_node = 1
    cfg.drop_columns = False
    score_cfg = ScoreConfig(
        query_path=str(query_path),
        candidates=CandidateConfig(scores=earlier_path, top_k=2),
    )
    score_dataset(cfg, score_cfg, PreprocessConfig())

    loaded = load_scores(tmp_path / "scores")
    candidates = loaded.candidates
    np.testing.assert_array_equal(
        candidates, select_candidates(score_cfg.candidates, num_rows)
    )
    assert loaded.is_written() and len(loaded) == len(candidates)
    scored = load_from_disk(str(tmp_path / "scores" / "data.hf"))
    assert scored["input_ids"] == train.select(candidates)["input_ids"]
