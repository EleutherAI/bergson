"""Scoring a value set from a streaming (IterableDataset) source."""

from pathlib import Path

import pytest
import torch
from datasets import Dataset

from bergson.build import build_worker
from bergson.config import IndexConfig, PreprocessConfig, ScoreConfig
from bergson.data import load_scores
from bergson.score.score import score_worker

N_ROWS = 6


def _dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4] for _ in range(N_ROWS)],
            "length": [4] * N_ROWS,
        }
    )


def _index_cfg(run_path: Path, shard_size: int) -> IndexConfig:
    return IndexConfig(
        run_path=str(run_path),
        model="sshleifer/tiny-gpt2",
        precision="fp32",
        token_batch_size=64,
        projection_dim=4,
        stream_shard_size=shard_size,
    )


def _build_query(tmp_path: Path) -> Path:
    """Build a query gradient index and return its (partial) path."""
    q_cfg = _index_cfg(tmp_path / "query", shard_size=1000)
    q_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    build_worker(0, 0, 1, q_cfg, PreprocessConfig(), _dataset())
    return q_cfg.partial_run_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streaming_score_matches_in_memory(tmp_path: Path):
    """A streamed score produces the same number of written score rows as a
    non-streamed one over the same value data."""
    query_path = _build_query(tmp_path)
    ds = _dataset()
    score_cfg = ScoreConfig(query_path=str(query_path))

    mem_cfg = _index_cfg(tmp_path / "mem", shard_size=1000)
    mem_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    score_worker(0, 0, 1, mem_cfg, score_cfg, PreprocessConfig(), ds)
    expected = load_scores(mem_cfg.partial_run_path)

    stream_cfg = _index_cfg(tmp_path / "stream", shard_size=1000)
    stream_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    score_worker(
        0, 0, 1, stream_cfg, score_cfg, PreprocessConfig(), ds.to_iterable_dataset()
    )
    got = load_scores(stream_cfg.partial_run_path)

    assert len(got) == len(expected) == N_ROWS
    assert got.is_written() and expected.is_written()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streaming_score_rejects_multiple_shards(tmp_path: Path):
    """A second shard would overwrite the first, so it must fail loudly."""
    query_path = _build_query(tmp_path)
    cfg = _index_cfg(tmp_path / "value", shard_size=2)  # 6 rows -> 3 shards
    cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    score_cfg = ScoreConfig(query_path=str(query_path))

    with pytest.raises(AssertionError, match="single shard"):
        score_worker(
            0,
            0,
            1,
            cfg,
            score_cfg,
            PreprocessConfig(),
            _dataset().to_iterable_dataset(),
        )
