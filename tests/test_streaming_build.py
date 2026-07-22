"""Building an index from a streaming (IterableDataset) source."""

from pathlib import Path

import pytest
import torch
from datasets import Dataset

from bergson.build import build_worker
from bergson.config import IndexConfig, PreprocessConfig
from bergson.data import load_gradients

N_ROWS = 6


def _dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4] for _ in range(N_ROWS)],
            "length": [4] * N_ROWS,
        }
    )


def _cfg(tmp_path: Path, shard_size: int) -> IndexConfig:
    return IndexConfig(
        run_path=str(tmp_path / "run"),
        model="sshleifer/tiny-gpt2",
        precision="fp32",
        token_batch_size=64,
        projection_dim=4,
        stream_shard_size=shard_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streaming_build_matches_in_memory(tmp_path: Path):
    """A streamed build produces the same number of gradient rows as a
    non-streamed one over the same data."""
    ds = _dataset()

    mem_cfg = _cfg(tmp_path / "mem", shard_size=1000)
    mem_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    build_worker(0, 0, 1, mem_cfg, PreprocessConfig(), ds)
    expected = load_gradients(mem_cfg.partial_run_path)

    stream_cfg = _cfg(tmp_path / "stream", shard_size=1000)
    stream_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    build_worker(0, 0, 1, stream_cfg, PreprocessConfig(), ds.to_iterable_dataset())
    got = load_gradients(stream_cfg.partial_run_path)

    assert len(got) == len(expected) == N_ROWS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streaming_build_rejects_multiple_shards(tmp_path: Path):
    """A second shard would overwrite the first, so it must fail loudly."""
    cfg = _cfg(tmp_path, shard_size=2)  # 6 rows -> 3 shards
    cfg.partial_run_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(AssertionError, match="single shard"):
        build_worker(0, 0, 1, cfg, PreprocessConfig(), _dataset().to_iterable_dataset())
