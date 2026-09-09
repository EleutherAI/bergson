import json
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from bergson.cli.trackstar import trackstar
from bergson.cli.trak import _train_label_probs, trak
from bergson.config import DataConfig, DistributedConfig, TrackstarConfig, TrakConfig
from bergson.config.config import TrackstarIndexConfig

MODEL = "EleutherAI/pythia-14m"


def _load(scores_dir: Path) -> np.ndarray:
    info = json.loads((scores_dir / "info.json").read_text())
    mmap = np.memmap(
        scores_dir / "scores.bin",
        dtype=info["dtype"],
        mode="r",
        shape=(info["num_rows"],),
    )
    for q in range(info["num_scores"]):
        assert mmap[f"written_{q}"].all()
    return np.stack([mmap[f"score_{q}"] for q in range(info["num_scores"])], 1)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """A tiny pretokenized dataset: 12 training rows, 3 queries."""
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("trak_data")
    rows = [torch.randint(10, 2000, (24,)).tolist() for _ in range(15)]
    train = Dataset.from_dict({"input_ids": rows[:12], "labels": rows[:12]})
    query = Dataset.from_dict({"input_ids": rows[12:], "labels": rows[12:]})
    train.save_to_disk(str(root / "train"))
    query.save_to_disk(str(root / "query"))
    return root


def _index_cfg(run_path: Path, data_dir: Path) -> TrackstarIndexConfig:
    return TrackstarIndexConfig(
        run_path=str(run_path),
        model=MODEL,
        data=DataConfig(dataset=str(data_dir / "train"), split="train"),
        distributed=DistributedConfig(nproc_per_node=1),
        projection_dim=8,
        token_batch_size=256,
        precision="fp32",
    )


def _query_cfg(data_dir: Path) -> DataConfig:
    return DataConfig(dataset=str(data_dir / "query"), split="train")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trak_matches_whitened_trackstar_and_weights_rows(tmp_path, data_dir):
    """Without the Q term TRAK is trackstar with the train Gram alone and no
    unit normalization; with it every training row is scaled by 1 - p_i."""
    plain = _index_cfg(tmp_path / "plain", data_dir)
    trak(plain, TrakConfig(query=_query_cfg(data_dir), q_weighting="none"))
    unweighted = _load(tmp_path / "plain" / "scores")

    ts = _index_cfg(tmp_path / "trackstar", data_dir)
    trackstar(
        ts,
        TrackstarConfig(
            query=_query_cfg(data_dir), mix_hessians=False, stats_sample_size=None
        ),
    )
    reference = _load(tmp_path / "trackstar" / "scores")
    assert unweighted.shape == reference.shape == (12, 3)
    np.testing.assert_allclose(unweighted, reference, rtol=1e-5, atol=1e-6)

    weighted_cfg = _index_cfg(tmp_path / "weighted", data_dir)
    trak(weighted_cfg, TrakConfig(query=_query_cfg(data_dir)))
    weighted = _load(tmp_path / "weighted" / "scores")
    probs = _train_label_probs(weighted_cfg, batch_size=4)
    assert probs.shape == (12,) and (0 < probs).all() and (probs < 1).all()
    saved = np.load(tmp_path / "weighted" / "scores" / "trak_weights.npy")
    np.testing.assert_allclose(saved, 1 - probs)
    np.testing.assert_allclose(
        weighted, unweighted * (1 - probs)[:, None], rtol=1e-4, atol=1e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trak_ensemble_averages_members(tmp_path, data_dir):
    cfg = _index_cfg(tmp_path / "ens", data_dir)
    trak(
        cfg,
        TrakConfig(
            query=_query_cfg(data_dir), checkpoints=[MODEL, MODEL], q_weighting="none"
        ),
    )
    members = [_load(tmp_path / "ens" / f"checkpoint_{i}" / "scores") for i in range(2)]
    mean = _load(tmp_path / "ens" / "scores")
    np.testing.assert_allclose(mean, (members[0] + members[1]) / 2, rtol=1e-6)
