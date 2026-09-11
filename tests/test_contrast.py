"""Contrastive queries: an aggregated query minus an aggregated control."""

from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from bergson.build import build, subtract_contrast
from bergson.config import DataConfig, DistributedConfig, IndexConfig, PreprocessConfig
from bergson.data import load_gradients, load_scores_loss_signed
from bergson.magic.cli import worker
from bergson.magic.config import MagicConfig

MODEL = "EleutherAI/pythia-14m"
SEQ_LEN = 16


def _rows(num_docs: int, offset: int) -> list[list[int]]:
    return [
        [((d + offset) * SEQ_LEN + t) % 50 + 1 for t in range(SEQ_LEN)]
        for d in range(num_docs)
    ]


def _save(path: Path, num_docs: int, offset: int) -> str:
    rows = _rows(num_docs, offset)
    Dataset.from_dict(
        {"input_ids": rows, "labels": rows, "length": [SEQ_LEN] * num_docs}
    ).save_to_disk(str(path))
    return str(path)


def _index_cfg(run_path: Path, dataset: str, **kwargs) -> IndexConfig:
    return IndexConfig(
        run_path=str(run_path),
        model=MODEL,
        data=DataConfig(dataset=dataset, split="train"),
        distributed=DistributedConfig(nproc_per_node=1),
        projection_dim=8,
        token_batch_size=256,
        precision="fp32",
        **kwargs,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_contrast_is_difference_of_means(tmp_path):
    query = _save(tmp_path / "query", 4, 0)
    control = _save(tmp_path / "control", 6, 4)
    mean = PreprocessConfig(aggregation="mean")

    build(_index_cfg(tmp_path / "q", query), mean)
    build(_index_cfg(tmp_path / "c", control), mean)
    build(
        _index_cfg(tmp_path / "qc", query, contrast=DataConfig(dataset=control)), mean
    )

    q = np.asarray(load_gradients(tmp_path / "q")[0], dtype=np.float64)
    c = np.asarray(load_gradients(tmp_path / "c")[0], dtype=np.float64)
    qc = np.asarray(load_gradients(tmp_path / "qc")[0], dtype=np.float64)
    assert (tmp_path / "qc" / "contrast" / "gradients.bin").is_file()
    # The stores hold reduced-precision gradients; compare at their resolution.
    np.testing.assert_allclose(qc, q - c, rtol=2e-2, atol=1e-3)
    assert np.abs(qc).max() > 0


def test_build_contrast_needs_aggregation(tmp_path):
    cfg = _index_cfg(tmp_path / "x", "unused", contrast=DataConfig(dataset="unused"))
    with pytest.raises(ValueError, match="aggregation"):
        subtract_contrast(cfg, PreprocessConfig(aggregation="none"))


def test_magic_contrast_needs_an_aggregated_query(tmp_path):
    with pytest.raises(ValueError, match="query_method"):
        MagicConfig(
            run_path=str(tmp_path),
            model=MODEL,
            query_method="none",
            query_contrast=DataConfig(dataset="unused"),
        )


def _magic_scores(tmp_path: Path, name: str, train: Dataset, query: Dataset, **kw):
    cfg = MagicConfig(
        run_path=str(tmp_path / name),
        model=MODEL,
        data=DataConfig(dataset="unused"),
        query=DataConfig(dataset="unused"),
        batch_size=2,
        num_epochs=1,
        query_method="mean",
        skip_validation=True,
        **kw,
    )
    worker(0, 0, 1, train, query, len(train), len(query), cfg)
    scores, _ = load_scores_loss_signed(str(tmp_path / name / "scores"))
    return scores.double().reshape(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_magic_contrast_scores_are_query_minus_control(tmp_path):
    """MAGIC is linear in the query gradient, so a contrastive query's scores
    are the query's scores minus the control's."""
    torch.manual_seed(0)
    train = Dataset.from_dict(
        {
            "input_ids": _rows(4, 10),
            "labels": _rows(4, 10),
            "doc_ids": [[d] * SEQ_LEN for d in range(4)],
            "length": [SEQ_LEN] * 4,
        }
    )
    query_rows, control_rows = _rows(2, 0), _rows(2, 2)
    as_query = lambda rows: Dataset.from_dict(  # noqa: E731
        {"input_ids": rows, "labels": rows, "length": [SEQ_LEN] * len(rows)}
    )
    control_path = _save(tmp_path / "control_ds", 2, 2)

    s_query = _magic_scores(tmp_path, "q", train, as_query(query_rows))
    s_control = _magic_scores(tmp_path, "c", train, as_query(control_rows))
    s_contrast = _magic_scores(
        tmp_path,
        "qc",
        train,
        as_query(query_rows),
        query_contrast=DataConfig(dataset=control_path, split="train"),
    )
    assert s_contrast.shape == s_query.shape == (4,)
    torch.testing.assert_close(s_contrast, s_query - s_control, rtol=1e-3, atol=1e-6)
