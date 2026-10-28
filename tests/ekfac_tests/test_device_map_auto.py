"""Splitting the model's layers across GPUs in one process must give the same
factors and scores as running it on one GPU."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from bergson.config import (
    DataConfig,
    DistributedConfig,
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    PreprocessConfig,
    QuerySetConfig,
    ScoreConfig,
)
from bergson.data import load_scores
from bergson.hessians.hessian_approximations import (
    FACTOR_SUBDIRS,
    approximate_hessians,
)
from bergson.hessians.pipeline import hessian_pipeline
from bergson.utils.worker_utils import setup_model_and_peft

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Needs two GPUs")


def _index_cfg(run_path: Path, distributed: DistributedConfig) -> IndexConfig:
    # The smallest Pythia that device_map="auto" splits across two GPUs.
    return IndexConfig(
        run_path=str(run_path),
        model="EleutherAI/pythia-70m",
        data=DataConfig(
            dataset="NeelNanda/pile-10k", split="train[:8]", truncation=True
        ),
        token_batch_size=512,
        precision="fp32",
        filter_modules="embed_out",
        distributed=distributed,
    )


# Dataset labels: sampled labels would differ between the two fits.
HESSIAN_CFG = HessianConfig(method="kfac", ev_correction=True, use_dataset_labels=True)


def test_model_is_split(tmp_path: Path):
    cfg = _index_cfg(tmp_path, DistributedConfig(device_map_auto=True))
    assert cfg.distributed.nproc_per_node == 1
    model, _ = setup_model_and_peft(cfg)
    assert len({p.device for p in model.parameters()}) > 1


def test_split_fit_matches_one_gpu(tmp_path: Path):
    one = approximate_hessians(
        _index_cfg(tmp_path / "one", DistributedConfig(nproc_per_node=1)),
        HESSIAN_CFG,
    )
    split = approximate_hessians(
        _index_cfg(tmp_path / "split", DistributedConfig(device_map_auto=True)),
        HESSIAN_CFG,
    )
    for sub in FACTOR_SUBDIRS:
        if not os.path.exists(os.path.join(one, sub)):
            continue
        a = load_file(os.path.join(one, sub, "shard_0.safetensors"))
        b = load_file(os.path.join(split, sub, "shard_0.safetensors"))
        assert a.keys() == b.keys(), sub
        for key in a:
            # Eigenvectors are sign-ambiguous per column; compare up to sign.
            x, y = (
                (a[key].abs(), b[key].abs())
                if sub.startswith("eigen_")
                else (
                    a[key],
                    b[key],
                )
            )
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5, msg=f"{sub}/{key}")


@pytest.mark.parametrize("aggregation", ["none", "mean"])
def test_split_pipeline_matches_one_gpu(tmp_path: Path, aggregation: str):
    def run(name: str, distributed: DistributedConfig) -> np.ndarray:
        cfg = _index_cfg(tmp_path / name, distributed)
        hessian_pipeline(
            cfg,
            HESSIAN_CFG,
            ScoreConfig(batch_size=64),
            PreprocessConfig(),
            HessianPipelineConfig(
                query=QuerySetConfig(
                    dataset="NeelNanda/pile-10k",
                    split="train[8:10]",
                    truncation=True,
                    aggregation=aggregation,
                ),
            ),
        )
        return load_scores(tmp_path / name / "scores")[:]

    expected = run("one", DistributedConfig(nproc_per_node=1))
    split = run("split", DistributedConfig(device_map_auto=True))
    np.testing.assert_allclose(split, expected, rtol=1e-4, atol=1e-6)
