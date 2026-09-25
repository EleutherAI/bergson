import os
import shutil

import pytest
import torch

from bergson.config import (
    DataConfig,
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    PreprocessConfig,
    QuerySetConfig,
    ScoreConfig,
)
from bergson.hessians.pipeline import hessian_pipeline


def test_hessian_pipeline_rejects_unit_normalize(tmp_path):
    """Cosine similarity (unit_normalize) is not supported with the
    Kronecker-factored Hessians hessian_pipeline fits and applies."""
    with pytest.raises(ValueError, match="unit_normalize"):
        hessian_pipeline(
            IndexConfig(run_path=str(tmp_path / "run")),
            HessianConfig(method="kfac"),
            ScoreConfig(),
            PreprocessConfig(unit_normalize=True),
            HessianPipelineConfig(),
        )


def test_hessian_pipeline_rejects_global_projection(tmp_path):
    """The query is saved per module, so it can't be scored against a globally
    projected index."""
    with pytest.raises(ValueError, match="projection_target"):
        hessian_pipeline(
            IndexConfig(
                run_path=str(tmp_path / "run"),
                projection_dim=16,
                projection_target="global",
            ),
            HessianConfig(method="kfac"),
            ScoreConfig(),
            PreprocessConfig(),
            HessianPipelineConfig(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hessian_pipeline_resume_reruns_interrupted_steps(tmp_path):
    """A fit or apply that died mid-way leaves only its ``.part`` output, and a
    resumed run must redo it rather than skip it."""
    run = tmp_path / "run"

    def run_pipeline():
        hessian_pipeline(
            IndexConfig(
                run_path=str(run),
                model="EleutherAI/pythia-14m",
                data=DataConfig(
                    dataset="NeelNanda/pile-10k", split="train[:8]", truncation=True
                ),
                token_batch_size=512,
                precision="fp32",
                filter_modules="embed_out",
            ),
            HessianConfig(method="kfac", ev_correction=True, use_dataset_labels=True),
            ScoreConfig(batch_size=64),
            PreprocessConfig(),
            HessianPipelineConfig(
                query=QuerySetConfig(
                    data=DataConfig(
                        dataset="NeelNanda/pile-10k",
                        split="train[8:10]",
                        truncation=True,
                    ),
                    aggregation="none",
                ),
                resume=True,
            ),
        )

    run_pipeline()
    finished = {p.name for p in run.iterdir()}
    assert {"query", "hessian", "kfac_query", "scores"} <= finished

    # Fake an interrupted fit and apply: only their .part directories remain.
    shutil.move(run / "hessian" / "kfac", run / "hessian" / "kfac.part")
    shutil.move(run / "kfac_query", run / "kfac_query.part")
    shutil.rmtree(run / "scores")
    run_pipeline()
    assert (run / "hessian" / "kfac").exists()
    assert not (run / "hessian" / "kfac.part").exists()
    assert (run / "kfac_query").exists()
    assert not (run / "kfac_query.part").exists()
    assert (run / "scores").exists()


def test_hessian_pipeline_passes_projection_config_to_apply(tmp_path, monkeypatch):
    """The apply step must compress the query with the same projection settings
    as the index it is scored against."""
    captured = []

    def fake_apply(name, fn, args, dist_cfg):
        captured.append(args[0])
        os.makedirs(args[0].run_path)

    for name in [
        "build_query",
        "approximate_hessians",
        "score_dataset",
        "save_run_config",
    ]:
        monkeypatch.setattr(f"bergson.hessians.pipeline.{name}", lambda *a, **kw: None)
    monkeypatch.setattr(
        "bergson.hessians.pipeline.launch_distributed_run",
        fake_apply,
    )

    index_cfg = IndexConfig(
        run_path=str(tmp_path / "run"),
        projection_dim=16,
        projection_type="normal",
        projection_scale="row_norm",
        projection_seed=3,
    )
    hessian_pipeline(
        index_cfg,
        HessianConfig(method="kfac"),
        ScoreConfig(),
        PreprocessConfig(),
        HessianPipelineConfig(),
    )

    (ekfac_cfg,) = captured
    assert ekfac_cfg.projection_dim == 16
    assert ekfac_cfg.projection_type == "normal"
    assert ekfac_cfg.projection_scale == "row_norm"
    assert ekfac_cfg.projection_seed == 3
