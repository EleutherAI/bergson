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
