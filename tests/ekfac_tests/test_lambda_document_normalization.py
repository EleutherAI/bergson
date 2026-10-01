"""The EK-FAC eigenvalues are a second moment per document.

``loss_reduction`` sets the scale inside a document -- the Hessian of the mean
loss is 1/T of the summed one, which
``test_fim_scaling.test_mean_reduction_fits_hessian_of_mean_loss`` pins. This
pins the outer axis: the fit sums one outer product per document, so dividing
by the document count leaves a per-document second moment, matching
kronfluence's ``lambda_matrix.div_(num_lambda_processed)``.

Only the absolute scale is at stake, and the relative damping every method
defaults to cancels it -- so nothing downstream catches a regression here
except the consumers that read the eigenvalues absolutely: absolute damping,
ASTRA's Gauss-Newton products and SOURCE's eigenfunctions.
"""

import glob
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset
from safetensors import safe_open

from bergson.config import DataConfig, HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import approximate_hessians


def _lambda_sum(run_path: Path) -> float:
    """Sum of the fitted eigenvalues, which no eigenbasis convention changes."""
    shards = sorted(
        glob.glob(str(run_path / "eigenvalue_correction_sharded" / "*.safetensors"))
    )
    assert shards, f"no eigenvalue shards under {run_path}"
    total = 0.0
    for path in shards:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                total += f.get_tensor(key).flatten().float().sum().item()
    return total


def _fit(tmp_path: Path, name: str, rows: list[list[int]]) -> float:
    data_path = tmp_path / f"{name}.hf"
    Dataset.from_dict({"input_ids": rows, "labels": rows}).save_to_disk(str(data_path))

    cfg = IndexConfig(
        run_path=str(tmp_path / name),
        model="EleutherAI/pythia-14m",
        precision="fp32",
        token_batch_size=512,
        filter_modules="embed_out",
        data=DataConfig(dataset=str(data_path)),
    )
    cfg.distributed.nproc_per_node = 1
    approximate_hessians(
        cfg, HessianConfig(method="kfac", ev_correction=True, use_dataset_labels=True)
    )
    return _lambda_sum(Path(cfg.run_path))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lambdas_are_per_document(tmp_path: Path):
    """Duplicating the dataset must leave the eigenvalues alone.

    A per-document second moment is unchanged; a plain sum over documents would
    be exactly twice as large, putting the Hessian a factor of the document
    count above every per-document gradient it preconditions.
    """
    rng = np.random.default_rng(0)
    rows = [rng.integers(2, 100, size=16).tolist() for _ in range(8)]

    once = _fit(tmp_path, "once", rows)
    twice = _fit(tmp_path, "twice", rows + rows)

    assert once > 0.0
    assert twice == pytest.approx(once, rel=1e-4), (
        f"duplicating the dataset changed the eigenvalues by "
        f"{twice / once:.4f}x; 2x means they are a sum over documents rather "
        f"than a mean"
    )
