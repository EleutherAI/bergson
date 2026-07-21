import subprocess
from pathlib import Path

import pytest
import torch

from bergson import GradientProcessor
from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import approximate_hessians

from .cli_command import bergson_cmd, bergson_env


def test_ekfac_rejects_nonzero_projection_dim(tmp_path: Path):
    """EK-FAC fitting doesn't support gradient projection, so a nonzero
    projection_dim must fail fast."""
    index_cfg = IndexConfig(run_path=str(tmp_path), projection_dim=16)
    hessian_cfg = HessianConfig(method="kfac", ev_correction=True)

    with pytest.raises(ValueError, match="projection_dim=0"):
        approximate_hessians(index_cfg, hessian_cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_autocorrelation_hessian_e2e(tmp_path: Path):
    result = subprocess.run(
        bergson_cmd(
            "hessian",
            "test_e2e",
            "--model",
            "EleutherAI/pythia-14m",
            "--dataset",
            "NeelNanda/pile-10k",
            "--split",
            "train[:100]",
            "--truncation",
            "--projection_dim",
            "4",
            "--token_batch_size",
            "1024",
            "--precision",
            "bf16",
            "--method",
            "autocorrelation",
        ),
        cwd=tmp_path,
        env=bergson_env(),
        capture_output=True,
        text=True,
    )

    assert "Error" not in result.stderr, f"Error found in stderr:\n{result.stderr}"

    processor = GradientProcessor.load(tmp_path / "test_e2e")

    assert processor.hessians is not None
    assert processor.hessians_eigen is not None

    assert len(processor.hessians) > 0
    assert len(processor.hessians_eigen) > 0
