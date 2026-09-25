import pytest

from bergson.cli.tracin import tracin
from bergson.config import IndexConfig, TracInConfig


def test_tracin_needs_one_lr_per_checkpoint(tmp_path):
    cfg = IndexConfig(run_path=str(tmp_path / "run"), model="EleutherAI/pythia-14m")
    with pytest.raises(ValueError, match="one learning rate each"):
        tracin(cfg, TracInConfig(checkpoints=["a", "b"], lr_list=[1e-3]))
