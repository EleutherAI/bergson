import shutil
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from bergson.hessians.hessian_approximations import merge_partitions


@pytest.mark.skipif(sys.platform != "linux", reason="Reads /proc/self/maps")
def test_merge_partitions_unmaps_before_deleting(tmp_path: Path, monkeypatch):
    """No partition file is still mapped when the partitions are deleted, since
    NFS can't remove a directory holding a deleted file that's still mapped."""
    for p in range(2):
        for sub in ("activation_sharded", "eigenvalue_correction_sharded"):
            (tmp_path / f"partition_{p}" / sub).mkdir(parents=True)
            save_file(
                {f"layer_{p}": torch.full((64, 64), float(p))},
                tmp_path / f"partition_{p}" / sub / "shard_0.safetensors",
            )

    real_rmtree = shutil.rmtree

    def rmtree(path):
        maps = Path("/proc/self/maps").read_text()
        assert str(path) not in maps, f"{path} is still mapped"
        real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", rmtree)
    merge_partitions(tmp_path, num_partitions=2, rank=0)

    assert not list(tmp_path.glob("partition_*"))
    merged = load_file(tmp_path / "eigenvalue_correction_sharded" / "shard_0.safetensors")
    assert sorted(merged) == ["layer_0", "layer_1"]
    assert merged["layer_1"].eq(1).all()
