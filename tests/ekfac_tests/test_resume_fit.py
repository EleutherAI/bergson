"""A fit interrupted after some passes over the data resumes from them and
gives the factors an uninterrupted fit does."""

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from bergson.config import DataConfig, DistributedConfig, HessianConfig, IndexConfig
from bergson.hessians import hessian_approximations
from bergson.hessians.hessian_approximations import (
    FACTOR_SUBDIRS,
    approximate_hessians,
)


def index_cfg(run_path: Path) -> IndexConfig:
    return IndexConfig(
        run_path=str(run_path),
        model="EleutherAI/pythia-14m",
        data=DataConfig(
            dataset="NeelNanda/pile-10k", split="train[:8]", truncation=True
        ),
        token_batch_size=512,
        precision="fp32",
        filter_modules="embed_out",
        distributed=DistributedConfig(nproc_per_node=1),
    )


class Interrupted(Exception):
    pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("partitions", [1, 2])
def test_resumed_fit_skips_finished_passes(tmp_path: Path, monkeypatch, partitions):
    # Dataset labels: sampled labels would differ between the fits.
    hessian_cfg = HessianConfig(
        method="kfac",
        ev_correction=True,
        use_dataset_labels=True,
        module_partitions=partitions,
    )
    expected = approximate_hessians(index_cfg(tmp_path / "whole"), hessian_cfg)

    collect = hessian_approximations.collect_hessians
    passes = []

    def interrupt_last_pass(*args, ev_correction=False, **kwargs):
        # The last partition's correction pass is the last pass of the fit.
        last = kwargs["path"].endswith(f"partition_{partitions - 1}")
        if ev_correction and (partitions == 1 or last):
            raise Interrupted
        return collect(*args, ev_correction=ev_correction, **kwargs)

    def record(*args, ev_correction=False, **kwargs):
        passes.append((os.path.basename(kwargs["path"]), ev_correction))
        return collect(*args, ev_correction=ev_correction, **kwargs)

    cfg = index_cfg(tmp_path / "resumed")
    monkeypatch.setattr(hessian_approximations, "collect_hessians", interrupt_last_pass)
    with pytest.raises(Interrupted):
        approximate_hessians(cfg, hessian_cfg)

    monkeypatch.setattr(hessian_approximations, "collect_hessians", record)
    resumed = approximate_hessians(cfg, hessian_cfg, resume=True)

    # Only the interrupted pass runs again.
    assert len(passes) == 1 and passes[0][1]
    assert not any(name.endswith(".done") for name in os.listdir(resumed))
    for sub in FACTOR_SUBDIRS:
        a = load_file(os.path.join(expected, sub, "shard_0.safetensors"))
        b = load_file(os.path.join(resumed, sub, "shard_0.safetensors"))
        assert a.keys() == b.keys(), sub
        for key in a:
            torch.testing.assert_close(b[key], a[key], msg=f"{sub}/{key}")
