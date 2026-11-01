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


def hessian_cfg(partitions: int) -> HessianConfig:
    # Dataset labels: sampled labels would differ between the fits.
    return HessianConfig(
        method="kfac",
        ev_correction=True,
        use_dataset_labels=True,
        module_partitions=partitions,
    )


def interrupted_fit(cfg: IndexConfig, partitions: int, monkeypatch, stop):
    """Fit until ``stop(path, ev_correction)`` says to interrupt a pass."""
    collect = hessian_approximations.collect_hessians

    def interrupt(*args, ev_correction=False, **kwargs):
        if stop(os.path.basename(kwargs["path"]), ev_correction):
            raise Interrupted
        return collect(*args, ev_correction=ev_correction, **kwargs)

    monkeypatch.setattr(hessian_approximations, "collect_hessians", interrupt)
    with pytest.raises(Interrupted):
        approximate_hessians(cfg, hessian_cfg(partitions))
    monkeypatch.setattr(hessian_approximations, "collect_hessians", collect)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "partitions, stop_at",
    [
        (1, (".part", True)),
        (2, ("partition_1", True)),
        # Between partitions: the second partition hasn't started.
        (2, ("partition_1", False)),
    ],
)
def test_resumed_fit_skips_finished_passes(
    tmp_path: Path, monkeypatch, partitions, stop_at
):
    expected = approximate_hessians(
        index_cfg(tmp_path / "whole"), hessian_cfg(partitions)
    )

    cfg = index_cfg(tmp_path / "resumed")
    interrupted_fit(
        cfg,
        partitions,
        monkeypatch,
        lambda name, ev: name.endswith(stop_at[0]) and ev == stop_at[1],
    )

    collect = hessian_approximations.collect_hessians
    passes = []

    def record(*args, ev_correction=False, **kwargs):
        passes.append((os.path.basename(kwargs["path"]), ev_correction))
        return collect(*args, ev_correction=ev_correction, **kwargs)

    monkeypatch.setattr(hessian_approximations, "collect_hessians", record)
    resumed = approximate_hessians(cfg, hessian_cfg(partitions), resume=True)

    # Only the interrupted pass and the ones after it run again.
    first = [p for p in passes if p[0].endswith(stop_at[0])][0]
    assert passes[0] == first
    assert all(not name.startswith("partition_0") for name, _ in passes)
    assert not any(name.endswith(".done") for name in os.listdir(resumed))
    for sub in FACTOR_SUBDIRS:
        a = load_file(os.path.join(expected, sub, "shard_0.safetensors"))
        b = load_file(os.path.join(resumed, sub, "shard_0.safetensors"))
        assert a.keys() == b.keys(), sub
        for key in a:
            torch.testing.assert_close(b[key], a[key], msg=f"{sub}/{key}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_resume_rejects_a_changed_setup(tmp_path: Path, monkeypatch):
    cfg = index_cfg(tmp_path / "run")
    interrupted_fit(cfg, 1, monkeypatch, lambda name, ev: ev)
    cfg.filter_modules = "*.mlp.*"
    with pytest.raises(ValueError, match="Rerun without resume"):
        approximate_hessians(cfg, hessian_cfg(1), resume=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fresh_fit_keeps_files_and_ignores_old_markers(tmp_path: Path, monkeypatch):
    cfg = index_cfg(tmp_path / "run")
    interrupted_fit(cfg, 1, monkeypatch, lambda name, ev: ev)
    (cfg.partial_run_path / "config.yaml").write_text("saved by the caller")

    collect = hessian_approximations.collect_hessians
    passes = []

    def record(*args, ev_correction=False, **kwargs):
        passes.append(ev_correction)
        return collect(*args, ev_correction=ev_correction, **kwargs)

    monkeypatch.setattr(hessian_approximations, "collect_hessians", record)
    out = approximate_hessians(cfg, hessian_cfg(1))
    assert passes == [False, True]
    assert (Path(out) / "config.yaml").read_text() == "saved by the caller"
