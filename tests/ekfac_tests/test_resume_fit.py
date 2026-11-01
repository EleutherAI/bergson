"""A fit interrupted part way resumes from the steps it finished and gives the
factors an uninterrupted fit does."""

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


def hessian_cfg(partitions: int) -> HessianConfig:
    # Dataset labels: sampled labels would differ between the fits.
    return HessianConfig(
        method="kfac",
        ev_correction=True,
        use_dataset_labels=True,
        module_partitions=partitions,
    )


class Interrupted(Exception):
    pass


COLLECT = hessian_approximations.collect_hessians
EIGENDECOMPOSE = hessian_approximations.compute_eigendecomposition


def group(path: str) -> str:
    """The module partition a path belongs to, or "all" without partitions."""
    name = os.path.basename(path)
    return name if name.startswith("partition_") else "all"


def record_steps(monkeypatch, steps: list, interrupt_at: int | None = None):
    """Record each pass over the data and each eigendecomposition the fit runs,
    raising ``Interrupted`` at step ``interrupt_at``."""

    def step(name):
        if len(steps) == interrupt_at:
            raise Interrupted
        steps.append(name)

    def recorded_collect(*args, ev_correction=False, **kwargs):
        step(
            (
                group(kwargs["path"]),
                "corrections" if ev_correction else "covariances",
            )
        )
        return COLLECT(*args, ev_correction=ev_correction, **kwargs)

    def recorded_eigendecompose(path, *args, **kwargs):
        step((group(os.path.dirname(path)), os.path.basename(path)))
        return EIGENDECOMPOSE(path, *args, **kwargs)

    monkeypatch.setattr(hessian_approximations, "collect_hessians", recorded_collect)
    monkeypatch.setattr(
        hessian_approximations, "compute_eigendecomposition", recorded_eigendecompose
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("partitions", [1, 2])
def test_resumed_fit_redoes_only_unfinished_steps(
    tmp_path: Path, monkeypatch, partitions
):
    steps: list = []
    record_steps(monkeypatch, steps)
    expected = approximate_hessians(
        index_cfg(tmp_path / "whole"), hessian_cfg(partitions)
    )
    all_steps = list(steps)

    # Each step that starts a resumable piece of work: a group's covariance pass,
    # its eigendecomposition, or its correction pass.
    starts = [i for i, (_, name) in enumerate(all_steps) if name != "gradient_sharded"]
    for n, start in enumerate(starts):
        cfg = index_cfg(tmp_path / f"resumed_{n}")
        steps.clear()
        record_steps(monkeypatch, steps, interrupt_at=start)
        with pytest.raises(Interrupted):
            approximate_hessians(cfg, hessian_cfg(partitions))

        steps.clear()
        record_steps(monkeypatch, steps)
        resumed = approximate_hessians(cfg, hessian_cfg(partitions), resume=True)
        assert steps == all_steps[start:]

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
    record_steps(monkeypatch, [], interrupt_at=1)
    with pytest.raises(Interrupted):
        approximate_hessians(cfg, hessian_cfg(1))
    cfg.filter_modules = "*.mlp.*"
    with pytest.raises(ValueError, match="Rerun without resume"):
        approximate_hessians(cfg, hessian_cfg(1), resume=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fresh_fit_keeps_files_and_ignores_old_markers(tmp_path: Path, monkeypatch):
    cfg = index_cfg(tmp_path / "run")
    record_steps(monkeypatch, [], interrupt_at=1)
    with pytest.raises(Interrupted):
        approximate_hessians(cfg, hessian_cfg(1))
    (cfg.partial_run_path / "config.yaml").write_text("saved by the caller")

    steps: list = []
    record_steps(monkeypatch, steps)
    out = approximate_hessians(cfg, hessian_cfg(1))
    assert [name for _, name in steps][0] == "covariances"
    assert (Path(out) / "config.yaml").read_text() == "saved by the caller"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reloaded_model_keeps_the_fresh_adapter(tmp_path: Path, monkeypatch):
    """The correction pass runs the same randomly initialized adapter as the
    covariance pass, though the model is reloaded in between."""
    adapters = []

    def record(model, *args, **kwargs):
        adapters.append(
            {
                name: p.detach().cpu().clone()
                for name, p in model.named_parameters()
                if "lora_" in name
            }
        )
        return COLLECT(model, *args, **kwargs)

    monkeypatch.setattr(hessian_approximations, "collect_hessians", record)
    cfg = index_cfg(tmp_path / "run")
    cfg.peft_init_kwargs = "r=4,target_modules=query_key_value|dense"
    approximate_hessians(cfg, hessian_cfg(1))

    covariance_pass, correction_pass = adapters
    assert covariance_pass and covariance_pass.keys() == correction_pass.keys()
    for name, weight in covariance_pass.items():
        torch.testing.assert_close(correction_pass[name], weight, msg=name)
