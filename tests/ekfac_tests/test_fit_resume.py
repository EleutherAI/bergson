"""An interrupted Hessian fit resumes to the same factors as an uninterrupted one."""

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

import bergson.hessians.hessian_approximations as approximations
from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import (
    collect_hessians,
    fit_factored_hessians,
)
from bergson.utils.utils import get_device
from tests.ekfac_tests.test_utils import load_sharded_covariances

DOCS = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]


def _fit(model, run_path, hessian_cfg, seed=0):
    ds = Dataset.from_dict({"input_ids": DOCS, "length": [len(d) for d in DOCS]})
    index_cfg = IndexConfig(run_path=str(run_path), include_bias=False)
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    # A resumed run is a new process, so its RNG starts elsewhere.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    collect_hessians(
        model=model,
        data=ds,
        index_cfg=index_cfg,
        batches=[[i] for i in range(len(DOCS))],
        target_modules=target_modules,
        hessian_cfg=hessian_cfg,
    )
    return index_cfg.partial_run_path


def _crash_after(monkeypatch, num_batches, num_passes=0):
    """Make the forward-backward of the pass after ``num_passes`` finished ones
    raise once ``num_batches`` have run."""
    factory = approximations.fwd_bwd_hessian_factory
    passes = {"n": 0}

    def crashing_factory(*args, **kwargs):
        fwd_bwd = factory(*args, **kwargs)
        calls = {"n": 0}
        this_pass = passes["n"]
        passes["n"] += 1

        def wrapped(*a, **kw):
            if this_pass == num_passes and calls["n"] == num_batches:
                raise RuntimeError("interrupted")
            calls["n"] += 1
            return fwd_bwd(*a, **kw)

        return wrapped

    monkeypatch.setattr(approximations, "fwd_bwd_hessian_factory", crashing_factory)


@pytest.mark.parametrize("method", ["kfac", "tkfac", "shampoo"])
def test_resumed_covariances_match_uninterrupted(tmp_path, model, monkeypatch, method):
    model = model.to(get_device(0))
    # Labels are sampled, so this also checks the RNG state is restored.
    cfg = HessianConfig(method=method, checkpoint_interval=1)

    full = _fit(model, tmp_path / "full", cfg)

    with monkeypatch.context() as m:
        _crash_after(m, 3)
        with pytest.raises(RuntimeError, match="interrupted"):
            _fit(model, tmp_path / "resumed", cfg)
    resumed = _fit(model, tmp_path / "resumed", cfg, seed=1)

    for sub in ("activation_sharded", "gradient_sharded"):
        expected = load_sharded_covariances(full / sub)
        got = load_sharded_covariances(resumed / sub)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.load(resumed / "total_processed.pt", weights_only=True),
        torch.load(full / "total_processed.pt", weights_only=True),
    )
    assert not os.path.exists(resumed / "checkpoint" / "covariance" / "rank_0.pt")


def test_resume_with_other_settings_is_an_error(tmp_path, model, monkeypatch):
    model = model.to(get_device(0))
    with monkeypatch.context() as m:
        _crash_after(m, 2)
        with pytest.raises(RuntimeError, match="interrupted"):
            _fit(model, tmp_path, HessianConfig(method="kfac", checkpoint_interval=1))

    other = HessianConfig(method="kfac", checkpoint_interval=1, hessian_dtype="fp64")
    with pytest.raises(ValueError, match="hessian_dtype changed"):
        _fit(model, tmp_path, other)


def test_autocorrelation_checkpointing_is_an_error(tmp_path):
    cfg = HessianConfig(method="autocorrelation", checkpoint_interval=1)
    with pytest.raises(ValueError, match="checkpoint_interval=0"):
        approximations.approximate_hessians(
            IndexConfig(run_path=str(tmp_path / "run")), cfg
        )
    assert not (tmp_path / "run.part").exists()


def _count_batches(monkeypatch) -> list[int]:
    """Count the forward-backward calls of each pass the fit runs."""
    factory = approximations.fwd_bwd_hessian_factory
    counts: list[int] = []

    def counting_factory(*args, **kwargs):
        fwd_bwd = factory(*args, **kwargs)
        counts.append(0)
        this_pass = len(counts) - 1

        def wrapped(*a, **kw):
            counts[this_pass] += 1
            return fwd_bwd(*a, **kw)

        return wrapped

    monkeypatch.setattr(approximations, "fwd_bwd_hessian_factory", counting_factory)
    return counts


def _fit_ekfac(model, run_path, seed=0, docs=DOCS, batches=None):
    ds = Dataset.from_dict({"input_ids": docs, "length": [len(d) for d in docs]})
    index_cfg = IndexConfig(run_path=str(run_path), include_bias=False)
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    fit_factored_hessians(
        model=model,
        data=ds,
        index_cfg=index_cfg,
        hessian_cfg=HessianConfig(
            method="kfac", ev_correction=True, checkpoint_interval=1
        ),
        target_modules=target_modules,
        attention_cfgs={},
        batches=batches or [[i] for i in range(len(docs))],
        path=str(index_cfg.partial_run_path),
        do_eigendecomposition=True,
    )
    return index_cfg.partial_run_path


def _factors(path: Path) -> dict[str, torch.Tensor]:
    return {
        f"{file.relative_to(path)}/{name}": tensor
        for file in sorted(path.rglob("*.safetensors"))
        for name, tensor in load_file(file).items()
    }


def _crash_in_eigendecomposition(monkeypatch):
    def crash(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(approximations, "save_uncorrected_eigenvalues", crash)


@pytest.mark.parametrize("phase", ["covariance", "eigendecomposition", "ev_correction"])
def test_fit_resumed_in_each_phase_matches_uninterrupted(
    tmp_path, model, monkeypatch, phase
):
    model = model.to(get_device(0))
    full = _fit_ekfac(model, tmp_path / "full")

    with monkeypatch.context() as m:
        if phase == "covariance":
            _crash_after(m, 2)
        elif phase == "eigendecomposition":
            _crash_in_eigendecomposition(m)
        else:
            _crash_after(m, 2, num_passes=1)
        with pytest.raises(RuntimeError, match="interrupted"):
            _fit_ekfac(model, tmp_path / "resumed")
    with monkeypatch.context() as m:
        counts = _count_batches(m)
        resumed = _fit_ekfac(model, tmp_path / "resumed", seed=1)
    # Only the batches after the last save are run again: two of the four in
    # the interrupted pass, none in a pass that finished.
    assert (
        counts
        == {
            "covariance": [2, 4],
            "eigendecomposition": [4],
            "ev_correction": [2],
        }[phase]
    )

    expected, got = _factors(full), _factors(resumed)
    assert got.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(got[key], expected[key], rtol=0, atol=0, msg=key)


# Two ranks with four one-document batches each.
WORLD_SIZE = 2
DIST_DOCS = DOCS + [[19, 20], [21, 22, 23], [24], [25, 26, 27, 28]]


def _resume_worker(rank, port, root, phase, results):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Phi3ForCausalLM-3")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    batches = [[i] for i in range(rank, len(DIST_DOCS), WORLD_SIZE)]
    fit = {"docs": DIST_DOCS, "batches": batches}

    _fit_ekfac(model, Path(root, "full"), **fit)
    with pytest.MonkeyPatch.context() as m:
        if phase == "covariance":
            _crash_after(m, 2)
        elif phase == "eigendecomposition":
            _crash_in_eigendecomposition(m)
        else:
            _crash_after(m, 2, num_passes=1)
        # Every rank stops at the same batch, so none waits on the others.
        with pytest.raises(RuntimeError, match="interrupted"):
            _fit_ekfac(model, Path(root, "resumed"), **fit)
    with pytest.MonkeyPatch.context() as m:
        counts = _count_batches(m)
        _fit_ekfac(model, Path(root, "resumed"), seed=1, **fit)
    results[rank] = counts
    dist.destroy_process_group()


@pytest.mark.parametrize("phase", ["covariance", "eigendecomposition", "ev_correction"])
def test_distributed_fit_resumes_to_uninterrupted(tmp_path, monkeypatch, phase):
    # gloo can't communicate CUDA tensors.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.spawn(
        _resume_worker,
        args=(port, str(tmp_path), phase, results),
        nprocs=WORLD_SIZE,
    )

    expected_counts = {
        "covariance": [2, 4],
        "eigendecomposition": [4],
        "ev_correction": [2],
    }[phase]
    assert [results[rank] for rank in range(WORLD_SIZE)] == [
        expected_counts
    ] * WORLD_SIZE

    expected = _factors(tmp_path / "full.part")
    got = _factors(tmp_path / "resumed.part")
    assert got.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(got[key], expected[key], rtol=0, atol=0, msg=key)
