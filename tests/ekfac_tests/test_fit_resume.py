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

import bergson.collector.collector as collection
import bergson.hessians.hessian_approximations as approximations
from bergson.config import HessianConfig, IndexConfig
from bergson.gradients import GradientProcessor
from bergson.hessians.hessian_approximations import (
    collect_hessians,
    fit_autocorrelation_hessian,
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


FACTORED = (approximations, "fwd_bwd_hessian_factory")
AUTOCORRELATION = (collection, "fwd_bwd_factory")
"""Where each kind of fit gets its forward-backward."""


def _crash_after(monkeypatch, num_batches, num_passes=0, fit=FACTORED):
    """Make the forward-backward of the pass after ``num_passes`` finished ones
    raise once ``num_batches`` have run."""
    factory = getattr(*fit)
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

    monkeypatch.setattr(*fit, crashing_factory)


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


def _count_batches(monkeypatch, fit=FACTORED) -> list[int]:
    """Count the forward-backward calls of each pass the fit runs."""
    factory = getattr(*fit)
    counts: list[int] = []

    def counting_factory(*args, **kwargs):
        fwd_bwd = factory(*args, **kwargs)
        counts.append(0)
        this_pass = len(counts) - 1

        def wrapped(*a, **kw):
            counts[this_pass] += 1
            return fwd_bwd(*a, **kw)

        return wrapped

    monkeypatch.setattr(*fit, counting_factory)
    return counts


def _fit_ekfac(model, run_path, seed=0, docs=DOCS, batches=None, hessian_dtype="fp32"):
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
            method="kfac",
            ev_correction=True,
            checkpoint_interval=1,
            hessian_dtype=hessian_dtype,
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


def test_resume_after_a_phase_with_other_settings_is_an_error(
    tmp_path, model, monkeypatch
):
    model = model.to(get_device(0))
    with monkeypatch.context() as m:
        _crash_in_eigendecomposition(m)
        with pytest.raises(RuntimeError, match="interrupted"):
            _fit_ekfac(model, tmp_path)

    with pytest.raises(ValueError, match="hessian_dtype changed"):
        _fit_ekfac(model, tmp_path, hessian_dtype="fp64")


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


def _fit_autocorrelation(
    model, run_path, projection_target, docs=DOCS, batches=None, **index_kwargs
):
    ds = Dataset.from_dict({"input_ids": docs, "length": [len(d) for d in docs]})
    index_cfg = IndexConfig(
        run_path=str(run_path),
        projection_dim=4,
        projection_target=projection_target,
        **index_kwargs,
    )
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    target_modules = {
        name
        for name, module in model.base_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    fit_autocorrelation_hessian(
        model,
        ds,
        index_cfg,
        HessianConfig(method="autocorrelation", checkpoint_interval=1),
        target_modules=target_modules,
        attention_cfgs={},
        batches=batches or [[i] for i in range(len(docs))],
    )
    return index_cfg.partial_run_path


def _assert_same_grams(got_path: Path, expected_path: Path):
    got = GradientProcessor.load(got_path)
    expected = GradientProcessor.load(expected_path)
    assert expected.hessians
    assert got.hessians.keys() == expected.hessians.keys()
    for name in expected.hessians:
        torch.testing.assert_close(
            got.hessians[name], expected.hessians[name], rtol=0, atol=0, msg=name
        )
        for g, e in zip(got.hessians_eigen[name], expected.hessians_eigen[name]):
            torch.testing.assert_close(g, e, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("projection_target", ["per_module", "global"])
def test_resumed_autocorrelation_matches_uninterrupted(
    tmp_path, model, monkeypatch, projection_target
):
    model = model.to(get_device(0))
    full = _fit_autocorrelation(model, tmp_path / "full", projection_target)

    with monkeypatch.context() as m:
        _crash_after(m, 3, fit=AUTOCORRELATION)
        with pytest.raises(RuntimeError, match="interrupted"):
            _fit_autocorrelation(model, tmp_path / "resumed", projection_target)
    with monkeypatch.context() as m:
        counts = _count_batches(m, fit=AUTOCORRELATION)
        resumed = _fit_autocorrelation(model, tmp_path / "resumed", projection_target)
    # Only the batch after the last save is run again.
    assert counts == [1]

    _assert_same_grams(resumed, full)
    torch.testing.assert_close(
        torch.load(resumed / "total_processed.pt", weights_only=True),
        torch.load(full / "total_processed.pt", weights_only=True),
    )
    assert not os.path.exists(resumed / "checkpoint" / "autocorrelation" / "rank_0.pt")


def test_autocorrelation_resume_with_another_projection_is_an_error(
    tmp_path, model, monkeypatch
):
    model = model.to(get_device(0))
    with monkeypatch.context() as m:
        _crash_after(m, 2, fit=AUTOCORRELATION)
        with pytest.raises(RuntimeError, match="interrupted"):
            _fit_autocorrelation(model, tmp_path, "per_module")

    with pytest.raises(ValueError, match="projection_type changed"):
        _fit_autocorrelation(model, tmp_path, "per_module", projection_type="normal")


def _autocorrelation_resume_worker(rank, port, root, projection_target, results):
    # Only rank 0 saves the processor.
    os.environ["LOCAL_RANK"] = str(rank)
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

    _fit_autocorrelation(model, Path(root, "full"), projection_target, **fit)
    with pytest.MonkeyPatch.context() as m:
        _crash_after(m, 3, fit=AUTOCORRELATION)
        # Every rank stops at the same batch, so none waits on the others.
        with pytest.raises(RuntimeError, match="interrupted"):
            _fit_autocorrelation(model, Path(root, "resumed"), projection_target, **fit)
    with pytest.MonkeyPatch.context() as m:
        counts = _count_batches(m, fit=AUTOCORRELATION)
        _fit_autocorrelation(model, Path(root, "resumed"), projection_target, **fit)
    results[rank] = counts
    dist.destroy_process_group()


@pytest.mark.parametrize("projection_target", ["per_module", "global"])
def test_distributed_autocorrelation_resumes_to_uninterrupted(
    tmp_path, monkeypatch, projection_target
):
    # gloo can't communicate CUDA tensors.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.spawn(
        _autocorrelation_resume_worker,
        args=(port, str(tmp_path), projection_target, results),
        nprocs=WORLD_SIZE,
    )

    assert [results[rank] for rank in range(WORLD_SIZE)] == [[1]] * WORLD_SIZE
    _assert_same_grams(tmp_path / "resumed.part", tmp_path / "full.part")
