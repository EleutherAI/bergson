"""An interrupted covariance pass resumes to the same sums as an uninterrupted one."""

import os

import pytest
import torch
from datasets import Dataset

import bergson.hessians.hessian_approximations as approximations
from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.hessian_approximations import collect_hessians
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


def _crash_after(monkeypatch, num_batches):
    """Make the fit's forward-backward raise once ``num_batches`` have run."""
    factory = approximations.fwd_bwd_hessian_factory

    def crashing_factory(*args, **kwargs):
        fwd_bwd = factory(*args, **kwargs)
        calls = {"n": 0}

        def wrapped(*a, **kw):
            if calls["n"] == num_batches:
                raise RuntimeError("interrupted")
            calls["n"] += 1
            return fwd_bwd(*a, **kw)

        return wrapped

    monkeypatch.setattr(approximations, "fwd_bwd_hessian_factory", crashing_factory)


def test_resumed_covariances_match_uninterrupted(tmp_path, model, monkeypatch):
    model = model.to(get_device(0))
    # Labels are sampled, so this also checks the RNG state is restored.
    cfg = HessianConfig(method="kfac", checkpoint_interval=1)

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
    assert not os.path.exists(resumed / "checkpoint" / "rank_0.pt")


def test_resume_with_other_settings_is_an_error(tmp_path, model, monkeypatch):
    model = model.to(get_device(0))
    with monkeypatch.context() as m:
        _crash_after(m, 2)
        with pytest.raises(RuntimeError, match="interrupted"):
            _fit(model, tmp_path, HessianConfig(method="kfac", checkpoint_interval=1))

    other = HessianConfig(method="kfac", checkpoint_interval=1, hessian_dtype="fp64")
    with pytest.raises(ValueError, match="hessian_dtype changed"):
        _fit(model, tmp_path, other)
