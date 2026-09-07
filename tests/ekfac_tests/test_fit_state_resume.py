"""Resuming an interrupted Hessian fit reproduces the uninterrupted covariances."""

import pytest
import torch

from bergson.collector.collector import CollectorComputer, fwd_bwd_hessian_factory
from bergson.config import HessianConfig, IndexConfig
from bergson.hessians.kfac import CovarianceCollector
from bergson.utils.utils import get_device
from tests.ekfac_tests.toy_model import (
    ToyDataConfig,
    ToyLM,
    ToyLMConfig,
    generate_batches,
    generate_dataset,
)


def _toy():
    config = ToyDataConfig(
        vocab_size=16, hidden_size=8, seq_lengths=(8,), num_batches=8
    )
    dataset = generate_dataset(config)
    batches = [[i] for batch in generate_batches(config) for i in batch]
    model = ToyLM(
        ToyLMConfig(vocab_size=config.vocab_size, hidden_size=config.hidden_size)
    )
    model.to(torch.device(get_device()))
    model.eval()
    return model, dataset, batches


def _fit(
    run_path, model, dataset, batches, *, resume=False, save_every=0, stop_after=None
):
    """Run a covariance fit, optionally interrupting it after N state saves."""
    index_cfg = IndexConfig(run_path=str(run_path), loss_reduction="sum")
    index_cfg.partial_run_path.mkdir(parents=True, exist_ok=True)
    collector = CovarianceCollector(
        model=model.base_model,
        target_modules={"linear"},
        dtype=torch.float32,
        path=str(index_cfg.partial_run_path),
    )
    computer = CollectorComputer(
        model=model, data=dataset, batches=batches, collector=collector, cfg=index_cfg
    )
    # Sampled labels would make the gradient covariance depend on RNG state.
    computer.forward_backward = fwd_bwd_hessian_factory(
        index_cfg, HessianConfig(method="kfac", use_dataset_labels=True)
    )

    if stop_after is not None:
        saves = 0
        original = computer._save_fit_state

        def interrupt(*args, **kwargs):
            nonlocal saves
            original(*args, **kwargs)
            saves += 1
            if saves >= stop_after:
                raise KeyboardInterrupt("preempted")

        computer._save_fit_state = interrupt

    computer.run_with_collector_hooks(
        state_name="fit_state", save_every=save_every, resume=resume
    )
    return collector


def test_resumed_fit_matches_uninterrupted(tmp_path):
    model, dataset, batches = _toy()

    whole = _fit(tmp_path / "whole", model, dataset, batches)

    part = tmp_path / "part"
    with pytest.raises(KeyboardInterrupt):
        _fit(part, model, dataset, batches, save_every=1, stop_after=3)
    assert (
        IndexConfig(run_path=str(part)).partial_run_path / "fit_state_rank0.pt"
    ).exists()

    resumed = _fit(part, model, dataset, batches, resume=True, save_every=1)

    for name, expected in whole.A_cov_dict.items():
        torch.testing.assert_close(resumed.A_cov_dict[name], expected)
    for name, expected in whole.S_cov_dict.items():
        torch.testing.assert_close(resumed.S_cov_dict[name], expected)


def test_completed_fit_removes_its_state(tmp_path):
    model, dataset, batches = _toy()
    cfg = IndexConfig(run_path=str(tmp_path / "run"))
    _fit(tmp_path / "run", model, dataset, batches, save_every=1)
    assert not (cfg.partial_run_path / "fit_state_rank0.pt").exists()


def test_state_from_a_different_batch_plan_is_rejected(tmp_path):
    model, dataset, batches = _toy()

    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        _fit(run, model, dataset, batches, save_every=1, stop_after=2)

    with pytest.raises(RuntimeError, match="delete it to start over"):
        _fit(run, model, dataset, batches[:-2], resume=True, save_every=1)
