"""``bergson hessian --resume`` skips a finished fit and continues an
interrupted one that saved its sums."""

import pytest

import bergson.cli.commands as commands
from bergson.cli.commands import Hessian
from bergson.config import HessianConfig, IndexConfig


@pytest.fixture
def fits(monkeypatch) -> list[bool]:
    """Whether the interrupted fit's save was there, for each fit that ran."""
    seen: list[bool] = []

    def fit(index_cfg, hessian_cfg):
        seen.append((index_cfg.partial_run_path / "checkpoint").exists())

    monkeypatch.setattr(commands, "approximate_hessians", fit)
    return seen


def _command(run_path, **kwargs) -> Hessian:
    resume = kwargs.pop("resume", False)
    return Hessian(
        hessian_cfg=HessianConfig(method="kfac", **kwargs),
        index_cfg=IndexConfig(run_path=str(run_path)),
        resume=resume,
    )


def _interrupt(run_path):
    (run_path.with_name(run_path.name + ".part") / "checkpoint").mkdir(parents=True)


def test_resume_continues_a_checkpointed_fit(tmp_path, fits):
    run_path = tmp_path / "run"
    _interrupt(run_path)
    _command(run_path, checkpoint_interval=5, resume=True).execute()
    assert fits == [True]


def test_resume_restarts_a_fit_without_checkpoints(tmp_path, fits):
    run_path = tmp_path / "run"
    _interrupt(run_path)
    _command(run_path, resume=True).execute()
    assert fits == [False]


def test_resume_skips_a_finished_fit(tmp_path, fits):
    run_path = tmp_path / "run"
    run_path.mkdir()
    _command(run_path, checkpoint_interval=5, resume=True).execute()
    assert fits == []


def test_interrupted_fit_is_an_error_without_resume(tmp_path, fits):
    run_path = tmp_path / "run"
    _interrupt(run_path)
    with pytest.raises(FileExistsError):
        _command(run_path, checkpoint_interval=5).execute()
    assert fits == []
