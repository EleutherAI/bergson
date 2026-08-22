"""Which phases of ``run_magic`` -- train, score, validate -- a command runs."""

import pytest

from bergson.cli.commands import Magic, Validate
from bergson.magic.cli import run_magic


def _calls(monkeypatch) -> list[dict]:
    calls = []
    monkeypatch.setattr(
        "bergson.cli.commands.run_magic", lambda cfg, **kw: calls.append(kw)
    )
    return calls


def test_magic_scores_and_stops(tmp_path, monkeypatch):
    calls = _calls(monkeypatch)
    Magic.from_dict({"run_path": str(tmp_path), "model": "gpt2"}).execute()
    assert calls == [{}]


def test_validate_runs_the_validation_phase(tmp_path, monkeypatch):
    calls = _calls(monkeypatch)
    Validate.from_dict(
        {"run_path": str(tmp_path), "model": "gpt2", "scores": "s"}
    ).execute()
    assert calls == [{"score_path": "s", "validate": True, "baseline_model": ""}]


def test_validate_passes_the_trained_model_through(tmp_path, monkeypatch):
    calls = _calls(monkeypatch)
    Validate.from_dict(
        {
            "run_path": str(tmp_path),
            "model": "gpt2",
            "scores": "s",
            "baseline_model": "runs/magic/model",
        }
    ).execute()
    assert calls[0]["baseline_model"] == "runs/magic/model"


def test_validate_reads_a_bank_without_training(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "bergson.cli.commands.evaluate_retrained",
        lambda cfg, dirs, score_path="": seen.setdefault("dirs", dirs),
    )
    Validate.from_dict(
        {"run_path": str(tmp_path), "scores": "s", "retrained_dir": "a,b"}
    ).execute()
    assert seen["dirs"] == ["a", "b"]


def test_magic_dispatches_a_validation_of_its_own_scores(tmp_path, monkeypatch):
    calls = _calls(monkeypatch)
    Magic.from_dict(
        {"run_path": str(tmp_path), "model": "gpt2", "skip_validation": False}
    ).execute()
    assert calls == [
        {},
        {
            "score_path": str(tmp_path / "scores"),
            "validate": True,
            "baseline_model": "",
        },
    ]


def test_a_chained_validation_reuses_the_trained_model(tmp_path, monkeypatch):
    (tmp_path / "retrained" / "base").mkdir(parents=True)
    calls = _calls(monkeypatch)
    Magic.from_dict(
        {"run_path": str(tmp_path), "model": "gpt2", "skip_validation": False}
    ).execute()
    assert calls[1]["baseline_model"] == str(tmp_path / "retrained" / "base")


def test_a_chained_validation_keeps_the_training_config(tmp_path, monkeypatch):
    """Both phases must train the same model for the baseline to transfer."""
    seen = []
    monkeypatch.setattr(
        "bergson.cli.commands.run_magic", lambda cfg, **kw: seen.append(cfg)
    )
    Magic.from_dict(
        {
            "run_path": str(tmp_path),
            "model": "gpt2",
            "skip_validation": False,
            "num_subsets": 7,
            "num_epochs": 3,
            "seed": 11,
        }
    ).execute()
    magic_cfg, validate_cfg = seen
    assert validate_cfg.num_subsets == 7
    assert (validate_cfg.num_epochs, validate_cfg.seed) == (
        magic_cfg.num_epochs,
        magic_cfg.seed,
    )


def test_a_chained_validation_does_not_wipe_the_scores(tmp_path, monkeypatch):
    """resume keeps run_magic's overwrite guard off the magic step's output."""
    calls = []
    monkeypatch.setattr(
        "bergson.cli.commands.run_magic", lambda cfg, **kw: calls.append(cfg)
    )
    Magic.from_dict(
        {
            "run_path": str(tmp_path),
            "model": "gpt2",
            "skip_validation": False,
            "overwrite": True,
        }
    ).execute()
    assert calls[1].resume is True


def test_validation_needs_a_validation_config(tmp_path):
    from bergson.config.config import TrainingConfig

    cfg = TrainingConfig(run_path=str(tmp_path), model="gpt2")
    with pytest.raises(TypeError, match="ValidationConfig"):
        run_magic(cfg, validate=True)
