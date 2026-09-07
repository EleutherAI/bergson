"""Which phases of ``run_magic`` -- train, score, validate -- a command runs."""

import csv

import pytest
from datasets import Dataset

from bergson.cli.commands import Magic, Validate
from bergson.config.config_io import read_config
from bergson.magic.cli import run_magic

MODEL = "trl-internal-testing/tiny-Phi3ForCausalLM"


def _datasets(tmp_path) -> dict:
    train = Dataset.from_dict({"text": [f"the cat sat on mat {i}" for i in range(8)]})
    train.save_to_disk(tmp_path / "train")
    query = Dataset.from_dict({"text": ["a dog ran", "birds fly high"]})
    query.save_to_disk(tmp_path / "query")
    return {
        "model": MODEL,
        "batch_size": 2,
        "num_epochs": 1,
        "num_subsets": 2,
        "data": {"dataset": str(tmp_path / "train")},
        "query": {"dataset": str(tmp_path / "query")},
        "distributed": {"nproc_per_node": 1},
    }


def test_magic_scores_and_stops(tmp_path):
    run = tmp_path / "run"
    Magic.from_dict(_datasets(tmp_path) | {"run_path": str(run)}).execute()

    assert (run / "scores").exists()
    assert not (run / "validation.csv").exists()
    assert not (run / "subsets.json").exists()


def test_magic_validates_the_scores_it_wrote(tmp_path):
    run = tmp_path / "run"
    Magic.from_dict(
        _datasets(tmp_path)
        | {"run_path": str(run), "skip_validation": False, "save_models": True}
    ).execute()

    with open(run / "validation.csv") as f:
        subsets = {row["subset"] for row in csv.DictReader(f)}
    assert subsets == {"0", "1"}
    assert sorted(p.name for p in (run / "retrained").iterdir()) == [
        "base",
        "subset_0",
        "subset_1",
    ]

    # The validation step saves its own config over the run's.
    (step,) = read_config(run)["steps"]
    assert set(step) == {"magic"}


def test_a_baseline_model_gives_the_same_baseline_as_training_one(tmp_path):
    shared = _datasets(tmp_path)
    magic_run = tmp_path / "magic"
    Magic.from_dict(
        shared | {"run_path": str(magic_run), "save_models": True}
    ).execute()

    def baselines(name, **overrides) -> list[str]:
        Validate.from_dict(
            shared
            | {"run_path": str(tmp_path / name), "scores": str(magic_run / "scores")}
            | overrides
        ).execute()
        with open(tmp_path / name / "summary.csv") as f:
            return [row["baseline_loss"] for row in csv.DictReader(f)]

    trained_here = baselines("trained_here")
    from_model = baselines(
        "from_model", baseline_model=str(magic_run / "retrained" / "base")
    )
    assert from_model == trained_here
    assert not (tmp_path / "from_model" / "checkpoints").exists()


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


def test_validation_needs_a_validation_config(tmp_path):
    from bergson.config.config import TrainingConfig

    cfg = TrainingConfig(run_path=str(tmp_path), model=MODEL)
    with pytest.raises(TypeError, match="ValidationConfig"):
        run_magic(cfg, validate=True)
