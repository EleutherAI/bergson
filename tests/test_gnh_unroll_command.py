"""The ``gnh-unroll`` command: training or replaying the run, then scoring."""

import os

import torch
from datasets import Dataset

from bergson.cli.commands import GnhUnroll
from bergson.config.config_io import read_config
from bergson.data import load_scores_loss_signed

MODEL = "trl-internal-testing/tiny-Phi3ForCausalLM-3"


def _options(tmp_path, **query) -> dict:
    train = Dataset.from_dict({"text": [f"the cat sat on mat {i}" for i in range(8)]})
    train.save_to_disk(tmp_path / "train")
    docs = Dataset.from_dict({"text": ["a dog ran", "birds fly high"]})
    docs.save_to_disk(tmp_path / "query")
    return {
        "model": MODEL,
        "batch_size": 2,
        "num_epochs": 1,
        "lr_schedule": {"lr": 1e-3},
        "data": {"dataset": str(tmp_path / "train")},
        "query": {"dataset": str(tmp_path / "query"), "chunk_length": 0} | query,
        "distributed": {"nproc_per_node": 1},
        "wandb_project": "",
    }


def test_scores_each_query(tmp_path):
    run = tmp_path / "run"
    GnhUnroll.from_dict(
        _options(tmp_path, aggregation="none") | {"run_path": str(run)}
    ).execute()

    scores, multi_query = load_scores_loss_signed(str(run / "scores"))
    assert multi_query and scores.shape == (8, 2)
    assert torch.isfinite(scores).all() and (scores != 0).any()
    assert not torch.allclose(scores[:, 0], scores[:, 1])
    assert sorted(os.listdir(run / "per_query")) == ["q0.pt", "q1.pt"]
    (step,) = read_config(run)["steps"]
    assert set(step) == {"gnh-unroll"}


def test_replays_another_runs_checkpoints(tmp_path):
    first = tmp_path / "first"
    GnhUnroll.from_dict(
        _options(tmp_path, aggregation="mean") | {"run_path": str(first)}
    ).execute()
    ckpts = first / "checkpoints"
    before = sorted(os.listdir(ckpts))

    second = tmp_path / "second"
    GnhUnroll.from_dict(
        _options(tmp_path, aggregation="mean")
        | {"run_path": str(second), "checkpoints": str(ckpts)}
    ).execute()

    assert sorted(os.listdir(ckpts)) == before
    assert not (second / "checkpoints").exists()
    first_scores, multi_query = load_scores_loss_signed(str(first / "scores"))
    second_scores, _ = load_scores_loss_signed(str(second / "scores"))
    assert not multi_query and first_scores.shape == (8, 1)
    torch.testing.assert_close(second_scores, first_scores)
