"""Unit tests for the metasmoothness command.

These tests exercise CLI parsing and the scoring function only — they never
call `.execute()`, so they need no GPU and no model downloads.
"""

import pytest
import torch
from datasets import Dataset
from simple_parsing import ArgumentParser, ConflictResolution

from bergson.__main__ import Main
from bergson.magic.metasmoothness import metasmoothness_score


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    return parser


def test_cli_parser_constructs_with_metasmoothness():
    """A single-character field name (``h``) makes simple_parsing derive a ``-h``
    short flag that collides with argparse's ``-h/--help``, which raises while the
    parser is still being built and takes down *every* subcommand, not just this
    one. Guard the whole-parser construction path."""
    build_parser()


def test_fd_step_and_direction_seed_parse():
    args = build_parser().parse_args(
        ["metasmoothness", "run/path", "--fd_step", "0.25", "--direction_seed", "7"]
    )
    assert args.prog.command.fd_step == 0.25
    assert args.prog.command.direction_seed == 7


def test_score_is_one_for_perfectly_linear_response():
    """Equal consecutive steps => both finite differences share a sign everywhere."""
    theta0 = torch.zeros(8)
    delta = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
    assert metasmoothness_score(theta0, theta0 + delta, theta0 + 2 * delta) == 1.0


def test_score_is_negative_when_response_reverses():
    """Second step undoes the first => signs disagree on every moved coordinate."""
    theta0 = torch.zeros(4)
    theta_h = torch.tensor([1.0, 2.0, 3.0, 4.0])
    theta_2h = torch.tensor([0.5, 1.0, 1.5, 2.0])
    assert metasmoothness_score(theta0, theta_h, theta_2h) == -1.0


def test_score_is_one_when_nothing_moves():
    theta = torch.zeros(4)
    assert metasmoothness_score(theta, theta, theta) == 1.0


def test_run_metasmoothness_uses_epoch_pipeline(tmp_path, monkeypatch):
    from datasets import Dataset

    from bergson.config.config import MetasmoothnessConfig
    from bergson.magic import metasmoothness as ms

    ds = Dataset.from_dict({"text": [f"doc {i}" for i in range(4)]})
    monkeypatch.setattr(ms, "setup_data_pipeline", lambda cfg: (ds, len(ds)))
    monkeypatch.setattr(ms, "attach_doc_ids_if_missing", lambda d: d)

    seen = {}
    monkeypatch.setattr(
        ms,
        "launch_distributed_run",
        lambda name, fn, args, dist: seen.setdefault("ds", args[0]),
    )
    cfg = MetasmoothnessConfig(run_path=str(tmp_path), num_epochs=3, seed=0)
    ms.run_metasmoothness(cfg)

    assert len(seen["ds"]) == 3 * len(ds)
    epochs = [seen["ds"]["text"][i * 4 : (i + 1) * 4] for i in range(3)]
    assert len({tuple(e) for e in epochs}) > 1


def test_worker_does_not_reexpand_epochs(monkeypatch):
    """``run_metasmoothness`` passes a dataset already expanded to ``num_epochs``
    shuffled copies; ``metasmoothness_worker`` must build its training stream from
    exactly those docs. Repeating again trains ``num_epochs**2`` epochs — the
    regression left behind when #393 moved epoch expansion into the pipeline."""
    from bergson.config.config import MetasmoothnessConfig
    from bergson.magic import metasmoothness as ms

    num_epochs, base = 3, 24
    expanded = Dataset.from_dict(
        {
            "text": ["x"] * (num_epochs * base),
            "doc_ids": list(range(num_epochs * base)),
        }
    )

    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)
    monkeypatch.setattr(
        ms,
        "pad_dataset_to_batch_size",
        lambda ds, bs, n, label, gr: (ds, len(ds), 0, 0),
    )

    seen = {}

    class _Stop(Exception):
        pass

    def _capture(ds, _bs, **_kw):
        seen["docs"] = len(ds)
        raise _Stop

    monkeypatch.setattr(ms, "DataStream", _capture)

    cfg = MetasmoothnessConfig(
        run_path="unused", num_epochs=num_epochs, batch_size=8, seed=0
    )
    with pytest.raises(_Stop):
        ms.metasmoothness_worker(0, 0, 1, expanded, num_epochs * base, cfg)

    assert seen["docs"] == num_epochs * base
