"""Unit tests for the metasmoothness command.

These tests exercise CLI parsing and the scoring function only — they never
call `.execute()`, so they need no GPU and no model downloads.
"""

import pytest
import torch
from datasets import Dataset
from simple_parsing import ArgumentParser, ConflictResolution

from bergson.__main__ import Main
from bergson.magic.data_stream import Padding


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
        lambda ds, bs, n, label, gr: (ds, len(ds), Padding()),
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


def test_worker_forwards_grad_accum_and_clipping(monkeypatch):
    """``metasmoothness_worker`` must pass ``grad_accum_steps`` and
    ``max_grad_norm`` through to ``trainer.train``: dropping them makes each
    rank run its whole batch in one graph (OOM at large batch/context) and
    diverges from the clipped trajectory MAGIC would train."""
    from bergson.config.config import MetasmoothnessConfig
    from bergson.magic import metasmoothness as ms

    ds = Dataset.from_dict({"text": ["x"] * 8, "doc_ids": list(range(8))})

    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: None)
    monkeypatch.setattr(
        ms,
        "pad_dataset_to_batch_size",
        lambda d, bs, n, label, gr: (d, len(d), Padding()),
    )

    class _Stream:
        def __init__(self, *a, **k):
            self.weights = torch.nn.Parameter(torch.ones(8))

        def __len__(self):
            return 1

    monkeypatch.setattr(ms, "DataStream", _Stream)

    seen = {}

    class _Stop(Exception):
        pass

    class _Trainer:
        def train(self, state, stream, **kwargs):
            seen.update(kwargs)
            raise _Stop

    monkeypatch.setattr(
        ms, "prepare_trainer", lambda cfg, rank, schedule: (_Trainer(), _State(), None)
    )

    class _State:
        def detach_(self):
            return self

    cfg = MetasmoothnessConfig(
        run_path="unused",
        batch_size=8,
        seed=0,
        grad_accum_steps=4,
        max_grad_norm=1.0,
    )
    with pytest.raises(_Stop):
        ms.metasmoothness_worker(0, 0, 1, ds, 8, cfg)

    assert seen.get("grad_accum_steps") == 4
    assert seen.get("max_grad_norm") == 1.0


def test_combine_ms_thetas_matches_in_process_score(tmp_path, monkeypatch):
    """Shards written by three ``BERGSON_MS_K`` runs combine to the score the
    whole vectors give, whether read from one run_path or one per training."""
    import json
    import sys

    from bergson.magic import combine_ms_thetas
    from bergson.magic.metasmoothness import metasmoothness_score

    torch.manual_seed(0)
    thetas = [torch.randn(40) for _ in range(3)]
    world = 4
    for k, theta in enumerate(thetas):
        shard_dir = tmp_path / f"k{k}" / "theta_shards"
        shard_dir.mkdir(parents=True)
        for rank, piece in enumerate(theta.chunk(world)):
            torch.save(
                {
                    "k": k,
                    "rank": rank,
                    "world_size": world,
                    "sharded": True,
                    "theta": piece,
                    "layout": [("layers.0.q_proj.lora_A.weight", piece.numel())],
                },
                shard_dir / f"theta_k{k}_rank{rank}.pt",
            )

    paths = [str(tmp_path / f"k{k}") for k in range(3)]
    monkeypatch.setattr(sys, "argv", ["combine", *paths, "--by-param"])
    combine_ms_thetas.main()
    result = json.loads((tmp_path / "k0" / "metasmoothness.json").read_text())

    assert result["score"] == pytest.approx(metasmoothness_score(*thetas), abs=1e-6)
    assert result["by_param"]["matrix A"]["movement_share"] == pytest.approx(1.0)

    # A missing rank must fail rather than score part of the parameters.
    (tmp_path / "k1" / "theta_shards" / "theta_k1_rank2.pt").unlink()
    with pytest.raises(ValueError, match="missing shards"):
        combine_ms_thetas.main()
