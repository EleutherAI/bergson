"""Method-specific configs through the real CLI and YAML entry points."""

from typing import get_args

import pytest
import yaml
from simple_parsing import ArgumentParser, ConflictResolution

from bergson.__main__ import Main
from bergson.cli.commands import Validate
from bergson.config.config_io import parse_steps, read_config, save_run_config
from bergson.config.validation import (
    ControlBank,
    FilterConfig,
    LDSConfig,
    NoControls,
    RandomControls,
    RandomSubsets,
    SubsetBank,
    WeightStepConfig,
)

registry = {
    cls.__name__.lower(): cls
    for cls in get_args(Main.__dataclass_fields__["command"].type)
}


def parse_cli(args):
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    return parser.parse_args(args).prog.command


@pytest.mark.parametrize(
    "flags,method_type,source_type,count",
    [
        ([], LDSConfig, RandomSubsets, 100),
        (["--method", "filter"], FilterConfig, RandomControls, 3),
        (["--method", "filter", "--count", "1"], FilterConfig, RandomControls, 1),
        (["--method", "filter", "--controls", "skip"], FilterConfig, NoControls, None),
        (
            [
                "--method",
                "filter",
                "--controls",
                "bank",
                "--paths",
                "/bank",
                "--count",
                "2",
            ],
            FilterConfig,
            ControlBank,
            2,
        ),
        (
            ["--method", "lds", "--subsets", "bank", "--paths", "/bank"],
            LDSConfig,
            SubsetBank,
            None,
        ),
    ],
)
def test_cli_and_saved_config_roundtrip(
    flags, method_type, source_type, count, tmp_path
):
    cfg = parse_cli(["validate", "runs/probe", *flags])
    assert type(cfg.method) is method_type
    source = cfg.method.controls if method_type is FilterConfig else cfg.method.subsets
    assert type(source) is source_type
    if count is not None:
        assert source.count == count
    save_run_config(cfg, tmp_path)
    restored = parse_steps(read_config(tmp_path)["steps"], registry)[0][1]
    assert restored == cfg
    assert type(restored.method) is method_type


def test_yaml_pipeline_and_matrix():
    steps = yaml.safe_load("""
- validate:
    run_path: runs/control_{n}
    matrix: {n: [1, 2, 3]}
    method:
      kind: filter
      fraction: 0.05
      controls: {source: random, count: "{n}"}
""")
    configs = [cfg for _, cfg in parse_steps(steps, registry)]
    assert [cfg.method.controls.count for cfg in configs] == [1, 2, 3]
    assert all(cfg.method.fraction == 0.05 for cfg in configs)


@pytest.mark.parametrize(
    "flags",
    [
        ["--method", "lds", "--controls", "skip"],
        ["--method", "filter", "--subsets", "random"],
        ["--method", "filter", "--controls", "skip", "--count", "3"],
        [
            "--method",
            "filter",
            "--controls",
            "bank",
            "--paths",
            "/bank",
            "--sampling_seed",
            "1",
        ],
    ],
)
def test_cli_rejects_inactive_options(flags):
    with pytest.raises(SystemExit) as exc:
        parse_cli(["validate", "runs/probe", *flags])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "method",
    [
        {"kind": "lds", "controls": {"source": "skip"}},
        {"kind": "filter", "subsets": {"source": "random"}},
        {"kind": "filter", "controls": {"source": "skip", "count": 3}},
        {"kind": "filter", "controls": {"source": "random", "fraction": 0.1}},
        {"kind": "unknown"},
        {"controls": {"source": "random"}},
    ],
)
def test_yaml_rejects_inactive_or_untagged_options(method):
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        Validate.from_dict(
            {"run_path": "runs/probe", "method": method}, drop_extra_fields=False
        )


@pytest.mark.parametrize(
    "method,expected,absent",
    [
        ("filter", "--controls", "--subsets"),
        ("lds", "--subsets", "--controls"),
    ],
)
def test_method_specific_help(method, expected, absent, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_cli(["validate", "--method", method, "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert expected in output
    assert absent not in output


@pytest.mark.parametrize(
    "legacy,expected",
    [
        ({"seed": 9}, LDSConfig(subsets=RandomSubsets(sampling_seed=9))),
        ({"num_subsets": 7}, LDSConfig(subsets=RandomSubsets(count=7))),
        (
            {"num_subsets": 7, "subset_fraction": 0.2, "seed": 9},
            LDSConfig(
                subsets=RandomSubsets(
                    count=7, sampling="random", fraction=0.2, sampling_seed=9
                )
            ),
        ),
        (
            {"retrained_dir": "a,b", "subset_start": 1, "subset_stop": 3},
            LDSConfig(subsets=SubsetBank(paths=["a", "b"]), start=1, stop=3),
        ),
        (
            {"method": "filter-proponents", "num_subsets": 3},
            FilterConfig(fraction=1 / 3, controls=RandomControls(count=3)),
        ),
        (
            {"method": "filter-detractors", "num_subsets": 0, "subset_fraction": 0.1},
            FilterConfig(direction="detractors", fraction=0.1, controls=NoControls()),
        ),
        (
            {"method": "filter-proponents", "retrained_dir": ["a"], "num_subsets": 2},
            FilterConfig(fraction=0.5, controls=ControlBank(paths=["a"], count=None)),
        ),
        (
            {"method": "filter-proponents", "retrained_dir": "a", "controls": "skip"},
            FilterConfig(fraction=0.01, controls=NoControls()),
        ),
        (
            {
                "method": "filter-proponents",
                "retrained_dir": "a",
                "controls": "retrain",
            },
            FilterConfig(fraction=0.01, controls=RandomControls(count=100)),
        ),
        ({"weight_lrs": [0.1, 0.2]}, WeightStepConfig(lrs=[0.1, 0.2])),
    ],
)
def test_legacy_yaml_preserves_semantics_and_saves_nested(legacy, expected):
    with pytest.warns(FutureWarning):
        cfg = Validate.from_dict({"run_path": "runs/probe", **legacy})
    assert cfg.method == expected
    payload = cfg.to_dict()
    assert isinstance(payload["method"], dict)
    assert "num_subsets" not in payload
    assert Validate.from_dict(payload, drop_extra_fields=False) == cfg


@pytest.mark.parametrize(
    "method",
    [
        {"kind": "filter", "fraction": 0},
        {"kind": "filter", "fraction": 1.1},
        {"kind": "filter", "controls": {"source": "random", "count": 0}},
        {"kind": "filter", "controls": {"source": "bank", "paths": []}},
        {"kind": "lds", "subsets": {"source": "random", "count": 0}},
        {"kind": "lds", "start": 2, "stop": 1},
        {"kind": "weight-step", "lrs": []},
    ],
)
def test_invalid_experiments_fail_during_parsing(method):
    with pytest.raises(ValueError):
        Validate.from_dict({"run_path": "runs/probe", "method": method})


def test_mixed_old_and_new_fields_rejected():
    with pytest.raises(ValueError, match="Cannot mix"):
        Validate.from_dict({"method": {"kind": "filter"}, "num_subsets": 3})


def test_weight_step_cli_and_roundtrip():
    cfg = parse_cli(
        ["validate", "runs/probe", "--method", "weight-step", "--lrs", "0.1", "0.2"]
    )
    assert cfg.method == WeightStepConfig(lrs=[0.1, 0.2])
    assert Validate.from_dict(cfg.to_dict()) == cfg


def test_sampling_seed_is_independent_of_training_seed():
    cfg = parse_cli(
        [
            "validate",
            "runs/probe",
            "--method",
            "filter",
            "--seed",
            "9",
            "--sampling_seed",
            "4",
        ]
    )
    assert cfg.seed == 9
    assert cfg.method.controls.sampling_seed == 4


@pytest.mark.parametrize("source", ["random", "skip", "bank"])
def test_filter_dispatch_always_runs_ranked_retraining(source, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "bergson.cli.commands.run_magic", lambda *a, **kw: calls.append(kw)
    )
    monkeypatch.setattr(
        "bergson.cli.commands.evaluate_retrained",
        lambda *a, **kw: pytest.fail("LDS dispatch"),
    )
    controls = {"source": source}
    if source == "bank":
        controls["paths"] = ["bank"]
    Validate.from_dict(
        {
            "run_path": "runs/probe",
            "scores": "s",
            "method": {
                "kind": "filter",
                "controls": controls,
            },
        }
    ).execute()
    assert len(calls) == 1 and calls[0]["validate"]
