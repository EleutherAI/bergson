"""CLI/YAML identity and compatibility for method-specific validation configs."""

import pytest
from simple_parsing import ArgumentParser, ConflictResolution

from bergson.__main__ import Main
from bergson.cli.commands import Validate
from bergson.config.config_io import parse_steps, read_config, save_run_config
from bergson.config.validation import FilterConfig, RandomControls


def test_filter_cli_survives_yaml_roundtrip(tmp_path):
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    cfg = parser.parse_args(
        ["validate", str(tmp_path), "--method", "filter", "--count", "1"]
    ).prog.command
    assert cfg.method == FilterConfig(fraction=0.05, controls=RandomControls(count=1))

    save_run_config(cfg, tmp_path)
    restored = parse_steps(read_config(tmp_path)["steps"], {"validate": Validate})[0][1]
    assert restored == cfg
    assert isinstance(restored.method, FilterConfig)


def test_legacy_filter_preserves_removal_size_and_sampling_seed():
    with pytest.warns(FutureWarning):
        cfg = Validate.from_dict(
            {
                "run_path": "runs/filter",
                "method": "filter-proponents",
                "num_subsets": 3,
                "seed": 9,
            }
        )
    assert cfg.method == FilterConfig(
        fraction=1 / 3, controls=RandomControls(count=3, sampling_seed=9)
    )
    payload = cfg.to_dict()
    assert payload["method"]["kind"] == "filter"
    assert "num_subsets" not in payload
    assert Validate.from_dict(payload, drop_extra_fields=False) == cfg
