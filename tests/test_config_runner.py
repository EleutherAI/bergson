"""Unit tests for the YAML config parser.

These tests exercise parsing only — they never call `.execute()`, so they
need no GPU and no model downloads.
"""

import sys
from typing import get_args

import pytest

from bergson.__main__ import Build, Hessian, Main, main
from bergson.config.config import HessianConfig, IndexConfig, PreprocessConfig
from bergson.config.config_io import (
    load_subconfig,
    parse_steps,
    read_config,
    save_pipeline_config,
)


@pytest.fixture
def registry() -> dict[str, type]:
    classes = get_args(Main.__dataclass_fields__["command"].type)
    return {cls.__name__.lower(): cls for cls in classes}


def write(tmp_path, body: str) -> str:
    path = tmp_path / "pipeline.yaml"
    path.write_text(body)
    return str(path)


def parse(yaml_path, registry):
    return parse_steps(read_config(yaml_path)["steps"], registry)


def test_parse_pipeline_hydrates_steps_in_order(tmp_path, registry):
    """A valid two-step pipeline produces typed commands with the right configs."""
    yaml_path = write(
        tmp_path,
        """
steps:
  - hessian:
      hessian_cfg:
        method: tkfac
      index_cfg:
        run_path: runs/test
        model: gpt2
  - build:
      index_cfg:
        run_path: runs/test
        model: gpt2
      preprocess_cfg: {}
""",
    )

    steps = parse(yaml_path, registry)

    assert [name for name, _ in steps] == ["hessian", "build"]

    _, hessian_cmd = steps[0]
    assert isinstance(hessian_cmd, Hessian)
    assert isinstance(hessian_cmd.hessian_cfg, HessianConfig)
    assert hessian_cmd.hessian_cfg.method == "tkfac"
    assert isinstance(hessian_cmd.index_cfg, IndexConfig)
    assert hessian_cmd.index_cfg.run_path == "runs/test"
    assert hessian_cmd.index_cfg.model == "gpt2"

    _, build_cmd = steps[1]
    assert isinstance(build_cmd, Build)
    assert isinstance(build_cmd.index_cfg, IndexConfig)
    assert build_cmd.index_cfg.run_path == "runs/test"
    assert isinstance(build_cmd.preprocess_cfg, PreprocessConfig)


def test_single_step_is_one_command(tmp_path, registry):
    """A one-entry `steps:` list is a single command."""
    yaml_path = write(
        tmp_path,
        """
steps:
  - hessian:
      hessian_cfg: {method: kfac}
      index_cfg: {run_path: runs/test}
""",
    )
    steps = parse(yaml_path, registry)
    assert [name for name, _ in steps] == ["hessian"]
    assert isinstance(steps[0][1], Hessian)


def test_run_path_is_read_from_top_level(tmp_path):
    yaml_path = write(
        tmp_path,
        """
run_path: runs/my_pipeline
steps:
  - hessian: {hessian_cfg: {method: kfac}, index_cfg: {run_path: runs/test}}
  - build: {index_cfg: {run_path: runs/test}, preprocess_cfg: {}}
""",
    )
    doc = read_config(yaml_path)
    assert doc["run_path"] == "runs/my_pipeline"
    assert len(doc["steps"]) == 2


def test_command_name_is_case_insensitive(tmp_path, registry):
    yaml_path = write(
        tmp_path,
        """
steps:
  - Hessian:
      hessian_cfg: {method: kfac}
      index_cfg: {run_path: runs/test}
""",
    )
    steps = parse(yaml_path, registry)
    assert isinstance(steps[0][1], Hessian)


def test_missing_steps_key_raises(tmp_path):
    """A config without a top-level `steps:` list is rejected."""
    yaml_path = write(
        tmp_path,
        """
hessian:
  hessian_cfg: {method: kfac}
""",
    )
    with pytest.raises(ValueError, match="must be a mapping with a `steps:` list"):
        read_config(yaml_path)


def test_invalid_top_level_type_raises(tmp_path):
    yaml_path = write(tmp_path, "just a string\n")
    with pytest.raises(ValueError, match="must be a mapping with a `steps:` list"):
        read_config(yaml_path)


def test_step_must_be_single_key_mapping(tmp_path, registry):
    yaml_path = write(
        tmp_path,
        """
steps:
  - hessian:
      hessian_cfg: {method: kfac}
      index_cfg: {run_path: runs/test}
    build:
      index_cfg: {run_path: runs/test}
      preprocess_cfg: {}
""",
    )
    with pytest.raises(ValueError, match="single command-key mapping"):
        parse(yaml_path, registry)


def test_unknown_command_raises(tmp_path, registry):
    yaml_path = write(
        tmp_path,
        """
steps:
  - not_a_real_command:
      foo: bar
""",
    )
    with pytest.raises(ValueError, match="Unknown command 'not_a_real_command'"):
        parse(yaml_path, registry)


def test_omitted_fields_use_dataclass_defaults(tmp_path, registry):
    """Fields omitted from a step body fall back to their dataclass defaults."""
    yaml_path = write(
        tmp_path,
        """
steps:
  - hessian:
      hessian_cfg: {method: kfac}
      index_cfg: {run_path: runs/test}
""",
    )
    steps = parse(yaml_path, registry)
    _, cmd = steps[0]
    assert isinstance(cmd, Hessian)
    assert cmd.hessian_cfg.method == "kfac"
    # `method` is required, but the remaining fields fall back to defaults.
    assert cmd.hessian_cfg.ev_correction == HessianConfig(method="kfac").ev_correction


def test_build_step_is_index_only(tmp_path, registry):
    """A build step builds the index only; Hessians are fit via the hessian step."""
    yaml_path = write(
        tmp_path,
        """
steps:
  - build:
      index_cfg: {run_path: runs/test}
      preprocess_cfg: {}
""",
    )
    steps = parse(yaml_path, registry)
    _, cmd = steps[0]
    assert isinstance(cmd, Build)


def make_steps() -> list[tuple[str, object]]:
    """A two-step pipeline; only the build step carries a preprocess_cfg."""
    return [
        (
            "hessian",
            Hessian(
                hessian_cfg=HessianConfig(method="kfac"),
                index_cfg=IndexConfig(run_path="runs/test"),
            ),
        ),
        (
            "build",
            Build(
                index_cfg=IndexConfig(run_path="runs/test"),
                preprocess_cfg=PreprocessConfig(),
            ),
        ),
    ]


def test_save_pipeline_config_writes_replayable_manifest(tmp_path):
    """(name, command) steps are resolved to dicts with run_path recorded."""
    run_path = tmp_path / "pipeline"
    save_pipeline_config(make_steps(), run_path)

    doc = read_config(run_path)
    assert doc["run_path"] == str(run_path)
    assert [list(step) for step in doc["steps"]] == [["hessian"], ["build"]]
    assert doc["steps"][0]["hessian"]["hessian_cfg"]["method"] == "kfac"


def test_save_pipeline_config_generates_run_path_when_missing(tmp_path, monkeypatch):
    """No run_path falls back to an auto-named directory under runs/."""
    monkeypatch.chdir(tmp_path)
    save_pipeline_config(make_steps(), None)

    configs = list((tmp_path / "runs").glob("*/config.yaml"))
    assert len(configs) == 1
    assert read_config(configs[0])["run_path"].startswith("runs/")


def test_load_subconfig_searches_all_steps(tmp_path):
    """The field is found in a later step; absent field or path gives None."""
    save_pipeline_config(make_steps(), tmp_path / "pipeline")

    sub = load_subconfig(tmp_path / "pipeline", "preprocess_cfg", PreprocessConfig)
    assert isinstance(sub, PreprocessConfig)

    assert load_subconfig(tmp_path / "pipeline", "score_cfg", PreprocessConfig) is None
    assert load_subconfig(tmp_path / "missing", "index_cfg", IndexConfig) is None


def test_hessian_step_fits_autocorrelation(tmp_path, registry):
    """Autocorrelation is fit via the hessian step, like every other method."""
    yaml_path = write(
        tmp_path,
        """
steps:
  - hessian:
      index_cfg: {run_path: runs/test}
      hessian_cfg: {method: autocorrelation}
""",
    )
    steps = parse(yaml_path, registry)
    _, cmd = steps[0]
    assert isinstance(cmd, Hessian)
    assert cmd.hessian_cfg.method == "autocorrelation"


def test_metadata_records_runtime_nccl_version():
    """Run metadata carries the NCCL version torch actually loaded.

    NCCL selects the all-reduce algorithms, so its version is part of what
    makes a multi-GPU run bit-reproducible; the runtime report can disagree
    with pip metadata, which is why the loaded version is the one recorded.
    """
    import torch
    import torch.cuda.nccl

    from bergson.config.config_io import make_metadata

    meta = make_metadata()
    if torch.distributed.is_nccl_available():
        parts = meta["nccl_version"].split(".")
        assert len(parts) >= 2 and all(p.isdigit() for p in parts)
        assert tuple(map(int, parts)) == tuple(torch.cuda.nccl.version())
    else:
        assert "nccl_version" not in meta


def test_missing_config_path_reports_the_path(tmp_path, monkeypatch, capsys):
    """A config path that does not exist names itself, rather than argparse's
    ``invalid choice``, which reads as a bad subcommand and hides the cause."""
    missing = str(tmp_path / "run" / "tune.yaml")
    monkeypatch.setattr(sys, "argv", ["bergson", missing])

    with pytest.raises(SystemExit) as excinfo:
        main()

    message = str(excinfo.value)
    assert missing in message, message
    assert "invalid choice" not in message, message


def test_bare_unknown_command_still_goes_to_argparse(monkeypatch):
    """A bare word is a subcommand, so an unknown one keeps argparse's error."""
    monkeypatch.setattr(sys, "argv", ["bergson", "definitely-not-a-command"])

    with pytest.raises(SystemExit) as excinfo:
        main()

    # argparse exits 2 on a usage error; the config branch raises with a message.
    assert excinfo.value.code == 2
