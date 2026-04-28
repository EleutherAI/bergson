"""Unit tests for the multi-step YAML pipeline runner.

These tests exercise parsing only — they never call `.execute()`, so they
need no GPU and no model downloads.
"""

from pathlib import Path
from typing import get_args

import pytest

from bergson.__main__ import Apply_Hessian, Build, Hessian, Main, Score
from bergson.config import HessianConfig, IndexConfig, PreprocessConfig
from bergson.hessians.apply_hessian import EkfacConfig
from bergson.yaml_pipeline import parse_pipeline

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "pipelines"


@pytest.fixture
def registry() -> dict[str, type]:
    classes = get_args(Main.__dataclass_fields__["command"].type)
    return {cls.__name__.lower(): cls for cls in classes}


def write(tmp_path, body: str) -> str:
    path = tmp_path / "pipeline.yaml"
    path.write_text(body)
    return str(path)


def test_parse_pipeline_hydrates_steps_in_order(tmp_path, registry):
    """A valid two-step pipeline produces typed commands with the right configs."""
    yaml_path = write(
        tmp_path,
        """
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

    steps = parse_pipeline(yaml_path, registry)

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


def test_command_name_is_case_insensitive(tmp_path, registry):
    yaml_path = write(
        tmp_path,
        """
- Hessian:
    hessian_cfg: {method: kfac}
    index_cfg: {run_path: runs/test}
""",
    )
    steps = parse_pipeline(yaml_path, registry)
    assert isinstance(steps[0][1], Hessian)


def test_top_level_must_be_a_list(tmp_path, registry):
    yaml_path = write(
        tmp_path,
        """
hessian:
  hessian_cfg: {method: kfac}
""",
    )
    with pytest.raises(ValueError, match="must be a list"):
        parse_pipeline(yaml_path, registry)


def test_step_must_be_single_key_mapping(tmp_path, registry):
    yaml_path = write(
        tmp_path,
        """
- hessian:
    hessian_cfg: {method: kfac}
    index_cfg: {run_path: runs/test}
  build:
    index_cfg: {run_path: runs/test}
    preprocess_cfg: {}
""",
    )
    with pytest.raises(ValueError, match="single-key mapping"):
        parse_pipeline(yaml_path, registry)


def test_unknown_command_raises(tmp_path, registry):
    yaml_path = write(
        tmp_path,
        """
- not_a_real_command:
    foo: bar
""",
    )
    with pytest.raises(ValueError, match="unknown command 'not_a_real_command'"):
        parse_pipeline(yaml_path, registry)


def test_build_ekfac_example_parses(registry):
    """`build_EKFAC.yaml` must parse into a single hessian step."""
    steps = parse_pipeline(str(EXAMPLES / "build_EKFAC.yaml"), registry)
    assert len(steps) == 1
    name, cmd = steps[0]
    assert name == "hessian"
    assert isinstance(cmd, Hessian)
    assert cmd.hessian_cfg.method == "kfac"
    assert cmd.hessian_cfg.ev_correction is True


def test_query_ekfac_example_parses(registry):
    """`query_EKFAC.yaml` must parse build → apply_hessian → score."""
    steps = parse_pipeline(str(EXAMPLES / "query_EKFAC.yaml"), registry)
    assert [name for name, _ in steps] == ["build", "apply_hessian", "score"]

    _, build_cmd = steps[0]
    assert isinstance(build_cmd, Build)
    assert build_cmd.preprocess_cfg.aggregation == "mean"

    _, apply_cmd = steps[1]
    assert isinstance(apply_cmd, Apply_Hessian)
    assert isinstance(apply_cmd.ekfac_cfg, EkfacConfig)
    assert apply_cmd.ekfac_cfg.hessian_method_path == "runs/ekfac/hessian/kfac"

    _, score_cmd = steps[2]
    assert isinstance(score_cmd, Score)
    # The score step's query_path must point at apply_hessian's output.
    assert score_cmd.score_cfg.query_path == apply_cmd.ekfac_cfg.run_path


def test_empty_step_body_uses_dataclass_defaults(tmp_path, registry):
    """An entry with no body should still hydrate."""
    yaml_path = write(
        tmp_path,
        """
- hessian:
    hessian_cfg: {}
    index_cfg: {run_path: runs/test}
""",
    )
    steps = parse_pipeline(yaml_path, registry)
    _, cmd = steps[0]
    assert isinstance(cmd, Hessian)
    # `method` has a default on HessianConfig — round-trip leaves it intact.
    assert cmd.hessian_cfg.method == HessianConfig().method
