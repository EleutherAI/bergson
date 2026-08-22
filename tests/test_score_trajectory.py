"""Tests for ``bergson score_trajectory`` (per-step score-magnitude plot)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from bergson.__main__ import Main
from bergson.config.config_io import parse_steps
from bergson.magic.score_trajectory import (
    batch_size_from_config,
    per_step_level,
    plot_score_trajectory,
)
from bergson.score.score_writer import save_sequence_scores


def test_per_step_level_recovers_known_levels():
    """A step whose |scores| are all 10**L has level L; all-zero steps are NaN."""
    bs = 4
    levels = [-3.0, -1.0, 0.0, 2.0, 4.0]
    rows = []
    for lv in levels:
        rows.extend([10.0**lv] * bs)
    scores = np.array(rows, dtype=np.float64)[:, None]

    got = per_step_level(scores, bs)
    assert got.shape == (len(levels),)
    np.testing.assert_allclose(got, levels, atol=1e-9)

    # An all-zero step has no defined level.
    scores[:bs] = 0.0
    got = per_step_level(scores, bs)
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1:], levels[1:], atol=1e-9)


def test_per_step_level_pools_trailing_axes():
    """Trailing axes (token / query) are pooled into the step, not kept."""
    bs, seq = 2, 3
    scores = np.full((bs * 4, seq), 10.0**1.5, dtype=np.float64)
    got = per_step_level(scores, bs)
    assert got.shape == (4,)
    np.testing.assert_allclose(got, 1.5, atol=1e-9)


def _make_run(tmp_path: Path, bs: int, n_steps: int, dead_steps: int = 0) -> Path:
    run = tmp_path / "magic_run"
    run.mkdir()
    rng = np.random.default_rng(0)
    # Level ramps down over training, like a real run.
    scores = np.concatenate(
        [10.0 ** rng.uniform(3 - s, 4 - s, size=bs) for s in np.linspace(0, 6, n_steps)]
    ).astype(np.float32)[:, None]
    # A MAGIC backward can leave the leading steps with no score at all.
    scores[: dead_steps * bs] = 0.0
    save_sequence_scores(run / "scores", scores)
    cfg = {"steps": [{"magic": {"batch_size": bs}}], "metadata": {}}
    with open(run / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    return run


def test_plot_score_trajectory_writes_png(tmp_path):
    run = _make_run(tmp_path, bs=4, n_steps=20)
    assert batch_size_from_config(str(run)) == 4

    out = plot_score_trajectory(str(run))
    assert out == str(run / "score_vs_step.png")
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_plot_score_trajectory_custom_out(tmp_path):
    run = _make_run(tmp_path, bs=4, n_steps=20)
    out = tmp_path / "custom.png"
    got = plot_score_trajectory(str(run), out=str(out))
    assert got == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_score_trajectory_with_leading_dead_steps(tmp_path):
    """Leading all-NaN steps get marked rather than plotted as absence."""
    run = _make_run(tmp_path, bs=4, n_steps=20, dead_steps=5)
    out = plot_score_trajectory(str(run), out=str(tmp_path / "dead.png"))
    assert Path(out).exists() and Path(out).stat().st_size > 0


def test_plot_score_trajectory_all_dead(tmp_path):
    run = _make_run(tmp_path, bs=4, n_steps=20, dead_steps=20)
    with pytest.raises(ValueError, match="all zero / non-finite"):
        plot_score_trajectory(str(run), out=str(tmp_path / "nope.png"))


def test_batch_size_from_pipeline_config(tmp_path):
    """A pipeline run dir holds every step, and the magic step is the one to read."""
    run = tmp_path / "pipeline_run"
    run.mkdir()
    cfg = {
        "run_path": str(run),
        "steps": [{"build": {"model": "x"}}, {"magic": {"batch_size": 8}}],
        "metadata": {},
    }
    with open(run / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    assert batch_size_from_config(str(run)) == 8


def test_batch_size_from_config_without_batch_size(tmp_path):
    run = tmp_path / "no_bs"
    run.mkdir()
    with open(run / "config.yaml", "w") as f:
        yaml.safe_dump({"steps": [{"build": {"model": "x"}}], "metadata": {}}, f)
    with pytest.raises(ValueError, match="batch_size"):
        batch_size_from_config(str(run))


def test_score_trajectory_step_parses_from_yaml(tmp_path):
    """`bergson <config.yaml>` builds every command through from_dict."""
    registry = {
        cls.__name__.lower(): cls
        for cls in Main.__dataclass_fields__["command"].type.__args__
    }
    steps = parse_steps([{"score_trajectory": {"run_path": str(tmp_path)}}], registry)
    ((name, cmd),) = steps
    assert name == "score_trajectory"
    assert cmd.run_path == str(tmp_path)
