"""Tests for ``bergson score_trajectory`` (per-step score-magnitude plot)."""

from pathlib import Path

import numpy as np
import yaml

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


def _make_run(tmp_path: Path, bs: int, n_steps: int) -> Path:
    run = tmp_path / "magic_run"
    run.mkdir()
    rng = np.random.default_rng(0)
    # Level ramps down over training, like a real run.
    scores = np.concatenate(
        [10.0 ** rng.uniform(3 - s, 4 - s, size=bs) for s in np.linspace(0, 6, n_steps)]
    ).astype(np.float32)[:, None]
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


def test_plot_score_trajectory_window_and_custom_out(tmp_path):
    run = _make_run(tmp_path, bs=4, n_steps=20)
    out = tmp_path / "custom.png"
    got = plot_score_trajectory(str(run), window=5, out=str(out))
    assert got == str(out)
    assert out.exists() and out.stat().st_size > 0
