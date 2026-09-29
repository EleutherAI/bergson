"""Tests for the per-step score plot a MAGIC run writes beside its scores."""

import numpy as np

from bergson.magic.score_plot import per_step_level, plot_score_trajectory


def test_plot_score_trajectory(tmp_path):
    """Each step's level is the level of its scores, and the plot is written."""
    batch_size = 4
    levels = [-3.0, -1.0, 0.0, 2.0]
    scores = np.concatenate([np.full((batch_size, 3), 10.0**lv) for lv in levels])
    scores[:batch_size] = 0.0  # a step the backward left with no score

    level = per_step_level(scores, batch_size)
    assert np.isnan(level[0])
    np.testing.assert_allclose(level[1:], levels[1:], atol=1e-9)

    out = plot_score_trajectory(scores, batch_size, tmp_path / "score_vs_step.png")
    assert out is not None and out.stat().st_size > 0
