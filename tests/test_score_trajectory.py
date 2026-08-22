"""Tests for ``bergson score_trajectory`` (per-step score-magnitude plot)."""

import numpy as np
import yaml

from bergson.__main__ import Main
from bergson.config.config_io import parse_steps
from bergson.magic.score_trajectory import per_step_level
from bergson.score.score_writer import save_sequence_scores


def test_per_step_level():
    """A step whose |scores| are all 10**L has level L, over any trailing axes."""
    bs, seq = 4, 3
    levels = [-3.0, -1.0, 0.0, 2.0, 4.0]
    scores = np.concatenate(
        [np.full((bs, seq), 10.0**lv) for lv in levels]
    )  # (bs * len(levels), seq)

    got = per_step_level(scores, bs)
    assert got.shape == (len(levels),)
    np.testing.assert_allclose(got, levels, atol=1e-9)

    # A step with no score at all -- a MAGIC backward leaves early steps
    # undefined -- has no level, rather than a level of zero.
    scores[:bs] = 0.0
    got = per_step_level(scores, bs)
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1:], levels[1:], atol=1e-9)


def test_score_trajectory_plots_a_pipeline_run(tmp_path):
    """A pipeline-shaped run plots end to end, from a yaml step to the PNG."""
    bs, n_steps, dead_steps = 4, 20, 5
    run = tmp_path / "magic_run"
    run.mkdir()
    rng = np.random.default_rng(0)
    # Level ramps down over training, like a real run.
    scores = np.concatenate(
        [10.0 ** rng.uniform(3 - s, 4 - s, size=bs) for s in np.linspace(0, 6, n_steps)]
    ).astype(np.float32)[:, None]
    scores[: dead_steps * bs] = 0.0
    save_sequence_scores(run / "scores", scores)
    # A pipeline writes every step to the run's config, so the magic step that
    # names the batch_size need not be the first one.
    cfg = {"steps": [{"build": {"model": "x"}}, {"magic": {"batch_size": bs}}]}
    with open(run / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    registry = {
        cls.__name__.lower(): cls
        for cls in Main.__dataclass_fields__["command"].type.__args__
    }
    ((name, cmd),) = parse_steps(
        [{"score_trajectory": {"run_path": str(run)}}], registry
    )
    assert name == "score_trajectory"
    cmd.execute()

    out = run / "score_vs_step.png"
    assert out.exists() and out.stat().st_size > 0
