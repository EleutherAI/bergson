"""Tests for the per-step score-magnitude plot a MAGIC run writes."""

from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

from bergson.magic.cli import save_magic_scores
from bergson.magic.score_trajectory import per_step_level


def test_per_step_level():
    """A step whose |scores| are all 10**L has level L, over any trailing axes."""
    bs, seq = 4, 3
    levels = [-3.0, -1.0, 0.0, 2.0, 4.0]
    scores = np.concatenate([np.full((bs, seq), 10.0**lv) for lv in levels])

    got = per_step_level(scores, bs)
    assert got.shape == (len(levels),)
    np.testing.assert_allclose(got, levels, atol=1e-9)

    # A step with no score at all -- a MAGIC backward leaves early steps
    # undefined -- has no level, rather than a level of zero.
    scores[:bs] = 0.0
    got = per_step_level(scores, bs)
    assert np.isnan(got[0])
    np.testing.assert_allclose(got[1:], levels[1:], atol=1e-9)


def test_saving_scores_writes_the_plot(tmp_path):
    """Saving a run's scores writes score_vs_step.png beside them."""
    bs, n_steps, dead_steps = 4, 20, 5
    rng = np.random.default_rng(0)
    # Level ramps down over training, like a real run.
    scores = torch.from_numpy(
        np.concatenate(
            [
                10.0 ** rng.uniform(3 - s, 4 - s, size=bs)
                for s in np.linspace(0, 6, n_steps)
            ]
        ).astype(np.float32)[:, None]
    )
    scores[: dead_steps * bs] = 0.0

    save_magic_scores(
        str(tmp_path),
        scores,
        Dataset.from_dict({"length": [1] * len(scores)}),
        pad_count=0,
        per_token=False,
        batch_size=bs,
    )

    out = Path(tmp_path) / "score_vs_step.png"
    assert out.exists() and out.stat().st_size > 0
