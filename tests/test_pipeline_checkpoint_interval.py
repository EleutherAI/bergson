"""trak and trackstar fit their Hessians with the pipeline's
``checkpoint_interval``, and ``resume`` leaves an interrupted fit's saves."""

import pytest

import bergson.cli.trackstar as trackstar_module
import bergson.cli.trak as trak_module
from bergson.config import TrackstarConfig, TrakConfig
from bergson.config.config import TrackstarIndexConfig


class Stop(Exception):
    """Ends the pipeline at its first Hessian fit."""


def _run(pipeline: str, run_path, **pipeline_kwargs):
    if pipeline == "trak":
        index_cfg = TrackstarIndexConfig(
            run_path=str(run_path), loss_fn="log_odds", projection_target="global"
        )
        trak_module.trak(index_cfg, TrakConfig(**pipeline_kwargs))
    else:
        index_cfg = TrackstarIndexConfig(run_path=str(run_path))
        trackstar_module.trackstar(index_cfg, TrackstarConfig(**pipeline_kwargs))


@pytest.mark.parametrize(
    "pipeline, hessian", [("trak", "train_hessian"), ("trackstar", "value_hessian")]
)
def test_pipeline_resumes_its_hessian_fit(tmp_path, monkeypatch, pipeline, hessian):
    module = trak_module if pipeline == "trak" else trackstar_module
    save = tmp_path / f"{hessian}.part" / "checkpoint" / "autocorrelation" / "rank_0.pt"
    fits = []

    def fit(index_cfg, hessian_cfg):
        fits.append((hessian_cfg.checkpoint_interval, save.exists()))
        raise Stop

    monkeypatch.setattr(module, "approximate_hessians", fit)

    with pytest.raises(Stop):
        _run(pipeline, tmp_path)
    # An interrupted fit left a save behind.
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text("x")
    with pytest.raises(Stop):
        _run(pipeline, tmp_path, checkpoint_interval=5, resume=True)

    assert fits == [(0, False), (5, True)]
