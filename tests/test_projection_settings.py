"""Projected gradients and Hessians are saved with the settings they were
projected with, so runs that reuse them can check that they match."""

import pytest
import torch
import yaml

from bergson import GradientProcessor
from bergson.config import IndexConfig, PreprocessConfig, ScoreConfig
from bergson.process_grads import mix_autocorrelation_matrices
from bergson.score.score import score_dataset

CHANGED = {
    "projection_dim": 8,
    "projection_type": "normal",
    "projection_scale": "row_norm",
    "projection_seed": 3,
    "projection_target": "global",
    "include_bias": True,
}


def test_matching_projections_pass():
    GradientProcessor(projection_dim=16).check_projection_matches(
        GradientProcessor(projection_dim=16), "The query"
    )
    GradientProcessor(projection_dim=None).check_projection_matches(
        GradientProcessor(projection_dim=0), "The query"
    )


@pytest.mark.parametrize("name", ["projection_seed", "include_bias"])
def test_each_projection_setting_must_match(name):
    other = GradientProcessor(**{"projection_dim": 16, name: CHANGED[name]})
    with pytest.raises(ValueError, match=f"{name}={CHANGED[name]!r}"):
        GradientProcessor(projection_dim=16).check_projection_matches(
            other, "The query"
        )


def test_other_settings_are_ignored_without_projection():
    unprojected = {**CHANGED, "projection_dim": None}
    GradientProcessor(projection_dim=None).check_projection_matches(
        GradientProcessor(**unprojected), "The query"
    )


def test_projecting_is_compared_with_not_projecting():
    with pytest.raises(ValueError, match="projection_dim=None, not 16"):
        GradientProcessor(projection_dim=16).check_projection_matches(
            GradientProcessor(projection_dim=None), "The query"
        )
    with pytest.raises(ValueError, match="projection_dim=16, not None"):
        GradientProcessor(projection_dim=None).check_projection_matches(
            GradientProcessor(projection_dim=16), "The query"
        )


def test_load_config_reads_only_the_config(tmp_path):
    """Configs from before projection_scale existed load as row_norm, the
    scaling they were built with."""
    GradientProcessor(projection_dim=8, projection_seed=5).save(tmp_path)
    cfg_path = tmp_path / "processor_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    del cfg["projection_scale"]
    cfg_path.write_text(yaml.safe_dump(cfg))
    (tmp_path / "normalizers.pth").unlink()

    processor = GradientProcessor.load_config(tmp_path)
    assert processor.projection_dim == 8
    assert processor.projection_seed == 5
    assert processor.projection_scale == "row_norm"


def test_scoring_rejects_a_query_projected_differently(tmp_path):
    query = tmp_path / "query"
    GradientProcessor(projection_dim=16, projection_seed=1).save(query)
    index_cfg = IndexConfig(
        run_path=str(tmp_path / "scores"), projection_dim=16, projection_seed=2
    )
    with pytest.raises(ValueError, match="projection_seed=1, not 2"):
        score_dataset(index_cfg, ScoreConfig(query_path=str(query)), PreprocessConfig())


def test_scoring_rejects_a_hessian_projected_differently(tmp_path):
    query, hessian = tmp_path / "query", tmp_path / "hessian"
    GradientProcessor(projection_dim=16).save(query)
    GradientProcessor(projection_dim=16, projection_scale="row_norm").save(hessian)
    index_cfg = IndexConfig(run_path=str(tmp_path / "scores"), projection_dim=16)
    with pytest.raises(ValueError, match="The Hessian at .*projection_scale"):
        score_dataset(
            index_cfg,
            ScoreConfig(query_path=str(query)),
            PreprocessConfig(hessian_path=str(hessian)),
        )


def _save_autocorrelation(path, **settings):
    eigen = (torch.tensor([2.0, 1.0]), torch.eye(2))
    GradientProcessor(
        hessians={"layer": torch.eye(2)},
        hessians_eigen={"layer": eigen},
        projection_dim=16,
        **settings,
    ).save(path)


def test_mixing_rejects_hessians_projected_differently(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "bergson.process_grads.assert_autocorrelation_hessian", lambda path: None
    )
    _save_autocorrelation(tmp_path / "query", projection_seed=1)
    _save_autocorrelation(tmp_path / "index", projection_seed=2)
    with pytest.raises(ValueError, match="The index Hessian"):
        mix_autocorrelation_matrices(
            tmp_path / "query", tmp_path / "index", tmp_path / "mixed"
        )


def test_mixing_keeps_every_projection_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "bergson.process_grads.assert_autocorrelation_hessian", lambda path: None
    )
    settings = {"projection_seed": 1, "projection_scale": "row_norm"}
    _save_autocorrelation(tmp_path / "query", **settings)
    _save_autocorrelation(tmp_path / "index", **settings)
    mix_autocorrelation_matrices(
        tmp_path / "query", tmp_path / "index", tmp_path / "mixed", 1
    )
    mixed = GradientProcessor.load(tmp_path / "mixed")
    assert (mixed.projection_seed, mixed.projection_scale) == (1, "row_norm")
