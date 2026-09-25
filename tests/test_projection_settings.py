"""Projected gradients and Hessians are saved with the settings they were
projected with, so runs that reuse them can check that they match."""

import torch
import yaml

from bergson import GradientProcessor
from bergson.process_grads import mix_autocorrelation_matrices


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


def _save_autocorrelation(path, **settings):
    eigen = (torch.tensor([2.0, 1.0]), torch.eye(2))
    GradientProcessor(
        hessians={"layer": torch.eye(2)},
        hessians_eigen={"layer": eigen},
        projection_dim=16,
        **settings,
    ).save(path)


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
