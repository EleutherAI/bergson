"""``matrix:`` expansion of pipeline steps."""

import pytest

from bergson.config.config_io import expand_matrix


def test_no_matrix_passthrough():
    cfg = {"run_path": "runs/x", "seed": 1}
    assert expand_matrix(dict(cfg)) == [cfg]


def test_cartesian_expansion_with_typed_and_string_substitution():
    out = expand_matrix(
        {
            "matrix": {"seed": [1, 2], "lr": [0.1]},
            "run_path": "runs/s{seed}_lr{lr}",
            "seed": "{seed}",
            "lr_schedule": {"lr": "{lr}"},
        }
    )
    assert len(out) == 2
    assert out[0] == {
        "run_path": "runs/s1_lr0.1",
        "seed": 1,
        "lr_schedule": {"lr": 0.1},
    }
    assert out[1]["seed"] == 2 and out[1]["run_path"] == "runs/s2_lr0.1"


def test_colliding_run_paths_raise():
    with pytest.raises(ValueError, match="both write"):
        expand_matrix(
            {"matrix": {"seed": [1, 2]}, "run_path": "runs/same", "seed": "{seed}"}
        )


def test_matrix_without_run_path_raises():
    with pytest.raises(ValueError, match="no run_path"):
        expand_matrix({"matrix": {"seed": [1, 2]}, "seed": "{seed}"})


def test_nested_run_path_collision_detected():
    with pytest.raises(ValueError, match="both write"):
        expand_matrix(
            {
                "matrix": {"d": [0.1, 0.2]},
                "index_cfg": {"run_path": "runs/fixed"},
                "damping": "{d}",
            }
        )


def test_bad_matrix_shape_raises():
    with pytest.raises(ValueError, match="non-empty lists"):
        expand_matrix({"matrix": {"seed": 5}})
