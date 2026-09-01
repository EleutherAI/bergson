"""Tests for averaging query-gradient indices across model checkpoints."""

import json

import numpy as np
import pytest

from bergson.data import average_gradient_indices, load_gradients


def _write_index(root, rows, dtype=np.float32, grad_sizes=None):
    root.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rows, dtype=dtype)
    info = {
        "num_grads": arr.shape[0],
        "grad_sizes": grad_sizes or {"mlp": arr.shape[1]},
        "base_dtype": np.dtype(dtype).name,
    }
    (root / "info.json").write_text(json.dumps(info))
    mm = np.memmap(root / "gradients.bin", dtype=dtype, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    return root


def test_average_gradient_indices(tmp_path):
    a = _write_index(tmp_path / "a", [[0.0, 2.0], [4.0, 6.0]])
    b = _write_index(tmp_path / "b", [[2.0, 4.0], [8.0, 10.0]])
    average_gradient_indices([a, b], tmp_path / "out")
    assert np.allclose(
        np.asarray(load_gradients(tmp_path / "out")), [[1.0, 3.0], [6.0, 8.0]]
    )
    info = json.loads((tmp_path / "out" / "info.json").read_text())
    assert info["num_grads"] == 2 and info["grad_sizes"] == {"mlp": 2}

    average_gradient_indices([a], tmp_path / "one")
    assert np.allclose(
        np.asarray(load_gradients(tmp_path / "one")), [[0.0, 2.0], [4.0, 6.0]]
    )

    # The float64 accumulator keeps a long running sum from drifting before the
    # divide; the exact mean here is 1e6 + 31.5.
    n = 64
    srcs = [_write_index(tmp_path / f"s{i}", [[1e6 + i]]) for i in range(n)]
    average_gradient_indices(srcs, tmp_path / "many")
    got = float(np.asarray(load_gradients(tmp_path / "many"))[0, 0])
    assert got == pytest.approx(1e6 + (n - 1) / 2, abs=0.05)

    with pytest.raises(ValueError, match="rows"):
        average_gradient_indices(
            [a, _write_index(tmp_path / "short", [[1.0, 1.0]])], tmp_path / "bad"
        )
    with pytest.raises(ValueError, match="module layout"):
        average_gradient_indices(
            [
                a,
                _write_index(
                    tmp_path / "attn",
                    [[1.0, 1.0], [1.0, 1.0]],
                    grad_sizes={"attn": 2},
                ),
            ],
            tmp_path / "bad",
        )
    with pytest.raises(ValueError):
        average_gradient_indices([], tmp_path / "bad")
