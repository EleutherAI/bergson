"""Tests for averaging query-gradient indices across model checkpoints."""

import json

import numpy as np
import pytest

from bergson.data import average_gradient_indices, load_gradients


def _write_index(root, rows, dtype=np.float32, grad_sizes=None):
    root.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rows, dtype=dtype)
    sizes = grad_sizes or {"mlp": arr.shape[1]}
    info = {
        "num_grads": arr.shape[0],
        "grad_sizes": sizes,
        "base_dtype": np.dtype(dtype).name,
    }
    (root / "info.json").write_text(json.dumps(info))
    mm = np.memmap(root / "gradients.bin", dtype=dtype, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    return root


def test_averages_elementwise(tmp_path):
    a = _write_index(tmp_path / "a", [[0.0, 2.0], [4.0, 6.0]])
    b = _write_index(tmp_path / "b", [[2.0, 4.0], [8.0, 10.0]])
    average_gradient_indices([a, b], tmp_path / "out")
    got = np.asarray(load_gradients(tmp_path / "out"))
    assert np.allclose(got, [[1.0, 3.0], [6.0, 8.0]])


def test_single_source_is_a_copy(tmp_path):
    a = _write_index(tmp_path / "a", [[1.5, -2.5]])
    average_gradient_indices([a], tmp_path / "out")
    assert np.allclose(np.asarray(load_gradients(tmp_path / "out")), [[1.5, -2.5]])


def test_preserves_shape_and_info(tmp_path):
    a = _write_index(tmp_path / "a", [[1.0, 1.0], [1.0, 1.0]])
    b = _write_index(tmp_path / "b", [[3.0, 3.0], [3.0, 3.0]])
    average_gradient_indices([a, b], tmp_path / "out")
    info = json.loads((tmp_path / "out" / "info.json").read_text())
    assert info["num_grads"] == 2
    assert info["grad_sizes"] == {"mlp": 2}


def test_row_count_mismatch_raises(tmp_path):
    a = _write_index(tmp_path / "a", [[1.0, 1.0]])
    b = _write_index(tmp_path / "b", [[1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="rows"):
        average_gradient_indices([a, b], tmp_path / "out")


def test_module_layout_mismatch_raises(tmp_path):
    a = _write_index(tmp_path / "a", [[1.0, 1.0]], grad_sizes={"mlp": 2})
    b = _write_index(tmp_path / "b", [[1.0, 1.0]], grad_sizes={"attn": 2})
    with pytest.raises(ValueError, match="module layout"):
        average_gradient_indices([a, b], tmp_path / "out")


def test_empty_sources_raises(tmp_path):
    with pytest.raises(ValueError):
        average_gradient_indices([], tmp_path / "out")


def test_many_sources_average_is_accurate(tmp_path):
    """Averaging many indices must not drift.

    The stored dtype is float32, so a sub-ULP difference between two sources can
    never survive the write no matter how the sum is accumulated -- that is a
    property of the store, not of this function. What the float64 accumulator
    buys is that a long running sum does not drift before the divide, which is
    what this checks: 64 sources whose exact mean is representable.
    """
    n = 64
    srcs = []
    for i in range(n):
        # Values straddling a large offset, so a naive low-precision running sum
        # loses the small terms; the exact mean is 1e6 + 31.5.
        srcs.append(_write_index(tmp_path / f"s{i}", [[1e6 + i]]))
    average_gradient_indices(srcs, tmp_path / "out")
    got = float(np.asarray(load_gradients(tmp_path / "out"))[0, 0])
    assert got == pytest.approx(1e6 + (n - 1) / 2, abs=0.05)
