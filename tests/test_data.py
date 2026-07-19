import numpy as np

from bergson.data import (
    create_index,
    create_token_index,
    load_gradients,
    load_module_gradients,
)


def test_large_gradients_build(tmp_path, dataset):
    # Uncompressed gradients from a large (~1.4B-param) model have rows whose
    # size (4 bytes * total elements) overflows the C-int cap (2**31 - 1 bytes)
    # that numpy imposes on a structured record's itemsize — the reason the
    # store is mapped flat. Fabricate per-module gradient sizes at that scale
    # directly so the test needs no model and no GPU.
    grad_sizes = {f"layer_{i}.weight": 50_000_000 for i in range(12)}  # 6e8 elems
    assert sum(grad_sizes.values()) * np.dtype(np.float32).itemsize > 2**31

    create_index(
        tmp_path,
        num_grads=len(dataset),
        grad_sizes=grad_sizes,
        dtype=np.float32,
    )

    mmap = load_gradients(tmp_path)
    assert mmap.shape == (len(dataset), sum(grad_sizes.values()))

    # The module-keyed view slices the same rows by column.
    grads = load_module_gradients(tmp_path)
    assert len(grads) == len(dataset)
    assert list(grads.keys()) == list(grad_sizes)
    assert grads["layer_3.weight"].shape == (len(dataset), 50_000_000)


def test_module_gradients_roundtrip(tmp_path, dataset):
    grad_sizes = {"a.weight": 3, "b.weight": 5}
    buffer = create_index(
        tmp_path,
        num_grads=len(dataset),
        grad_sizes=grad_sizes,
        dtype=np.float32,
    )
    rng = np.random.default_rng(0)
    expected = rng.standard_normal((len(dataset), 8)).astype(np.float32)
    buffer[:] = expected
    buffer.flush()

    grads = load_module_gradients(tmp_path)
    np.testing.assert_array_equal(grads["a.weight"], expected[:, :3])
    np.testing.assert_array_equal(grads["b.weight"], expected[:, 3:])


def test_load_gradients_token_index_roundtrip(tmp_path):
    """load_gradients must also read the per-token store created by
    create_token_index."""
    grad_sizes = {"a.weight": 3, "b.weight": 5}
    num_token_grads = np.array([2, 3, 1], dtype=np.int64)
    buffer, offsets = create_token_index(
        tmp_path,
        num_token_grads=num_token_grads,
        grad_sizes=grad_sizes,
        dtype=np.float32,
    )
    total_tokens = int(offsets[-1])
    assert total_tokens == num_token_grads.sum()

    rng = np.random.default_rng(0)
    expected = rng.standard_normal((total_tokens, 8)).astype(np.float32)
    buffer[:] = expected
    buffer.flush()

    mmap = load_gradients(tmp_path)
    assert mmap.shape == (total_tokens, 8)
    np.testing.assert_array_equal(mmap[:], expected)

    grads = load_module_gradients(tmp_path)
    assert len(grads) == total_tokens
    np.testing.assert_array_equal(grads["a.weight"], expected[:, :3])
    np.testing.assert_array_equal(grads["b.weight"], expected[:, 3:])
