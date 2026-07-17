import numpy as np
import pytest

from bergson.data import create_index, load_gradients


def test_large_gradients_build(tmp_path, dataset):
    # Uncompressed gradients from a large (~1.4B-param) model produce a structured
    # numpy dtype whose itemsize (4 bytes * total elements) overflows the C-int cap
    # (2**31 - 1 bytes), so structured loading must fail while unstructured loading
    # succeeds. Fabricate per-module gradient sizes at that scale directly so the
    # test needs no model and no GPU.
    grad_sizes = {f"layer_{i}.weight": 50_000_000 for i in range(12)}  # 6e8 elems
    assert sum(grad_sizes.values()) * np.dtype(np.float32).itemsize > 2**31

    create_index(
        tmp_path,
        num_grads=len(dataset),
        grad_sizes=grad_sizes,
        dtype=np.float32,
        with_structure=False,
    )

    # Unstructured load works; structured load exceeds numpy's max item size.
    load_gradients(tmp_path, structured=False)
    with pytest.raises(ValueError):
        load_gradients(tmp_path, structured=True)
