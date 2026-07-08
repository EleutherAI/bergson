import numpy as np

from bergson.data import FlatGradientView, create_index, load_gradients


def _oversized_grad_sizes() -> dict[str, int]:
    """Two modules whose combined fp32 record overflows numpy's C-int itemsize
    cap (2**31 - 1 bytes), i.e. > 2**31 / 4 float32 elements total.

    This is the regime an unprojected EK-FAC query gradient hits once a model
    has more than ~537M tracked-linear params. The store is allocated as a
    *sparse* file (``truncate`` only sets the size), so the test writes/reads a
    handful of pages rather than materializing ~2 GB.
    """
    per_module = 2**31 // 4 // 2 + 1  # just over half the fp32 element cap
    grad_sizes = {"layer_0.weight": per_module, "layer_1.weight": per_module}
    assert sum(grad_sizes.values()) * np.dtype(np.float32).itemsize > 2**31
    return grad_sizes


def test_large_gradient_store_roundtrips(tmp_path):
    # A structured record for a >537M-param model overflows numpy's C-int
    # itemsize cap, so create_index/load_gradients must transparently fall back
    # to the byte-identical flat 2D layout (served via FlatGradientView) instead
    # of raising while constructing the structured dtype. CPU-only, no model.
    grad_sizes = _oversized_grad_sizes()
    names = list(grad_sizes)
    size = grad_sizes[names[0]]

    # create_index defaults to with_structure=True (the EK-FAC path); it must not
    # raise on the oversized record.
    buf = create_index(tmp_path, num_grads=1, grad_sizes=grad_sizes, dtype=np.float32)
    assert isinstance(buf, FlatGradientView)
    assert buf.dtype.names == tuple(names)
    assert len(buf) == 1

    # Field access returns the module's column block; write sentinels at the
    # start/end of each module so we exercise the full flattened offset range,
    # including the boundary between the two modules.
    buf[names[0]][0, 0] = 1.5
    buf[names[0]][0, size - 1] = 2.5
    buf[names[1]][0, 0] = 3.5
    buf[names[1]][0, size - 1] = 4.5
    buf.flush()

    # Structured load now succeeds (previously raised ValueError) and round-trips.
    view = load_gradients(tmp_path, structured=True)
    assert isinstance(view, FlatGradientView)
    assert view.dtype.names == tuple(names)
    assert view[names[0]][0, 0] == 1.5
    assert view[names[0]][0, size - 1] == 2.5
    assert view[names[1]][0, 0] == 3.5
    assert view[names[1]][0, size - 1] == 4.5

    # The flat (unstructured) view of the same bytes must agree: the structured
    # record layout and the flat 2D layout are byte-identical, so the second
    # module's block starts exactly at column `size`.
    flat = load_gradients(tmp_path, structured=False)
    assert flat.shape == (1, 2 * size)
    assert flat[0, 0] == 1.5
    assert flat[0, size - 1] == 2.5
    assert flat[0, size] == 3.5
    assert flat[0, 2 * size - 1] == 4.5


def test_small_gradient_store_stays_structured(tmp_path):
    # Below the itemsize cap the behavior is unchanged: create_index/load_gradients
    # return a genuine structured numpy memmap with real field access.
    grad_sizes = {"layer_0.weight": 8, "layer_1.weight": 16}
    assert sum(grad_sizes.values()) * np.dtype(np.float32).itemsize <= 2**31

    buf = create_index(tmp_path, num_grads=3, grad_sizes=grad_sizes, dtype=np.float32)
    assert isinstance(buf, np.memmap)
    assert buf.dtype.names == ("layer_0.weight", "layer_1.weight")

    buf["layer_0.weight"][:] = 1.0
    buf["layer_1.weight"][:] = 2.0
    buf.flush()

    view = load_gradients(tmp_path, structured=True)
    assert isinstance(view, np.memmap)
    assert np.all(view["layer_0.weight"] == 1.0)
    assert np.all(view["layer_1.weight"] == 2.0)
