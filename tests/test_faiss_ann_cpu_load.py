"""Regression test for the FAISS ANN off-the-shelf breakage.

`FaissIndex.__init__` reads each on-disk shard with ``IO_FLAG_MMAP`` (a CPU index).
When ``mmap_index=False`` (the shipped default) and ``device="cpu"``, the loader
calls ``index_to_device(shard, "cpu")``. That used to run ``faiss.index_gpu_to_cpu``
on the already-CPU index, cloning it and raising ``RuntimeError: clone not supported
... OnDiskInvertedLists`` for any IVF/ANN index mmap'd from disk. Only an exact
``Flat`` index survived. ``index_to_device`` now detects GPU residency and treats a
CPU->CPU request as a no-op, so an already-CPU shard is returned unchanged.

The end-to-end tests build tiny on-disk indices with ``device="cpu"`` and confirm
the CPU load path (``mmap_index=False``) no longer raises and returns usable
neighbours for both an ANN (IVF) index and an exact ``Flat`` index. The unit tests
exercise the ``index_to_device`` guard directly. Adapted from the bug2 repro.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from bergson.config import FaissConfig
from bergson.query.faiss_index import FaissIndex, _is_gpu_resident, index_to_device


def _has_faiss() -> bool:
    try:
        import faiss  # type: ignore[import]  # noqa: F401

        return True
    except ImportError:
        return False


requires_faiss = pytest.mark.skipif(not _has_faiss(), reason="faiss not available")


def _write_gradient_store(root: Path, n: int, dim: int) -> np.ndarray:
    """Write a tiny on-disk gradient store matching bergson's info.json layout."""
    root.mkdir(parents=True, exist_ok=True)
    grad_sizes = {"module_a": dim}
    struct_dtype = {
        "names": list(grad_sizes),
        "formats": [f"({s},)<f4" for s in grad_sizes.values()],
        "itemsize": 4 * sum(grad_sizes.values()),
    }
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, dim)).astype("<f4")
    mm = np.memmap(
        root / "gradients.bin",
        dtype=np.dtype(struct_dtype),
        mode="w+",
        shape=(n,),
    )
    mm["module_a"] = data
    mm.flush()
    with (root / "info.json").open("w") as f:
        json.dump(
            {
                "num_grads": n,
                "dtype": struct_dtype,
                "grad_sizes": grad_sizes,
                "base_dtype": "float32",
            },
            f,
        )
    return data


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _build_and_load(
    grad_root: Path,
    idx_path: Path,
    index_factory: str,
    mmap_index: bool,
    query: np.ndarray,
):
    cfg = FaissConfig(index_factory=index_factory, mmap_index=mmap_index, num_shards=1)
    # unit_norm=True stores cosine-normalized gradients; with the inner-product
    # metric the self-match then has cosine 1.0, so a query row's own index is
    # deterministically its top neighbour (independent of vector magnitudes).
    FaissIndex.create_index(
        gradients_path=grad_root,
        faiss_path=idx_path,
        faiss_cfg=cfg,
        device="cpu",
        unit_norm=True,
        hessians={},
    )
    # This constructor is where the bug used to raise for an mmap'd IVF index.
    fi = FaissIndex(idx_path, device="cpu", mmap_index=mmap_index)
    distances, indices = fi.search(_l2_normalize(query), k=3)
    return distances, indices


@requires_faiss
def test_ann_ivf_cpu_load_does_not_raise(tmp_path: Path):
    """An IVF (ANN) index mmap'd from disk must load on CPU without raising."""
    n, dim = 256, 16
    data = _write_gradient_store(tmp_path / "grads", n, dim)
    query = data[:2].copy()

    # device="cpu", mmap_index=False is the shipped default that used to crash.
    distances, indices = _build_and_load(
        tmp_path / "grads",
        tmp_path / "faiss_ivf",
        index_factory="IVF16,Flat",
        mmap_index=False,
        query=query,
    )

    assert indices.shape == (2, 3)
    # Each query vector is a stored gradient, so its own row should be the
    # top (self) match. IVF is approximate but with nprobe covering the tiny
    # index the exact self-match is recovered.
    assert indices[0, 0] == 0
    assert indices[1, 0] == 1
    # Results must be real (not the -1 "not found" sentinel).
    assert (indices >= 0).all()


@requires_faiss
def test_exact_flat_cpu_load_still_works(tmp_path: Path):
    """The exact Flat path that already worked must keep working."""
    n, dim = 256, 16
    data = _write_gradient_store(tmp_path / "grads", n, dim)
    query = data[:2].copy()

    distances, indices = _build_and_load(
        tmp_path / "grads",
        tmp_path / "faiss_flat",
        index_factory="Flat",
        mmap_index=False,
        query=query,
    )

    assert indices.shape == (2, 3)
    assert indices[0, 0] == 0
    assert indices[1, 0] == 1
    assert (indices >= 0).all()


@requires_faiss
def test_index_to_device_cpu_is_noop_on_in_memory_cpu_index():
    """`index_to_device(idx, "cpu")` returns an in-memory CPU index unchanged."""
    import faiss  # type: ignore[import]

    idx = faiss.index_factory(16, "IVF16,Flat", faiss.METRIC_INNER_PRODUCT)
    assert not _is_gpu_resident(idx)

    result = index_to_device(idx, "cpu")

    # Same object, not a clone: proves we did not route it through
    # `index_gpu_to_cpu` (which would return a fresh index).
    assert result is idx


@requires_faiss
def test_index_to_device_cpu_is_noop_on_mmapd_ondisk_index(tmp_path: Path):
    """The guard must no-op on the mmap'd OnDisk index that used to crash.

    Reading an IVF shard with ``IO_FLAG_MMAP`` yields a CPU index backed by
    ``OnDiskInvertedLists``. Calling ``faiss.index_gpu_to_cpu`` on it raises
    ``clone not supported ... OnDiskInvertedLists`` -- we assert that raw failure
    to prove the case is real, then assert ``index_to_device`` sidesteps it.
    """
    import faiss  # type: ignore[import]

    n, dim = 256, 16
    _write_gradient_store(tmp_path / "grads", n, dim)
    FaissIndex.create_index(
        gradients_path=tmp_path / "grads",
        faiss_path=tmp_path / "faiss_ivf",
        faiss_cfg=FaissConfig(index_factory="IVF16,Flat", num_shards=1),
        device="cpu",
        unit_norm=True,
        hessians={},
    )
    (shard_path,) = (tmp_path / "faiss_ivf").glob("*.faiss")
    shard = faiss.read_index(
        str(shard_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
    )

    assert not _is_gpu_resident(shard)
    # The raw clone that the old code performed still fails on this index...
    with pytest.raises(RuntimeError, match="clone not supported"):
        faiss.index_gpu_to_cpu(shard)
    # ...but the guarded helper returns it unchanged instead of cloning.
    assert index_to_device(shard, "cpu") is shard


@requires_faiss
def test_index_to_device_cpu_converts_a_gpu_resident_index():
    """When the index needs moving, `index_to_device` actually converts it.

    A CPU-only faiss build can't allocate a real ``GpuIndex``, but an
    ``IndexShards`` container is exactly what a multi-GPU move returns and what
    ``_is_gpu_resident`` classifies as needing conversion. Bringing it to CPU must
    return a *new*, non-container, still-searchable index -- i.e. an op, not a
    no-op.
    """
    import faiss  # type: ignore[import]

    d, n = 8, 6
    vecs = np.random.default_rng(0).standard_normal((n, d)).astype("float32")

    shards = faiss.IndexShards(d)
    sub = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
    sub.add(vecs)
    shards.add_shard(sub)

    assert _is_gpu_resident(shards)

    out = index_to_device(shards, "cpu")

    # A real conversion happened: a new object that is no longer a container.
    assert out is not shards
    assert not _is_gpu_resident(out)
    assert out.ntotal == n
    # ...and the converted index still searches.
    _, indices = out.search(vecs[:1], 1)
    assert indices.shape == (1, 1)
    assert 0 <= indices[0, 0] < n


@requires_faiss
def test_ann_ivf_mmap_index_true_still_works(tmp_path: Path):
    """The mmap_index=True path (skips index_to_device) must be unaffected."""
    n, dim = 256, 16
    data = _write_gradient_store(tmp_path / "grads", n, dim)
    query = data[:2].copy()

    distances, indices = _build_and_load(
        tmp_path / "grads",
        tmp_path / "faiss_ivf_mmap",
        index_factory="IVF16,Flat",
        mmap_index=True,
        query=query,
    )

    assert indices.shape == (2, 3)
    assert indices[0, 0] == 0
    assert indices[1, 0] == 1
    assert (indices >= 0).all()
