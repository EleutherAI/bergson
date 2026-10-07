"""Fitting and applying the factors in module partitions must match doing it
in one pass: the merged factor stores and the IVHP output are the same."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from bergson.config import (
    DataConfig,
    DistributedConfig,
    HessianConfig,
    IndexConfig,
    InversionConfig,
)
from bergson.data import create_index, load_gradients
from bergson.hessians import hessian_approximations
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.hessians.hessian_approximations import (
    FACTOR_SUBDIRS,
    approximate_hessians,
    merge_partitions,
    partition_modules,
)


def test_partition_modules_contiguous():
    names = [f"m{i}" for i in range(7)]
    assert partition_modules(names, 3) == [names[:3], names[3:5], names[5:]]
    assert partition_modules(names, 1) == [names]
    assert partition_modules(names, 20) == [[n] for n in names]
    with pytest.raises(ValueError):
        partition_modules(names, 0)


def write_partitions(run_path: Path, num_partitions: int, num_ranks: int) -> dict:
    """Write random partition shards and return the merged tensors expected
    for each ``(subdir, rank)``."""
    expected = {}
    gen = torch.Generator().manual_seed(0)
    for sub in FACTOR_SUBDIRS[:3]:
        for rank in range(num_ranks):
            expected[sub, rank] = {}
            for p in range(num_partitions):
                tensors = {
                    f"layer{p}.m{rank}.{i}": torch.randn(32, 32, generator=gen)
                    for i in range(2)
                }
                expected[sub, rank].update(tensors)
                path = run_path / f"partition_{p}" / sub
                path.mkdir(parents=True, exist_ok=True)
                save_file(tensors, path / f"shard_{rank}.safetensors")
    return expected


def assert_merged(run_path: Path, expected: dict):
    assert sorted(os.listdir(run_path)) == sorted({sub for sub, _ in expected})
    for (sub, rank), tensors in expected.items():
        assert sorted(os.listdir(run_path / sub)) == [
            f"shard_{r}.safetensors" for r in sorted({r for _, r in expected})
        ]
        merged = load_file(run_path / sub / f"shard_{rank}.safetensors")
        assert merged.keys() == tensors.keys()
        for key in tensors:
            torch.testing.assert_close(merged[key], tensors[key], rtol=0, atol=0)


def disk_usage(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def test_merge_partitions_frees_partitions_as_it_merges(tmp_path: Path, monkeypatch):
    run_path = tmp_path / "run.part"
    expected = write_partitions(run_path, num_partitions=3, num_ranks=2)
    before = disk_usage(run_path)

    peak = 0
    save_file_orig = hessian_approximations.save_file

    def save_and_measure(tensors, filename):
        nonlocal peak
        save_file_orig(tensors, filename)
        peak = max(peak, disk_usage(run_path))

    monkeypatch.setattr(hessian_approximations, "save_file", save_and_measure)
    # Without a process group, run rank 0 last since it deletes the partitions.
    for rank in (1, 0):
        merge_partitions(run_path, 3, rank)

    assert_merged(run_path, expected)
    largest_shard = max(
        f.stat().st_size for f in run_path.rglob("*.safetensors") if f.is_file()
    )
    assert disk_usage(run_path) <= before
    assert peak <= before + largest_shard


def test_merge_partitions_resumes_after_interruption(tmp_path: Path, monkeypatch):
    run_path = tmp_path / "run.part"
    expected = write_partitions(run_path, num_partitions=3, num_ranks=1)

    # Interrupt partway through deleting the second store's partition shards.
    remove_orig = os.remove
    removed = 0

    def remove_then_crash(path):
        nonlocal removed
        if removed == 4:
            raise KeyboardInterrupt
        removed += 1
        remove_orig(path)

    monkeypatch.setattr(os, "remove", remove_then_crash)
    with pytest.raises(KeyboardInterrupt):
        merge_partitions(run_path, 3, 0)
    monkeypatch.setattr(os, "remove", remove_orig)

    # A stale temporary file from a merge killed while saving.
    sub = FACTOR_SUBDIRS[2]
    (run_path / sub).mkdir()
    (run_path / sub / "shard_0.safetensors.tmp").write_bytes(b"truncated")

    merge_partitions(run_path, 3, 0)
    assert_merged(run_path, expected)


def test_merge_partitions_replaces_shards_from_an_earlier_run(tmp_path: Path):
    run_path = tmp_path / "run.part"
    write_partitions(run_path, num_partitions=3, num_ranks=1)
    merge_partitions(run_path, 3, 0)

    # A resumed fit rewrites every partition next to the earlier merged shards.
    for sub in FACTOR_SUBDIRS[:3]:
        stale = load_file(run_path / sub / "shard_0.safetensors")
        save_file({k: v + 1 for k, v in stale.items()}, run_path / sub / "x")
        os.replace(run_path / sub / "x", run_path / sub / "shard_0.safetensors")
    expected = write_partitions(run_path, num_partitions=3, num_ranks=1)

    merge_partitions(run_path, 3, 0)
    assert_merged(run_path, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_module_partitions_match_single_pass(tmp_path: Path):
    def fit(partitions: int) -> str:
        cfg = IndexConfig(
            run_path=str(tmp_path / f"kfac_p{partitions}"),
            model="EleutherAI/pythia-14m",
            data=DataConfig(
                dataset="NeelNanda/pile-10k", split="train[:8]", truncation=True
            ),
            token_batch_size=512,
            precision="fp32",
            filter_modules="embed_out",
            distributed=DistributedConfig(nproc_per_node=1),
        )
        # Dataset labels: sampled labels would differ between the two fits.
        hessian_cfg = HessianConfig(
            method="kfac",
            ev_correction=True,
            use_dataset_labels=True,
            module_partitions=partitions,
        )
        return approximate_hessians(cfg, hessian_cfg)

    single, split = fit(1), fit(3)
    assert not any(
        d.startswith("partition_") for d in os.listdir(split)
    ), "partition directories must be merged away"

    for sub in FACTOR_SUBDIRS:
        a = load_file(os.path.join(single, sub, "shard_0.safetensors"))
        b = load_file(os.path.join(split, sub, "shard_0.safetensors"))
        assert a.keys() == b.keys(), sub
        for key in a:
            # Eigenvectors are sign-ambiguous per column; compare up to sign.
            if sub.startswith("eigen_"):
                torch.testing.assert_close(
                    a[key].abs(), b[key].abs(), rtol=1e-4, atol=1e-5
                )
            else:
                torch.testing.assert_close(
                    a[key], b[key], rtol=1e-4, atol=1e-5, msg=f"{sub}/{key}"
                )

    eigen_a = load_file(
        os.path.join(single, "eigen_activation_sharded/shard_0.safetensors")
    )
    eigen_g = load_file(
        os.path.join(single, "eigen_gradient_sharded/shard_0.safetensors")
    )
    grad_sizes = {k: eigen_g[k].shape[1] * eigen_a[k].shape[1] for k in eigen_a}
    query_path = tmp_path / "queries"
    index = create_index(
        root=query_path, num_grads=3, grad_sizes=grad_sizes, dtype=np.float32
    )
    index[:] = np.random.default_rng(0).standard_normal(index.shape).astype(np.float32)
    index.flush()

    outputs = []
    for partitions in (1, 3):
        out = tmp_path / f"ivhp_p{partitions}"
        cfg = EkfacConfig(
            hessian_method_path=single,
            gradient_path=str(query_path),
            run_path=str(out),
            ev_correction=True,
            module_partitions=partitions,
        )
        EkfacApplicator(cfg, inversion_cfg=InversionConfig()).compute_ivhp_sharded()
        outputs.append(np.asarray(load_gradients(out)))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
