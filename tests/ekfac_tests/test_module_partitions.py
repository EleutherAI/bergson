"""Fitting and applying the factors in module partitions must match doing it
in one pass: the merged factor stores and the IVHP output are the same."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from bergson.config import (
    DataConfig,
    DistributedConfig,
    HessianConfig,
    IndexConfig,
    InversionConfig,
)
from bergson.data import create_index, load_gradients
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.hessians.hessian_approximations import (
    FACTOR_SUBDIRS,
    approximate_hessians,
    partition_modules,
)


def test_partition_modules_contiguous():
    names = [f"m{i}" for i in range(7)]
    assert partition_modules(names, 3) == [names[:3], names[3:5], names[5:]]
    assert partition_modules(names, 1) == [names]
    assert partition_modules(names, 20) == [[n] for n in names]
    with pytest.raises(ValueError):
        partition_modules(names, 0)


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
