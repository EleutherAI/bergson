"""End-to-end tests for the EKFAC inversion methods.

Runs `EkfacApplicator.compute_ivhp_sharded` against the real fitted Hessian
factors (via the `ekfac_results_path` fixture) for every `inversion` option and
checks each is wired up, produces finite output, and that the methods differ
from one another.
"""

import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from bergson.config import InversionConfig
from bergson.data import create_index, load_module_gradients
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.hessians.inversion import INVERSIONS


def _make_query_gradients(query_path: str, grad_sizes: dict[str, int], num_grads: int):
    """Write a small query-gradient index with random gradients."""
    index = create_index(
        root=Path(query_path),
        num_grads=num_grads,
        grad_sizes=grad_sizes,
        dtype=np.float32,
    )
    rng = np.random.default_rng(0)
    index[:] = rng.standard_normal(index.shape).astype(np.float32)
    index.flush()


def _apply(
    hessian_path,
    query_path,
    out_path,
    inversion,
    damping_factor=0.1,
    ev_correction=True,
):
    cfg = EkfacConfig(
        hessian_method_path=hessian_path,
        gradient_path=query_path,
        run_path=out_path,
        ev_correction=ev_correction,
    )
    inversion_cfg = InversionConfig(inversion=inversion, damping_factor=damping_factor)
    EkfacApplicator(cfg, inversion_cfg=inversion_cfg).compute_ivhp_sharded()
    return load_module_gradients(out_path)


def test_inversion_methods_apply(ekfac_results_path: str, tmp_path):
    hessian_path = ekfac_results_path

    # Derive grad_sizes (O * I per module) from the fitted eigenvectors.
    eigen_a = load_file(
        os.path.join(hessian_path, "eigen_activation_sharded/shard_0.safetensors")
    )
    eigen_g = load_file(
        os.path.join(hessian_path, "eigen_gradient_sharded/shard_0.safetensors")
    )
    grad_sizes = {k: eigen_g[k].shape[1] * eigen_a[k].shape[1] for k in eigen_a}

    num_grads = 3
    query_path = str(tmp_path / "query")
    _make_query_gradients(query_path, grad_sizes, num_grads)

    sample = sorted(grad_sizes.keys())[0]
    outputs = {}
    for inversion in INVERSIONS:
        # factored_tikhonov and pseudoinverse_factored operate on the Kronecker
        # factors, which assumes λ = λ_G ⊗ λ_A; the EV-corrected eigenvalues do
        # not factorize, so they are applied to the uncorrected grid instead.
        ev_correction = inversion not in (
            "factored_tikhonov",
            "pseudoinverse_factored",
        )
        out = _apply(
            hessian_path,
            query_path,
            str(tmp_path / f"out_{inversion}"),
            inversion,
            ev_correction=ev_correction,
        )
        t = torch.from_numpy(np.asarray(out[sample][:]))
        assert torch.isfinite(t).all(), f"{inversion} produced non-finite output"
        assert t.abs().sum() > 0, f"{inversion} produced all-zero output"
        outputs[inversion] = t

    # Each inversion regularizes differently, so the preconditioned gradients
    # should differ pairwise (same query, same damping_factor).
    for a in INVERSIONS:
        for b in INVERSIONS:
            if a < b:
                assert not torch.allclose(
                    outputs[a], outputs[b]
                ), f"{a} and {b} produced identical output"
