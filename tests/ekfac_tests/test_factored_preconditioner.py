"""Validate the in-memory FactoredPreconditioner against the canonical,
file-based EkfacApplicator.

The Attributor uses FactoredPreconditioner to apply a factored (EKFAC) inverse
Hessian to query/index gradients in a single process. It must produce exactly
what the distributed pipeline's EkfacApplicator writes to disk for the same
factors and the same gradients — this test pins that equivalence for every
inversion mode.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from bergson.collector.collector import create_projection_matrix
from bergson.config import InversionConfig
from bergson.data import create_index, load_module_gradients
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.hessians.inversion import INVERSIONS
from bergson.hessians.preconditioner import FactoredPreconditioner
from bergson.hessians.sharded_computation import shard_bounds


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("inversion", INVERSIONS)
def test_factored_matches_ekfac_applicator(
    ekfac_results_path: str, tmp_path, inversion: str
):
    hessian_path = ekfac_results_path

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

    # factored_tikhonov assumes the eigenvalue grid factorizes as λ = λ_G ⊗ λ_A,
    # which the EV-corrected eigenvalues do not; apply it to the uncorrected grid.
    ev_correction = inversion != "factored_tikhonov"

    # Reference: the pipeline's file-based applicator (full inverse).
    ref = _apply(
        hessian_path,
        query_path,
        str(tmp_path / f"out_{inversion}"),
        inversion,
        ev_correction=ev_correction,
    )
    reference = {
        name: torch.from_numpy(np.asarray(ref[name][:])) for name in grad_sizes
    }

    # In-memory query gradients, matching what the Attributor holds.
    src = load_module_gradients(query_path)
    grads = {
        name: torch.from_numpy(np.asarray(src[name][:])).float() for name in grad_sizes
    }

    pre = FactoredPreconditioner.from_path(
        hessian_path,
        inversion_cfg=InversionConfig(inversion=inversion, damping_factor=0.1),
        power=-1.0,
        ev_correction=ev_correction,
        device="cuda",
    )
    got = pre.apply(grads)

    for name in grad_sizes:
        assert torch.allclose(
            got[name].cpu(), reference[name], atol=1e-4, rtol=1e-4
        ), f"{inversion}: factored disagrees with EkfacApplicator on {name}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_two_sided_factored_equals_full_inverse(ekfac_results_path: str, tmp_path):
    """Applying H^{-1/2} twice must equal applying H^{-1} once."""
    hessian_path = ekfac_results_path
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
    src = load_module_gradients(query_path)
    grads = {
        name: torch.from_numpy(np.asarray(src[name][:])).float() for name in grad_sizes
    }

    cfg = InversionConfig(inversion="damped_inverse", damping_factor=0.1)
    full = FactoredPreconditioner.from_path(
        hessian_path, inversion_cfg=cfg, power=-1.0, ev_correction=True, device="cuda"
    )
    half = FactoredPreconditioner.from_path(
        hessian_path, inversion_cfg=cfg, power=-0.5, ev_correction=True, device="cuda"
    )

    once = full.apply(grads)
    twice = half.apply(half.apply(grads))

    for name in grad_sizes:
        assert torch.allclose(
            twice[name].cpu(), once[name].cpu(), atol=1e-4, rtol=1e-4
        ), f"H^-1/2 applied twice != H^-1 on {name}"


def _write_factored_hessian(
    path: Path, modules: dict[str, tuple[int, int]], num_shards: int, seed: int = 0
) -> None:
    """Write a synthetic factored Hessian to ``path`` split into ``num_shards``.

    Matches the on-disk layout of :mod:`bergson.hessians.eigenvectors`:
    ``eigen_activation_sharded`` (Q_A ``[I, I]``), ``eigen_gradient_sharded``
    (Q_G ``[O, O]``), and ``eigenvalue_sharded`` (λ grid ``[O, I]``) are row-
    sharded along dim 0; ``factor_eig_g`` (λ_G ``[O]``) is sharded along O;
    ``factor_eig_a`` (λ_A ``[I]``) is **replicated** — the full vector in every
    shard. The same ``seed`` reproduces identical full factors at any shard count.
    """
    g = torch.Generator().manual_seed(seed)
    subdirs = [
        "eigen_activation_sharded",
        "eigen_gradient_sharded",
        "eigenvalue_sharded",
        "factor_eig_a",
        "factor_eig_g",
    ]
    per_shard: dict[str, list[dict[str, torch.Tensor]]] = {
        sub: [{} for _ in range(num_shards)] for sub in subdirs
    }
    for name, (o, i) in modules.items():
        q_a = torch.randn(i, i, generator=g)
        q_g = torch.randn(o, o, generator=g)
        lam_a = torch.rand(i, generator=g) + 0.1
        lam_g = torch.rand(o, generator=g) + 0.1
        grid = torch.outer(lam_g, lam_a)  # [O, I]
        for r in range(num_shards):
            ia, ib = shard_bounds(i, r, num_shards)
            oa, ob = shard_bounds(o, r, num_shards)
            per_shard["eigen_activation_sharded"][r][name] = q_a[ia:ib].contiguous()
            per_shard["eigen_gradient_sharded"][r][name] = q_g[oa:ob].contiguous()
            per_shard["eigenvalue_sharded"][r][name] = grid[oa:ob].contiguous()
            per_shard["factor_eig_g"][r][name] = lam_g[oa:ob].contiguous()
            per_shard["factor_eig_a"][r][name] = lam_a.contiguous()  # replicated

    for sub, shards in per_shard.items():
        d = path / sub
        d.mkdir(parents=True, exist_ok=True)
        for r in range(num_shards):
            save_file(shards[r], str(d / f"shard_{r}.safetensors"))


def test_factored_loader_factor_a_replicated_not_concatenated(tmp_path):
    """factor_eig_a is replicated across shards — loading must not concatenate it.

    With N shards, naively concatenating would give ``[N*I]`` and break the
    factored-Tikhonov multiplier; it must stay ``[I]``.
    """
    modules = {"layer": (4, 6)}  # (O, I)
    _write_factored_hessian(tmp_path, modules, num_shards=3)

    pre = FactoredPreconditioner.from_path(
        tmp_path,
        inversion_cfg=InversionConfig(inversion="factored_tikhonov"),
        device="cpu",
    )
    assert pre.factor_eig_a["layer"].shape == (6,)  # λ_A [I], not [3*I]
    assert pre.factor_eig_g["layer"].shape == (4,)  # λ_G [O]


@pytest.mark.parametrize("inversion", INVERSIONS)
def test_factored_apply_invariant_to_shard_count(tmp_path, inversion: str):
    """from_path apply must be identical whether the Hessian is 1 shard or 2."""
    # Uneven dims exercise shard_bounds' remainder handling.
    modules = {"a": (4, 6), "b": (5, 3)}
    _write_factored_hessian(tmp_path / "one", modules, num_shards=1, seed=0)
    _write_factored_hessian(tmp_path / "two", modules, num_shards=2, seed=0)

    cfg = InversionConfig(inversion=inversion, damping_factor=0.1)
    pre1 = FactoredPreconditioner.from_path(
        tmp_path / "one", inversion_cfg=cfg, device="cpu"
    )
    pre2 = FactoredPreconditioner.from_path(
        tmp_path / "two", inversion_cfg=cfg, device="cpu"
    )

    rng = torch.Generator().manual_seed(1)
    grads = {
        name: torch.randn(3, o * i, generator=rng) for name, (o, i) in modules.items()
    }
    out1 = pre1.apply({k: v.clone() for k, v in grads.items()})
    out2 = pre2.apply({k: v.clone() for k, v in grads.items()})

    for name in modules:
        assert torch.allclose(
            out1[name], out2[name], atol=1e-5
        ), f"{inversion}: apply on {name} depends on shard count"


def test_apply_hessian_compresses_per_module(tmp_path):
    """``EkfacConfig.projection_dim > 0`` compresses the IVHP output to a
    ``[p, p]`` per-module Kronecker random projection, applied *after* H^-1
    on the full gradient. It must use the exact same
    ``create_projection_matrix`` convention (identifiers ``f"{name}/left"``
    / ``f"{name}/right"``) that ``bergson build`` uses on training
    gradients, so a compressed query is directly comparable to a compressed
    training-side gradient store.
    """
    modules = {"a": (4, 6), "b": (5, 3)}  # (O, I)
    hessian_path = tmp_path / "hessian"
    _write_factored_hessian(hessian_path, modules, num_shards=1, seed=0)

    grad_sizes = {name: o * i for name, (o, i) in modules.items()}
    num_grads = 3
    query_path = str(tmp_path / "query")
    _make_query_gradients(query_path, grad_sizes, num_grads)

    inversion_cfg = InversionConfig(inversion="damped_inverse", damping_factor=0.1)

    # Reference: full, uncompressed IVHP output.
    ref = _apply(
        str(hessian_path),
        query_path,
        str(tmp_path / "out_full"),
        "damped_inverse",
        ev_correction=False,
    )

    p = 2
    compressed_cfg = EkfacConfig(
        hessian_method_path=str(hessian_path),
        gradient_path=query_path,
        run_path=str(tmp_path / "out_compressed"),
        ev_correction=False,
        projection_dim=p,
    )
    EkfacApplicator(compressed_cfg, inversion_cfg=inversion_cfg).compute_ivhp_sharded()
    got = load_module_gradients(str(tmp_path / "out_compressed"))

    for name, (o, i) in modules.items():
        full = torch.from_numpy(np.asarray(ref[name][:])).view(num_grads, o, i)
        P_l = create_projection_matrix(
            f"{name}/left", p, o, full.dtype, full.device, "rademacher"
        )
        P_r = create_projection_matrix(
            f"{name}/right", p, i, full.dtype, full.device, "rademacher"
        )
        expected = torch.einsum("ps,nsa,ra->npr", P_l, full, P_r)

        compressed = torch.from_numpy(np.asarray(got[name][:])).view(num_grads, p, p)
        assert compressed.shape == (num_grads, p, p)
        assert torch.allclose(
            compressed, expected, atol=1e-4, rtol=1e-4
        ), f"{name}: compressed IVHP output doesn't match manual post-projection"


def test_apply_hessian_rejects_compression_with_ev_correction():
    """projection_dim compression is not supported with EK-FAC (ev_correction)."""
    cfg = EkfacConfig(
        hessian_method_path="unused",
        gradient_path="unused",
        run_path="unused",
        ev_correction=True,
        projection_dim=8,
    )
    with pytest.raises(ValueError, match="EK-FAC"):
        EkfacApplicator(cfg, inversion_cfg=InversionConfig())


def test_apply_hessian_batches_queries(tmp_path):
    """The default ``apply_batch_size`` is small enough for large models, and
    batching the queries through the EK-FAC inverse (ev_correction, the path
    that also holds the corrected eigenvalue grid on-device) must give the same
    output as applying them all at once."""
    assert InversionConfig().apply_batch_size == 2

    modules = {"a": (4, 6), "b": (5, 3)}  # (O, I)
    hessian_path = tmp_path / "hessian"
    _write_factored_hessian(hessian_path, modules, num_shards=1, seed=0)
    # EK-FAC applies the corrected eigenvalue grid; derive one from the grid.
    grid = load_file(str(hessian_path / "eigenvalue_sharded" / "shard_0.safetensors"))
    corrected_dir = hessian_path / "eigenvalue_correction_sharded"
    corrected_dir.mkdir()
    save_file(
        {k: v * 1.5 for k, v in grid.items()},
        str(corrected_dir / "shard_0.safetensors"),
    )
    grad_sizes = {name: o * i for name, (o, i) in modules.items()}
    num_grads = 5
    query_path = str(tmp_path / "query")
    _make_query_gradients(query_path, grad_sizes, num_grads)
    inversion_cfg = InversionConfig(inversion="damped_inverse", damping_factor=0.1)

    outs = []
    for label, batch_size in [("default", None), ("all", num_grads)]:
        kwargs = {} if batch_size is None else {"apply_batch_size": batch_size}
        cfg = EkfacConfig(
            hessian_method_path=str(hessian_path),
            gradient_path=query_path,
            run_path=str(tmp_path / f"out_{label}"),
            ev_correction=True,
            **kwargs,
        )
        assert cfg.apply_batch_size == (batch_size or 2)
        EkfacApplicator(cfg, inversion_cfg=inversion_cfg).compute_ivhp_sharded()
        outs.append(load_module_gradients(str(tmp_path / f"out_{label}")))

    for name in modules:
        batched = torch.from_numpy(np.asarray(outs[0][name][:]))
        whole = torch.from_numpy(np.asarray(outs[1][name][:]))
        assert batched.shape[0] == num_grads
        assert torch.allclose(batched, whole, atol=1e-6), name
