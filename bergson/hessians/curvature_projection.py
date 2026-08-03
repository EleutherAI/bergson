"""Curvature-folded random projections for factored (K-FAC family) Hessians.

The scoring pipeline compresses gradients with a per-module Kronecker random
projection ``P_S (.) P_A`` (see :func:`bergson.collector.collector.create_projection_matrix`).
K-FAC influence additionally preconditions the query by the inverse Hessian,
``H^-1 G_q = S^-1 G_q A^-1``. Applying the inverse and *then* projecting
(``P_S (H^-1 G_q) P_A^T``) materializes the full ``[O, I]`` per-module matrix.

Under **factored (Martens-Grosse) Tikhonov damping** the damped K-FAC inverse is
*separable*: the damped eigenvalue grid factors exactly as ``u (x) v`` with

    u = lambda_G + pi * sqrt(c * mean)      (output / gradient side, [O])
    v = lambda_A + sqrt(c * mean) / pi      (input / activation side, [I])

so ``H^power = S^power (x) A^power`` with ``S^power = Q_G diag(u^power) Q_G^T`` and
``A^power = Q_A diag(v^power) Q_A^T``. The curvature can then be *folded into the
projection matrices* once, offline:

    L = P_S @ S^power      [p, O]
    R = A^power @ P_A^T     [I, p]

and the compressed, curvature-corrected query is a single ``L @ G_q @ R``. For
``power = -1`` (one-sided: full inverse on the query, ``L @ G_q @ R`` equals
``P_S (H^-1 G_q) P_A^T`` exactly) this is a pure reassociation of the apply-time
path. For ``power = -1/2`` the same factors whiten *both* the query and the
training gradients symmetrically (two-sided), which is what a curvature-aware
cosine similarity (``unit_normalize``) needs -- that path is a follow-up.

The separability holds *only* under ``factored_tikhonov``; the default
``damped_inverse`` uses the non-separable grid ``1 / (lambda_G (x) lambda_A + c)``,
for which no such ``S^power (x) A^power`` split exists.
"""

import os
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from bergson.collector.collector import create_projection_matrix
from bergson.config import InversionConfig
from bergson.hessians.preconditioner import FactoredPreconditioner
from bergson.utils.logger import get_logger

CURVATURE_PROJECTION_SUBDIR = "curvature_projection"

logger = get_logger("CurvatureProjection")


def factored_side_eigenvalues(
    factor_eig_g: Tensor,
    factor_eig_a: Tensor,
    grid_mean: Tensor | float,
    damping_factor: float,
) -> tuple[Tensor, Tensor]:
    """Per-side damped eigenvalues ``(u, v)`` of the factored-Tikhonov inverse.

    ``u`` ``[O]`` damps the gradient (output) factor ``lambda_G`` and ``v`` ``[I]``
    the activation (input) factor ``lambda_A``. Their outer product ``u (x) v``
    equals :func:`bergson.hessians.inversion.factored_tikhonov_damped` evaluated on
    the eigenvalue grid ``lambda_G (x) lambda_A``; this is the separable form the
    fold relies on. ``grid_mean`` is ``mean(lambda_G (x) lambda_A)`` -- pass the
    same globally-reduced value the preconditioner uses.
    """
    eps = torch.finfo(factor_eig_g.dtype).tiny
    mean_a = factor_eig_a.clamp_min(0).mean()
    mean_g = factor_eig_g.clamp_min(0).mean()
    lambda_abs = damping_factor * grid_mean
    sqrt_lambda = (
        lambda_abs.clamp_min(0).sqrt()
        if isinstance(lambda_abs, Tensor)
        else max(lambda_abs, 0.0) ** 0.5
    )
    pi = (mean_a / mean_g.clamp_min(eps)).clamp_min(eps).sqrt()
    u = factor_eig_g + pi * sqrt_lambda
    v = factor_eig_a + sqrt_lambda / pi
    return u, v


def side_curvature_matrix(eigvecs: Tensor, damped_eig: Tensor, power: float) -> Tensor:
    """``Q diag(damped_eig^power) Q^T`` -- one Kronecker side of ``H^power``.

    ``eigvecs`` is ``Q`` ``[d, d]`` (columns are eigenvectors); ``damped_eig`` is
    the ``[d]`` per-side damped eigenvalue vector from
    :func:`factored_side_eigenvalues`.
    """
    scaled = damped_eig.clamp_min(0).pow(power)
    return (eigvecs * scaled) @ eigvecs.T


def compute_folded_projections(
    preconditioner: FactoredPreconditioner,
    *,
    projection_dim: int,
    projection_type: Literal["normal", "rademacher"] = "rademacher",
    projection_scale: Literal["jl", "row_norm"] = "jl",
    power: float = -1.0,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Build ``{name: (L, R)}`` curvature-folded projection factors.

    ``L = P_S @ S^power`` ``[p, O]`` and ``R = A^power @ P_A^T`` ``[I, p]`` per
    module, using the same seeded ``P_S`` / ``P_A`` the collector regenerates.
    Requires a ``factored_tikhonov`` preconditioner (per-factor eigenvalues loaded).
    """
    if preconditioner.inversion_cfg.inversion != "factored_tikhonov":
        raise ValueError(
            "compute_folded_projections requires factored_tikhonov inversion; the "
            f"default '{preconditioner.inversion_cfg.inversion}' grid inverse is not "
            "separable into per-side S^power (x) A^power factors."
        )
    if not preconditioner.factor_eig_a or not preconditioner.factor_eig_g:
        raise ValueError(
            "preconditioner is missing per-factor eigenvalues (factor_eig_a / "
            "factor_eig_g); build it with inversion_cfg.inversion='factored_tikhonov'."
        )

    p = projection_dim
    c = preconditioner.inversion_cfg.damping_factor
    out: dict[str, tuple[Tensor, Tensor]] = {}
    for name, q_a in preconditioner.eigen_a.items():
        q_g = preconditioner.eigen_g[name]
        o, i = q_g.shape[1], q_a.shape[1]
        grid_mean = preconditioner.lambdas[name].mean()
        u, v = factored_side_eigenvalues(
            preconditioner.factor_eig_g[name],
            preconditioner.factor_eig_a[name],
            grid_mean,
            c,
        )
        s_power = side_curvature_matrix(q_g, u, power)  # [O, O]
        a_power = side_curvature_matrix(q_a, v, power)  # [I, I]

        p_s = create_projection_matrix(
            f"{name}/left", p, o, q_g.dtype, q_g.device, projection_type, projection_scale
        )
        p_a = create_projection_matrix(
            f"{name}/right", p, i, q_a.dtype, q_a.device, projection_type, projection_scale
        )
        out[name] = (p_s @ s_power, a_power @ p_a.T)
    return out


def apply_folded_projection(
    grads: dict[str, Tensor], factors: dict[str, tuple[Tensor, Tensor]]
) -> dict[str, Tensor]:
    """Compress each ``[n, O, I]`` gradient to ``[n, p, p]`` via ``L @ G @ R``.

    Modules absent from ``factors`` pass through unchanged.
    """
    out: dict[str, Tensor] = {}
    for name, g in grads.items():
        if name not in factors:
            out[name] = g
            continue
        left, right = factors[name]
        out[name] = torch.einsum("ps,nsa,ar->npr", left, g, right)
    return out


def save_folded_projections(
    factors: dict[str, tuple[Tensor, Tensor]], out_dir: str | os.PathLike
) -> None:
    """Write ``{name: (L, R)}`` to ``out_dir`` as a single safetensors file with
    ``"{name}/left"`` / ``"{name}/right"`` keys (contiguous, on CPU)."""
    os.makedirs(out_dir, exist_ok=True)
    flat: dict[str, Tensor] = {}
    for name, (left, right) in factors.items():
        flat[f"{name}/left"] = left.detach().to("cpu").contiguous()
        flat[f"{name}/right"] = right.detach().to("cpu").contiguous()
    save_file(flat, os.path.join(str(out_dir), "factors.safetensors"))


def load_folded_projections(
    factors_dir: str | os.PathLike, device: str | torch.device = "cpu"
) -> dict[str, tuple[Tensor, Tensor]]:
    """Inverse of :func:`save_folded_projections`."""
    flat = load_file(os.path.join(str(factors_dir), "factors.safetensors"), device=str(device))
    names = sorted({k.rsplit("/", 1)[0] for k in flat})
    return {name: (flat[f"{name}/left"], flat[f"{name}/right"]) for name in names}


def has_folded_projections(method_path: str | os.PathLike) -> bool:
    """Whether curvature-folded factors were written under ``method_path``."""
    return (
        Path(method_path) / CURVATURE_PROJECTION_SUBDIR / "factors.safetensors"
    ).is_file()


def build_and_save_folded_projections(
    method_path: str | os.PathLike,
    *,
    projection_dim: int,
    projection_type: Literal["normal", "rademacher"] = "rademacher",
    projection_scale: Literal["jl", "row_norm"] = "jl",
    inversion_cfg: InversionConfig,
    power: float = -1.0,
    device: str | torch.device = "cpu",
) -> str:
    """Load the factored Hessian at ``method_path``, fold curvature into its
    projection matrices, and save them under ``method_path/curvature_projection``.

    Returns the directory the factors were written to.
    """
    preconditioner = FactoredPreconditioner.from_path(
        str(method_path), inversion_cfg=inversion_cfg, power=power, device=device
    )
    factors = compute_folded_projections(
        preconditioner,
        projection_dim=projection_dim,
        projection_type=projection_type,
        projection_scale=projection_scale,
        power=power,
    )
    out_dir = os.path.join(str(method_path), CURVATURE_PROJECTION_SUBDIR)
    save_folded_projections(factors, out_dir)
    logger.info(
        "Saved curvature-folded projections for %d modules to %s",
        len(factors),
        out_dir,
    )
    return out_dir
