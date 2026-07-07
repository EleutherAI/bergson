"""Eigenvalue inversion math for the Hessian.

Two representations of the Hessian coexist in the library:

- the factored EKFAC family (kfac / tkfac / shampoo), where the Hessian is a
  Kronecker product applied via the eigenbasis rotation in
  :class:`~bergson.hessians.preconditioner.FactoredPreconditioner`, and
- the dense autocorrelation Gram, where :func:`invert_psd_matrix`
  eigendecomposes a per-module ``[d, d]`` matrix.

:func:`eigenvalue_multiplier` regularizes and inverts the eigenvalues for both.
Writing ``c = damping_factor`` and ``λ`` for an eigenvalue, with ``mean(λ)`` the
mean over the whole spectrum:

- ``damped_inverse``: ``1 / (λ + c·mean(λ))`` — uniform Tikhonov damping.
- ``tikhonov_filtered``: ``λ / (λ² + α²)`` with ``α = c·mean(λ)`` (the Tikhonov
  filter factor; formerly named ``cauchy`` for its Lorentzian shape).
- ``pseudoinverse``: ``1/λ`` where ``λ > c·mean(λ)``, else ``0``.
- ``factored_tikhonov``: damping split across the Kronecker activation (A) and
  gradient (G) factors (see :func:`factored_tikhonov_damped`). This has no dense
  analogue (there is no A/G split for a dense Gram), so on the dense path it
  falls back to ``damped_inverse``.
"""

from typing import Callable, Literal

import torch
from torch import Tensor

Inversion = Literal[
    "damped_inverse", "factored_tikhonov", "pseudoinverse", "tikhonov_filtered"
]
"""Eigenvalue function used to invert a Hessian / preconditioner matrix."""

INVERSIONS: tuple[Inversion, ...] = (
    "damped_inverse",
    "factored_tikhonov",
    "pseudoinverse",
    "tikhonov_filtered",
)


def damped_inverse_eigfn(
    eigvals: Tensor, mean: Tensor, damping_factor: float
) -> Tensor:
    """Uniform Tikhonov damped inverse
    ``Let λ = eigenvals, c = damping_factor.``
    ``1 / (λ + c·mean(λ))``.

    ``eigvals`` may be only this rank's shard of the spectrum, so ``mean`` is
    passed in (the globally-reduced ``mean(λ)``) rather than recomputed here — a
    local ``eigvals.mean()`` would be the per-shard mean, not the global one.
    """
    return (eigvals + damping_factor * mean).reciprocal()


def tikhonov_filtered_eigfn(
    eigenvals: Tensor, mean: Tensor, damping_factor: float
) -> Tensor:
    """Tikhonov-filtered inverse (the Tikhonov filter factor)
    ``Let λ = eigenvals, c = damping_factor.``
    ``λ / (λ² + α²)``, ``α = c·mean(λ)``.

    This is the per-eigenvalue action of the ridge / Tikhonov-regularized
    least-squares solution ``(H² + α²I)⁻¹H`` — the standard filter factor for
    regularizing ill-posed inverse problems. Unlike ``1/(λ + α)`` it decays to 0
    as ``λ → 0`` rather than saturating at ``1/α``.

    ``eigenvals`` may be a shard, so the
    globally-reduced ``mean`` is passed in rather than recomputed.
    """
    alpha_sq = (damping_factor * mean) ** 2
    return eigenvals / (eigenvals * eigenvals + alpha_sq)


def pseudoinverse_eigfn(
    eigenvals: Tensor, mean: Tensor, damping_factor: float
) -> Tensor:
    """Truncated pseudoinverse
    ``Let λ = eigenvals, c = damping_factor.``
    ``1/λ`` for ``λ > c·mean(λ)``, else ``0``.

    ``eigenvals`` may be a shard, so the
    globally-reduced ``mean`` is passed in rather than recomputed."""
    tol = damping_factor * mean
    return torch.where(
        eigenvals > tol, eigenvals.reciprocal(), torch.zeros_like(eigenvals)
    )


# Inversion modes expressible as a pure function of ``(λ, mean(λ), c)``. These
# operate elementwise, so they apply unchanged to the dense 1-D spectrum and to
# the EKFAC eigenvalue grid. ``factored_tikhonov`` is intentionally absent: it
# needs the per-factor A/G eigenvalues and only exists for the factored path.
EIGENFNS: dict[str, Callable[[Tensor, Tensor, float], Tensor]] = {
    "damped_inverse": damped_inverse_eigfn,
    "tikhonov_filtered": tikhonov_filtered_eigfn,
    "pseudoinverse": pseudoinverse_eigfn,
}


def factored_tikhonov_damped(
    eigvals: Tensor,
    factor_a: Tensor,
    factor_g: Tensor,
    mean: Tensor,
    mean_a: Tensor,
    mean_g: Tensor,
    damping_factor: float,
) -> Tensor:
    """The factored-Tikhonov *damped eigenvalues* (before inversion).

    Splits the damping ``c·mean(λ)`` across the Kronecker activation (A) and
    gradient (G) factors via the Martens & Grosse trace ratio
    ``π = sqrt(mean(λ_A) / mean(λ_G))``::

        λ + π·√(c·mean(λ))·λ_G + (√(c·mean(λ))/π)·λ_A + c·mean(λ)

    ``eigvals`` is the EKFAC eigenvalue grid ``λ`` (= ``λ_G ⊗ λ_A``); ``factor_a``
    / ``factor_g`` are the per-factor eigenvalues ``λ_A`` ``[I]`` / ``λ_G`` ``[O]``.
    All means are passed in so the distributed path can supply globally-reduced
    values rather than per-shard ones. Reciprocate the result for the inverse.
    """
    eps = torch.finfo(eigvals.dtype).tiny
    lambda_abs = damping_factor * mean
    sqrt_lambda = lambda_abs.clamp_min(0).sqrt()
    pi = (mean_a / mean_g.clamp_min(eps)).clamp_min(eps).sqrt()
    return (
        eigvals
        + (pi * sqrt_lambda) * factor_g.unsqueeze(1)
        + (sqrt_lambda / pi) * factor_a.unsqueeze(0)
        + lambda_abs
    )


def eigenvalue_multiplier(
    inversion: Inversion,
    eigvals: Tensor,
    mean: Tensor,
    damping_factor: float,
    *,
    factor_a: Tensor | None = None,
    factor_g: Tensor | None = None,
    mean_a: Tensor | None = None,
    mean_g: Tensor | None = None,
    power: float = -1.0,
) -> Tensor:
    """The elementwise inverse multiplier ``m(λ)``.

    Maps an ``inversion`` mode (and matrix ``power``) to the factor that scales a
    Hessian's eigenvalues toward its inverse.

    ``power`` selects how much inverse: ``-1`` for the full inverse ``m ≈ 1/λ``,
    ``-0.5`` for the matrix square-root inverse (the multiplier is raised to
    ``-power`` so that applying it twice recovers the full inverse).

    ``factored_tikhonov`` requires the per-factor eigenvalues and their means; the
    other modes are pure functions of ``(λ, mean, c)``.
    """
    if inversion == "factored_tikhonov":
        assert (
            factor_a is not None
            and factor_g is not None
            and mean_a is not None
            and mean_g is not None
        ), "factored_tikhonov needs the per-factor eigenvalues and their means"
        multiplier = factored_tikhonov_damped(
            eigvals, factor_a, factor_g, mean, mean_a, mean_g, damping_factor
        ).reciprocal()
    else:
        multiplier = EIGENFNS[inversion](eigvals, mean, damping_factor)

    frac = -power
    if frac != 1.0:
        multiplier = multiplier.clamp_min(0).pow(frac)
    return multiplier


def invert_psd_matrix(
    H: Tensor,
    inversion: Inversion = "damped_inverse",
    damping_factor: float = 0.1,
    power: float = -1.0,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Dense regularized inverse power of a p.s.d. matrix H via eigendecomposition.

    Returns ``H`` rotated, scaled by the chosen ``inversion`` eigenvalue
    function, and rotated back. ``power`` selects how much inverse to apply:

    - ``-1.0`` — full inverse ``H^{-1}`` (one-sided preconditioning), and
    - ``-0.5`` — matrix square-root inverse ``H^{-1/2}`` (split/two-sided
      preconditioning, applied to both query and index gradients).

    The multiplier comes from :func:`eigenvalue_multiplier` (shared with the
    factored path); for ``power = -0.5`` it is raised to ``0.5`` so that applying
    the result twice recovers the full inverse.

    ``factored_tikhonov`` has no dense analogue (there is no Kronecker A/G split
    for a dense Gram) and falls back to ``damped_inverse`` here.
    """
    original_dtype = H.dtype
    H = H.to(dtype=dtype)

    eigvals, eigvecs = torch.linalg.eigh(H)
    # `eigh` emits small negative eigenvalues for a numerically p.s.d. Gram;
    # clamp so the regularized multipliers stay non-negative (and the power
    # below is real-valued).
    eigvals = eigvals.clamp_min(0)

    dense_inversion = (
        "damped_inverse" if inversion == "factored_tikhonov" else inversion
    )
    scaled = eigenvalue_multiplier(
        dense_inversion, eigvals, eigvals.mean(), damping_factor, power=power
    )

    return (eigvecs * scaled @ eigvecs.mH).to(original_dtype)
