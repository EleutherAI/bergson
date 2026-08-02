"""Unit tests for the eigenvalue inversion math (CPU, no model or fixture).

Covers the dense ``invert_psd_matrix``, the shared ``eigenvalue_multiplier``
dispatch, and the factored-Tikhonov damping — none of which had a direct
correctness test before (they were only exercised end-to-end / via the sharded
path).
"""

import torch

from bergson.hessians.inversion import (
    eigenvalue_multiplier,
    factored_tikhonov_damped,
    invert_psd_matrix,
    rank_eigenvalue_threshold,
)


def _psd(n: int, seed: int = 0) -> torch.Tensor:
    """A well-conditioned positive-definite ``[n, n]`` matrix in fp64."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=torch.float64)
    return a @ a.T + n * torch.eye(n, dtype=torch.float64)


# ── invert_psd_matrix (dense) ──────────────────────────────────────────────


def test_damped_inverse_zero_damping_is_true_inverse():
    """With no damping, the damped inverse is the exact matrix inverse."""
    h = _psd(6)
    h_inv = invert_psd_matrix(h, "damped_inverse", damping_factor=0.0)
    assert torch.allclose(h_inv @ h, torch.eye(6, dtype=torch.float64), atol=1e-8)


def test_half_power_squared_is_full_inverse():
    """H^(-1/2) applied twice equals H^(-1) (dense path)."""
    h = _psd(6)
    half = invert_psd_matrix(h, power=-0.5, damping_factor=0.1)
    full = invert_psd_matrix(h, power=-1.0, damping_factor=0.1)
    assert torch.allclose(half @ half, full, atol=1e-8)


def test_inverse_is_symmetric():
    """The inverse of a symmetric PSD matrix is symmetric."""
    h = _psd(6)
    h_inv = invert_psd_matrix(h, damping_factor=0.1)
    assert torch.allclose(h_inv, h_inv.T, atol=1e-10)


def test_pseudoinverse_matches_torch_pinv_on_rank_deficient():
    """Truncated pseudoinverse equals the Moore-Penrose inverse when the damping
    threshold sits between zero and the smallest nonzero eigenvalue."""
    g = torch.Generator().manual_seed(1)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    eig = torch.tensor([3.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    h = q @ torch.diag(eig) @ q.T

    pinv = invert_psd_matrix(h, "pseudoinverse", damping_factor=1e-9)
    assert torch.allclose(pinv, torch.linalg.pinv(h), atol=1e-7)


def test_factored_tikhonov_falls_back_to_damped_inverse_on_dense():
    """The dense Gram has no Kronecker split, so factored_tikhonov == damped."""
    h = _psd(6)
    factored = invert_psd_matrix(h, "factored_tikhonov", damping_factor=0.1)
    damped = invert_psd_matrix(h, "damped_inverse", damping_factor=0.1)
    assert torch.allclose(factored, damped)


# ── eigenvalue_multiplier (the shared dispatch) ────────────────────────────


def test_damped_inverse_formula():
    lam = torch.rand(5) + 0.1
    mean = lam.mean()
    m = eigenvalue_multiplier("damped_inverse", lam, mean, 0.1)
    assert torch.allclose(m, 1.0 / (lam + 0.1 * mean))


def test_tikhonov_filtered_formula():
    lam = torch.rand(5) + 0.1
    mean = lam.mean()
    m = eigenvalue_multiplier("tikhonov_filtered", lam, mean, 0.1)
    alpha = 0.1 * mean
    assert torch.allclose(m, lam / (lam * lam + alpha * alpha))


def test_pseudoinverse_truncates_below_threshold():
    lam = torch.tensor([1.0, 0.5, 0.001])
    mean = lam.mean()
    tol = 0.1 * mean
    m = eigenvalue_multiplier("pseudoinverse", lam, mean, 0.1)
    expected = torch.where(lam > tol, lam.reciprocal(), torch.zeros_like(lam))
    assert torch.allclose(m, expected)
    assert (m[lam <= tol] == 0).all()  # sub-threshold directions dropped


def test_power_half_is_sqrt_of_full():
    lam = torch.rand(5) + 0.1
    mean = lam.mean()
    full = eigenvalue_multiplier("damped_inverse", lam, mean, 0.1, power=-1.0)
    half = eigenvalue_multiplier("damped_inverse", lam, mean, 0.1, power=-0.5)
    assert torch.allclose(half * half, full)


# ── factored Tikhonov: does the output look sensible? ──────────────────────


def test_factored_tikhonov_grid_factorizes():
    """The damped eigenvalue grid must equal the product of *independently*
    damped Kronecker factors (Martens & Grosse). This checks the additive
    expansion in the code against the intended factored form: damp ``λ_A`` by
    ``π·√(c·mean)`` and ``λ_G`` by ``√(c·mean)/π``, then take the outer product.
    """
    o, i, c = 4, 5, 0.1
    g = torch.Generator().manual_seed(2)
    lam_a = torch.rand(i, generator=g) + 0.1  # λ_A [I]
    lam_g = torch.rand(o, generator=g) + 0.1  # λ_G [O]
    grid = torch.outer(lam_g, lam_a)  # λ grid [O, I]
    mean = grid.mean()

    damped = factored_tikhonov_damped(
        grid, lam_a, lam_g, mean, lam_a.mean(), lam_g.mean(), c
    )

    pi = (lam_a.mean() / lam_g.mean()).sqrt()
    sqrt_lam = (c * mean).sqrt()
    expected = torch.outer(lam_g + sqrt_lam / pi, lam_a + pi * sqrt_lam)
    assert torch.allclose(damped, expected, atol=1e-6)


def test_factored_tikhonov_regularizes():
    """Damping strictly increases the eigenvalues, so the inverse shrinks toward
    zero and stays finite and positive."""
    o, i = 4, 5
    g = torch.Generator().manual_seed(3)
    lam_a = torch.rand(i, generator=g) + 0.1
    lam_g = torch.rand(o, generator=g) + 0.1
    grid = torch.outer(lam_g, lam_a)
    mean = grid.mean()

    damped = factored_tikhonov_damped(
        grid, lam_a, lam_g, mean, lam_a.mean(), lam_g.mean(), 0.1
    )
    assert (damped > grid).all()  # damping increases every eigenvalue

    inv = eigenvalue_multiplier(
        "factored_tikhonov",
        grid,
        mean,
        0.1,
        factor_a=lam_a,
        factor_g=lam_g,
        mean_a=lam_a.mean(),
        mean_g=lam_g.mean(),
    )
    assert torch.isfinite(inv).all() and (inv > 0).all()
    assert (inv < grid.reciprocal()).all()  # regularized inverse < raw 1/λ


# ── rank-based pseudoinverse (Meta distributed_shampoo semantics) ──────────


def test_rank_threshold_default_matches_matrix_rank_convention():
    """rank_rtol=None uses the torch.linalg.matrix_rank / Meta default,
    numel(λ)·machine_eps, scaled by the spectrum max."""
    lam = torch.tensor([4.0, 1.0, 1e-20], dtype=torch.float64)
    tol = rank_eigenvalue_threshold(lam.max(), lam.numel(), None, 0.0)
    expected = lam.numel() * torch.finfo(lam.dtype).eps * lam.max()
    assert torch.allclose(tol, expected)
    # atol acts as an absolute floor over the relative cutoff
    tol_floored = rank_eigenvalue_threshold(lam.max(), lam.numel(), None, 0.5)
    assert tol_floored == 0.5


def test_pseudoinverse_rank_absolute_floor_truncates_then_powers():
    """rank_rtol=0 + rank_atol=eps is the eps-style absolute cutoff: eigenvalues
    at or below eps are zeroed, survivors get λ^p (truncate, then power)."""
    lam = torch.tensor([0.0, 1e-16, 1e-3, 1.0, 4.0], dtype=torch.float64)
    m = eigenvalue_multiplier(
        "pseudoinverse_rank",
        lam,
        lam.mean(),
        0.1,  # damping_factor must be ignored by this mode
        spectrum_max=lam.max(),
        spectrum_numel=lam.numel(),
        rank_rtol=0.0,
        rank_atol=1e-15,
        power=-0.25,
    )
    expected = torch.where(lam > 1e-15, lam.pow(-0.25), torch.zeros_like(lam))
    assert torch.allclose(m, expected)


def test_invert_psd_matrix_pseudoinverse_rank_matches_torch_pinv():
    """With default tolerances, the dense rank-truncated pseudoinverse at
    power=-1 equals torch.linalg.pinv on a rank-deficient PSD matrix."""
    g = torch.Generator().manual_seed(5)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=g, dtype=torch.float64))
    eig = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0, 0.0], dtype=torch.float64)
    h = q @ torch.diag(eig) @ q.T

    pinv = invert_psd_matrix(h, "pseudoinverse_rank")
    assert torch.allclose(pinv, torch.linalg.pinv(h, hermitian=True), atol=1e-7)


def test_pseudoinverse_factored_equals_per_side_truncated_roots():
    """The factored multiplier must equal the outer product of the two factors'
    independently truncated inverse roots — distributed_shampoo's
    L^{†p} G R^{†p} — including cells whose *product* is large but where one
    factor eigenvalue falls below its own cutoff."""
    lam_a = torch.tensor([2.0, 1.0, 1e-18], dtype=torch.float64)  # λ_A [I]
    lam_g = torch.tensor([4.0, 1e-18, 0.5], dtype=torch.float64)  # λ_G [O]
    grid = torch.outer(lam_g, lam_a)
    power = -0.25
    rtol, atol = 0.0, 1e-15

    m = eigenvalue_multiplier(
        "pseudoinverse_factored",
        grid,
        grid.mean(),
        0.1,  # damping_factor must be ignored by this mode
        factor_a=lam_a,
        factor_g=lam_g,
        max_a=lam_a.max(),
        max_g=lam_g.max(),
        numel_a=lam_a.numel(),
        numel_g=lam_g.numel(),
        rank_rtol=rtol,
        rank_atol=atol,
        power=power,
    )

    def side(lam):
        return torch.where(lam > atol, lam.pow(power), torch.zeros_like(lam))

    assert torch.allclose(m, torch.outer(side(lam_g), side(lam_a)))
    # The joint-grid mode would keep e.g. cell (0, 2): 4.0·1e-18 > 0 product
    # semantics differ — per-side must zero the whole row/column instead.
    assert (m[1, :] == 0).all() and (m[:, 2] == 0).all()


def test_pseudoinverse_factored_dense_fallback_is_rank_mode():
    """The dense Gram has no Kronecker split, so pseudoinverse_factored falls
    back to pseudoinverse_rank."""
    h = _psd(6)
    factored = invert_psd_matrix(h, "pseudoinverse_factored", rank_atol=1e-12)
    rank = invert_psd_matrix(h, "pseudoinverse_rank", rank_atol=1e-12)
    assert torch.allclose(factored, rank)


def test_factored_tikhonov_zero_damping_is_raw_inverse():
    """With no damping the factored inverse is just 1/λ over the grid."""
    o, i = 4, 5
    g = torch.Generator().manual_seed(4)
    lam_a = torch.rand(i, generator=g) + 0.1
    lam_g = torch.rand(o, generator=g) + 0.1
    grid = torch.outer(lam_g, lam_a)
    inv = eigenvalue_multiplier(
        "factored_tikhonov",
        grid,
        grid.mean(),
        0.0,
        factor_a=lam_a,
        factor_g=lam_g,
        mean_a=lam_a.mean(),
        mean_g=lam_g.mean(),
    )
    assert torch.allclose(inv, grid.reciprocal())
