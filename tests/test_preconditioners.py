"""Parity tests for the factored preconditioners.

EKFAC is checked against the trusted
:meth:`bergson.hessians.apply_hessian.EkfacApplicator.compute_ivhp_sharded`
reference implementation: build synthetic factor shards on disk, run the
applicator to produce an output gradient buffer, run the new
:class:`~bergson.preconditioners.EkfacPreconditioner` on the same input,
and compare element-wise.

KFAC has no existing trusted applicator, so we check against a direct
einsum implementation of the KFAC rotate-scale-rotate (Λ = Λ_G ⊗ Λ_A).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from bergson.data import create_index, load_gradients
from bergson.hessians.apply_hessian import EkfacApplicator, EkfacConfig
from bergson.config import PreprocessConfig
from bergson.preconditioners import (
    AutocorrelationPreconditioner,
    EkfacPreconditioner,
    KfacPreconditioner,
    _detect_variant,
    is_factored_preconditioner,
    load_preconditioner,
)


_MODULE_NAME = "layers.0.mlp.gate_proj"


def _random_orthogonal(d: int, seed: int) -> torch.Tensor:
    """Draw a random d×d orthogonal matrix via QR."""
    gen = torch.Generator().manual_seed(seed)
    m = torch.randn(d, d, generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(m)
    return q.contiguous()


def _write_ekfac_factors(
    path: Path,
    q_a: torch.Tensor,
    q_s: torch.Tensor,
    lam: torch.Tensor,
    module_name: str = _MODULE_NAME,
) -> None:
    """Write single-rank EKFAC factor shards into ``path``."""
    for subdir, tensor in [
        ("eigen_activation_sharded", q_a),
        ("eigen_gradient_sharded", q_s),
        ("eigenvalue_correction_sharded", lam),
    ]:
        d = path / subdir
        d.mkdir(parents=True, exist_ok=True)
        save_file({module_name: tensor}, str(d / "shard_0.safetensors"))


def _write_gradient_index(
    path: Path,
    grads: torch.Tensor,
    module_name: str = _MODULE_NAME,
) -> None:
    """Write a single-module, single-rank gradient mmap compatible with
    :func:`bergson.data.load_gradients` and the ``EkfacApplicator`` loader."""
    n, d = grads.shape
    grad_sizes = {module_name: d}
    buf = create_index(
        path, num_grads=n, grad_sizes=grad_sizes, dtype=np.float32,
    )
    buf[module_name][:] = grads.cpu().numpy()
    buf.flush()


def _ekfac_reference(
    g_flat: torch.Tensor,
    q_a: torch.Tensor,
    q_s: torch.Tensor,
    lam: torch.Tensor,
    power: float,
    damp: float,
) -> torch.Tensor:
    """Direct einsum reference for rotate-scale-rotate; used by the KFAC test."""
    flat_leading = g_flat.shape[0]
    O, I = lam.shape
    g = g_flat.to(torch.float32).view(flat_leading, O, I)
    g = torch.einsum("n o a, a i -> n o i", g, q_a)
    g = torch.einsum("n a i, a o -> n o i", g, q_s)
    mean_lam = lam.mean()
    inv_lam = (lam + damp * mean_lam).pow(power)
    g = g * inv_lam
    g = torch.einsum("n a i, o a -> n o i", g, q_s)
    g = torch.einsum("n o a, i a -> n o i", g, q_a)
    return g.reshape(flat_leading, O * I)


# ───────────────────────── variant detection ────────────────────────────────


def test_detect_variant_ekfac(tmp_path):
    for name in (
        "eigen_activation_sharded",
        "eigen_gradient_sharded",
        "eigenvalue_correction_sharded",
    ):
        (tmp_path / name).mkdir()
    assert _detect_variant(tmp_path) == "ekfac"


def test_detect_variant_kfac(tmp_path):
    for name in ("eigen_activation_sharded", "eigen_gradient_sharded"):
        (tmp_path / name).mkdir()
    assert _detect_variant(tmp_path) == "kfac"


def test_detect_variant_autocorrelation(tmp_path):
    # A GradientProcessor dump has preconditioners.pth etc. but no eigen_ dirs.
    (tmp_path / "preconditioners.pth").touch()
    assert _detect_variant(tmp_path) == "autocorrelation"


def test_is_factored_preconditioner(tmp_path):
    """Public helper used by build_worker to gate projection placement."""
    # None / empty path → not factored.
    assert is_factored_preconditioner(PreprocessConfig()) is False
    assert is_factored_preconditioner(PreprocessConfig(preconditioner_path="")) is False

    # Directory with only a GradientProcessor dump → autocorrelation, not factored.
    autocorr_dir = tmp_path / "autocorr"
    autocorr_dir.mkdir()
    (autocorr_dir / "preconditioners.pth").touch()
    assert is_factored_preconditioner(
        PreprocessConfig(preconditioner_path=str(autocorr_dir))
    ) is False

    # KFAC layout (no eigenvalue_correction_sharded/) → factored.
    kfac_dir = tmp_path / "kfac"
    for name in ("eigen_activation_sharded", "eigen_gradient_sharded"):
        (kfac_dir / name).mkdir(parents=True)
    assert is_factored_preconditioner(
        PreprocessConfig(preconditioner_path=str(kfac_dir))
    ) is True

    # EKFAC layout (with correction dir) → factored.
    ekfac_dir = tmp_path / "ekfac"
    for name in (
        "eigen_activation_sharded",
        "eigen_gradient_sharded",
        "eigenvalue_correction_sharded",
    ):
        (ekfac_dir / name).mkdir(parents=True)
    assert is_factored_preconditioner(
        PreprocessConfig(preconditioner_path=str(ekfac_dir))
    ) is True


def test_load_preconditioner_none_is_autocorrelation():
    p = load_preconditioner(None, device=torch.device("cpu"))
    assert isinstance(p, AutocorrelationPreconditioner)
    assert p.h_inv == {}
    # Empty autocorrelation is a no-op — must not mutate the input.
    g = {"m": torch.randn(3, 4)}
    out = p.apply(g)
    assert torch.equal(out["m"], g["m"])


# ───────────────────────── EKFAC parity ─────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ekfac_preconditioner_parity_vs_applicator(tmp_path: Path):
    """EkfacPreconditioner.apply matches EkfacApplicator.compute_ivhp_sharded."""
    torch.manual_seed(7)
    device = torch.device("cuda", 0)

    # Small module: O=4, I=6 ⇒ flat dim 24. Five examples.
    O, I = 4, 6
    num_grads = 5

    q_a = _random_orthogonal(I, seed=1)
    q_s = _random_orthogonal(O, seed=2)
    # Λ must be strictly positive so damping doesn't flip signs in the
    # reference — LambdaCollector always produces non-negative entries.
    lam = torch.rand(O, I, dtype=torch.float32).add_(0.1)
    grads = torch.randn(num_grads, O * I, dtype=torch.float32)

    hessian_path = tmp_path / "hessian"
    grad_path = tmp_path / "gradients"
    out_path = tmp_path / "ivhp"

    _write_ekfac_factors(hessian_path, q_a, q_s, lam)
    _write_gradient_index(grad_path, grads)

    # Reference: trusted applicator. Writes to out_path.
    ekfac_cfg = EkfacConfig(
        hessian_method_path=str(hessian_path),
        gradient_path=str(grad_path),
        run_path=str(out_path),
        lambda_damp_factor=0.1,
    )
    EkfacApplicator(ekfac_cfg).compute_ivhp_sharded()
    ref_buf = load_gradients(out_path)
    ref = torch.from_numpy(
        np.ascontiguousarray(ref_buf[_MODULE_NAME][:])
    ).to(device=device, dtype=torch.float32)

    # Candidate: EkfacPreconditioner.apply via load_preconditioner.
    precond = load_preconditioner(
        hessian_path, device=device, power=-1.0, lambda_damp_factor=0.1,
    )
    assert isinstance(precond, EkfacPreconditioner)
    candidate = precond.apply({_MODULE_NAME: grads.to(device)})[_MODULE_NAME]

    assert candidate.shape == ref.shape, f"{candidate.shape} vs {ref.shape}"
    max_abs = (candidate - ref).abs().max().item()
    assert torch.allclose(candidate, ref, atol=1e-5), (
        f"EKFAC parity mismatch: max abs diff {max_abs}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ekfac_preconditioner_per_token_shape(tmp_path: Path):
    """Per-token input [T, O*I] must round-trip through the same math."""
    torch.manual_seed(11)
    device = torch.device("cuda", 0)
    O, I = 3, 5
    T = 17

    q_a = _random_orthogonal(I, seed=3)
    q_s = _random_orthogonal(O, seed=4)
    lam = torch.rand(O, I, dtype=torch.float32).add_(0.1)

    hessian_path = tmp_path / "hessian"
    _write_ekfac_factors(hessian_path, q_a, q_s, lam)

    precond = load_preconditioner(
        hessian_path, device=device, power=-1.0, lambda_damp_factor=0.1,
    )
    token_grads = torch.randn(T, O * I, dtype=torch.float32, device=device)
    out = precond.apply({_MODULE_NAME: token_grads})[_MODULE_NAME]
    assert out.shape == (T, O * I)

    # Expected: same math as sequences, just with T in the leading dim.
    expected = _ekfac_reference(
        token_grads.cpu(),
        q_a.to(torch.float32),
        q_s.to(torch.float32),
        lam.to(torch.float32),
        power=-1.0,
        damp=0.1,
    ).to(device)
    assert torch.allclose(out, expected, atol=1e-5)


# ───────────────────────── KFAC parity ──────────────────────────────────────


def _write_kfac_factors(
    path: Path,
    q_a: torch.Tensor,
    q_s: torch.Tensor,
    lam_a: torch.Tensor,
    lam_s: torch.Tensor,
    module_name: str = _MODULE_NAME,
) -> None:
    """Write KFAC factor shards: eigenvectors + per-factor eigenvalues,
    no EKFAC correction."""
    for subdir, tensor in [
        ("eigen_activation_sharded", q_a),
        ("eigen_gradient_sharded", q_s),
        ("eigenvalues_activation_sharded", lam_a),
        ("eigenvalues_gradient_sharded", lam_s),
    ]:
        d = path / subdir
        d.mkdir(parents=True, exist_ok=True)
        save_file({module_name: tensor}, str(d / "shard_0.safetensors"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kfac_preconditioner_matches_reference(tmp_path: Path):
    """KFAC with Λ = Λ_G ⊗ Λ_A^T matches the einsum reference."""
    torch.manual_seed(13)
    device = torch.device("cuda", 0)
    O, I = 4, 6
    N = 5

    q_a = _random_orthogonal(I, seed=5)
    q_s = _random_orthogonal(O, seed=6)
    # Per-factor eigenvalues are positive (covariance eigenvalues).
    lam_a = torch.rand(I, dtype=torch.float32).add_(0.1)
    lam_s = torch.rand(O, dtype=torch.float32).add_(0.1)
    grads = torch.randn(N, O * I, dtype=torch.float32)

    hessian_path = tmp_path / "hessian"
    _write_kfac_factors(hessian_path, q_a, q_s, lam_a, lam_s)

    # Detection: no eigenvalue_correction_sharded ⇒ variant = "kfac".
    assert _detect_variant(hessian_path) == "kfac"

    precond = load_preconditioner(
        hessian_path, device=device, power=-1.0, lambda_damp_factor=0.1,
    )
    assert isinstance(precond, KfacPreconditioner)

    candidate = precond.apply({_MODULE_NAME: grads.to(device)})[_MODULE_NAME]

    expected = _ekfac_reference(
        grads,
        q_a,
        q_s,
        torch.outer(lam_s, lam_a),
        power=-1.0,
        damp=0.1,
    ).to(device)
    max_abs = (candidate - expected).abs().max().item()
    assert torch.allclose(candidate, expected, atol=1e-5), (
        f"KFAC parity mismatch: max abs diff {max_abs}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kfac_preconditioner_missing_eigenvalues_errors(tmp_path: Path):
    """KFAC artifact without per-factor eigenvalues raises a clear error."""
    torch.manual_seed(17)
    O, I = 3, 4
    q_a = _random_orthogonal(I, seed=7)
    q_s = _random_orthogonal(O, seed=8)

    hessian_path = tmp_path / "hessian"
    for subdir, tensor in [
        ("eigen_activation_sharded", q_a),
        ("eigen_gradient_sharded", q_s),
    ]:
        d = hessian_path / subdir
        d.mkdir(parents=True, exist_ok=True)
        save_file({_MODULE_NAME: tensor}, str(d / "shard_0.safetensors"))

    with pytest.raises(FileNotFoundError, match="eigenvalue"):
        load_preconditioner(
            hessian_path, device=torch.device("cuda", 0), power=-1.0,
        )
