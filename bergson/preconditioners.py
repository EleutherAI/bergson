"""Preconditioner interface and auto-detect loader.

Three first-class variants of Hessian approximation are supported, each
differing in how ``H^power`` is represented:

* ``autocorrelation`` — unfactored per-module ``Σ vec(g) vec(g)ᵀ`` (what the
  existing ``GradientProcessor`` pipeline stores). The preconditioner is a
  single ``dict[str, Tensor]`` of ``H^power`` matrices; ``apply`` is a per-
  module right-matmul.
* ``kfac`` — factored activation/gradient covariances. ``H^power ≈
  Q_S (Λ_G ⊗ Λ_A)^power Q_S^T ⊗ Q_A (·)^power Q_A^T``; the diagonal ``Λ`` is
  the outer product of the activation and gradient eigenvalues.
* ``ekfac`` — factored + per-element eigenvalue correction. Same rotations as
  KFAC, but ``Λ`` is replaced by an empirical per-element correction
  computed by ``LambdaCollector``.

``load_preconditioner`` auto-detects the variant from directory contents
(see §3.3 of ``COMPRESSED_EKFAC_PLAN.md``) so callers don't have to carry a
variant tag in config.

Full-load-per-rank for MVP: each rank loads all shards for its variant and
concatenates them along dim 0, so ``.apply`` is a plain per-batch op with
no cross-rank communication. True sharded per-batch apply is a follow-up.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from safetensors.torch import load_file
from torch import Tensor

from .process_grads import get_trackstar_preconditioner, precondition_grad


@runtime_checkable
class Preconditioner(Protocol):
    """Applies an ``H^power`` approximation to a per-module gradient dict."""

    def apply(self, mod_grads: dict[str, Tensor]) -> dict[str, Tensor]: ...


class AutocorrelationPreconditioner:
    """Unfactored per-module preconditioner.

    Holds ``H^power`` per module as a plain ``dict[str, Tensor]`` and
    applies it via a right-matmul per example (existing trackstar
    semantics). ``h_inv`` is exposed so code paths that need the raw
    per-module operators (e.g. score-time split preconditioning) can reach
    through; new code should prefer ``.apply``.
    """

    def __init__(self, h_inv: dict[str, Tensor]):
        self.h_inv = h_inv

    def apply(self, mod_grads: dict[str, Tensor]) -> dict[str, Tensor]:
        return precondition_grad(mod_grads, self.h_inv)


def _load_sharded_dict(
    shard_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Load all ``shard_*.safetensors`` in a directory and concatenate per key.

    Shards are assumed to be split along dim 0 (matching
    :func:`bergson.hessians.eigenvectors._merge_and_shard_eigenvectors` and
    :class:`bergson.hessians.eigenvectors.LambdaCollector`). Output per key
    is the full matrix on ``device`` in ``dtype``.
    """
    shard_files = sorted(shard_dir.glob("shard_*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No shards found in {shard_dir}")

    per_key_shards: dict[str, list[Tensor]] = {}
    for f in shard_files:
        shard = load_file(str(f), device=str(device))
        for k, v in shard.items():
            per_key_shards.setdefault(k, []).append(v.to(dtype=dtype))

    return {k: torch.cat(v, dim=0) for k, v in per_key_shards.items()}


class _FactoredPreconditioner:
    """Shared base for KFAC/EKFAC: rotate-scale-rotate in the eigenbasis.

    Given activation eigenvectors ``Q_A`` [I, I], gradient eigenvectors
    ``Q_S`` [O, O], and a per-module eigenvalue matrix ``Λ`` [O, I], the
    operation applied per module is

        H^power G ≈ Q_S (Q_S^T G Q_A * λ^power) Q_A^T

    where ``λ = Λ + damp * mean(Λ)`` is the damped eigenvalue and ``G`` is
    reshaped from [_, O*I] → [_, O, I]. Subclasses differ only in how ``Λ``
    is sourced.
    """

    def __init__(
        self,
        q_a: dict[str, Tensor],
        q_s: dict[str, Tensor],
        lam: dict[str, Tensor],
        power: float = -1.0,
        lambda_damp_factor: float = 0.1,
    ):
        # Per-module shapes, cached once so .apply is pure tensor work.
        # Λ is [O, I] so its shape is the source of truth for (O, I).
        if not (set(q_a) == set(q_s) == set(lam)):
            missing_qa = set(lam) - set(q_a)
            missing_qs = set(lam) - set(q_s)
            raise ValueError(
                f"Factor dicts must agree on module names. "
                f"missing Q_A: {missing_qa}, missing Q_S: {missing_qs}"
            )
        self.q_a = q_a
        self.q_s = q_s
        self.lam = lam
        self.power = power
        self.lambda_damp_factor = lambda_damp_factor
        self._inv_lambda = {
            name: self._damped_lambda_power(lam[name]) for name in lam
        }
        self._shapes = {
            name: (lam[name].shape[0], lam[name].shape[1]) for name in lam
        }

    def _damped_lambda_power(self, lam_oi: Tensor) -> Tensor:
        mean_lam = lam_oi.mean()
        return (lam_oi + self.lambda_damp_factor * mean_lam).pow(self.power)

    def apply(self, mod_grads: dict[str, Tensor]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for name, g in mod_grads.items():
            O, I = self._shapes[name]
            q_a = self.q_a[name]
            q_s = self.q_s[name]
            inv_lam = self._inv_lambda[name]

            flat_leading = g.shape[0]
            # Cast to the factors' device/dtype (factors live on GPU in
            # float32 by construction). Match reference EkfacApplicator.
            g = g.to(device=q_a.device, dtype=torch.float32).view(flat_leading, O, I)

            # Rotate into eigenbasis: Q_S^T G Q_A
            g = torch.einsum("n o a, a i -> n o i", g, q_a)  # G @ Q_A
            g = torch.einsum("n a i, a o -> n o i", g, q_s)  # Q_S^T @ (G @ Q_A)

            # Scale by damped eigenvalues raised to ``power``
            g = g * inv_lam

            # Rotate back to parameter space: Q_S (·) Q_A^T
            g = torch.einsum("n a i, o a -> n o i", g, q_s)  # Q_S @ (·)
            g = torch.einsum("n o a, i a -> n o i", g, q_a)  # (·) @ Q_A^T

            out[name] = g.reshape(flat_leading, O * I)
        return out


class EkfacPreconditioner(_FactoredPreconditioner):
    """EKFAC: Λ is the per-element empirical eigenvalue correction.

    Loaded from ``<path>/eigenvalue_correction_sharded/shard_*.safetensors``,
    produced by :class:`bergson.hessians.eigenvectors.LambdaCollector`.
    """

    @classmethod
    def from_disk(
        cls,
        path: Path,
        device: torch.device,
        power: float = -1.0,
        lambda_damp_factor: float = 0.1,
    ) -> "EkfacPreconditioner":
        q_a = _load_sharded_dict(path / "eigen_activation_sharded", device)
        q_s = _load_sharded_dict(path / "eigen_gradient_sharded", device)
        lam = _load_sharded_dict(path / "eigenvalue_correction_sharded", device)
        return cls(q_a=q_a, q_s=q_s, lam=lam, power=power,
                   lambda_damp_factor=lambda_damp_factor)


class KfacPreconditioner(_FactoredPreconditioner):
    """KFAC: Λ is the outer product of per-factor eigenvalues ``Λ_G ⊗ Λ_A``.

    Expects ``eigenvalues_activation_sharded/`` and
    ``eigenvalues_gradient_sharded/`` alongside the eigenvector directories.
    These are written by the eigendecomposition step; if they are missing,
    the artifact was produced by an older version and must be regenerated.
    """

    @classmethod
    def from_disk(
        cls,
        path: Path,
        device: torch.device,
        power: float = -1.0,
        lambda_damp_factor: float = 0.1,
    ) -> "KfacPreconditioner":
        q_a = _load_sharded_dict(path / "eigen_activation_sharded", device)
        q_s = _load_sharded_dict(path / "eigen_gradient_sharded", device)

        lam_a_dir = path / "eigenvalues_activation_sharded"
        lam_s_dir = path / "eigenvalues_gradient_sharded"
        if not lam_a_dir.is_dir() or not lam_s_dir.is_dir():
            raise FileNotFoundError(
                f"KFAC preconditioner at {path} is missing per-factor "
                f"eigenvalue shards ({lam_a_dir.name}/ and/or "
                f"{lam_s_dir.name}/). Regenerate the artifact with the "
                "current version of compute_eigendecomposition."
            )
        lam_a = _load_sharded_dict(lam_a_dir, device)  # per module: [I]
        lam_s = _load_sharded_dict(lam_s_dir, device)  # per module: [O]
        lam = {name: torch.outer(lam_s[name], lam_a[name]) for name in lam_s}
        return cls(q_a=q_a, q_s=q_s, lam=lam, power=power,
                   lambda_damp_factor=lambda_damp_factor)


def _detect_variant(path: Path) -> str:
    """Return one of ``"autocorrelation"``, ``"kfac"``, ``"ekfac"``."""
    has_ea = (path / "eigen_activation_sharded").is_dir()
    has_eg = (path / "eigen_gradient_sharded").is_dir()
    has_ev = (path / "eigenvalue_correction_sharded").is_dir()

    if has_ea and has_eg:
        return "ekfac" if has_ev else "kfac"
    return "autocorrelation"


def load_preconditioner(
    preconditioner_path: str | Path | None,
    device: torch.device,
    power: float = -0.5,
    return_dtype: torch.dtype | None = None,
    lambda_damp_factor: float = 0.1,
) -> Preconditioner:
    """Load the right :class:`Preconditioner` for the artifact on disk.

    Parameters mirror :func:`bergson.process_grads.get_trackstar_preconditioner`
    for the autocorrelation path. Detection is by directory contents:

    * ``eigen_activation_sharded/`` + ``eigen_gradient_sharded/`` + ``eigenvalue_correction_sharded/`` → EKFAC
    * ``eigen_activation_sharded/`` + ``eigen_gradient_sharded/`` (no ``eigenvalue_correction_sharded/``) → KFAC
    * otherwise (or ``None``) → autocorrelation (``GradientProcessor`` dump)

    ``return_dtype`` and ``lambda_damp_factor`` are ignored by the variants
    that don't use them (factored preconditioners always work in float32
    internally; ``return_dtype`` only affects the autocorrelation path).
    """
    if preconditioner_path is None:
        return AutocorrelationPreconditioner(h_inv={})

    path = Path(preconditioner_path)
    variant = _detect_variant(path)

    if variant == "ekfac":
        return EkfacPreconditioner.from_disk(
            path, device=device, power=power,
            lambda_damp_factor=lambda_damp_factor,
        )

    if variant == "kfac":
        return KfacPreconditioner.from_disk(
            path, device=device, power=power,
            lambda_damp_factor=lambda_damp_factor,
        )

    h_inv = get_trackstar_preconditioner(
        str(path),
        device=device,
        power=power,
        return_dtype=return_dtype,
    )
    return AutocorrelationPreconditioner(h_inv=h_inv)
