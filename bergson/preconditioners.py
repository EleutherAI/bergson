"""Preconditioner interface and auto-detect loader.

Three first-class variants of Hessian approximation are supported, each
differing in how `H^power` is represented:

* ``autocorrelation`` — unfactored per-module ``Σ vec(g) vec(g)ᵀ`` (what the
  existing ``GradientProcessor`` pipeline stores). The preconditioner is a
  single ``dict[str, Tensor]`` of ``H^power`` matrices; ``apply`` is a per-
  module right-matmul.
* ``kfac`` — factored activation/gradient covariances (no eigenvalue
  correction). Implemented in commit 2.
* ``ekfac`` — factored + per-element eigenvalue correction. Implemented in
  commit 2.

``load_preconditioner`` auto-detects the variant from directory contents
(see §3.3 of ``COMPRESSED_EKFAC_PLAN.md``) so callers don't have to carry a
variant tag in config.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
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
) -> Preconditioner:
    """Load the right :class:`Preconditioner` for the artifact on disk.

    Parameters mirror :func:`bergson.process_grads.get_trackstar_preconditioner`
    for the autocorrelation path. Detection is by directory contents:

    * ``eigen_activation_sharded/`` + ``eigen_gradient_sharded/`` + ``eigenvalue_correction_sharded/`` → EKFAC
    * ``eigen_activation_sharded/`` + ``eigen_gradient_sharded/`` (no ``eigenvalue_correction_sharded/``) → KFAC
    * otherwise (or ``None``) → autocorrelation (``GradientProcessor`` dump)
    """
    if preconditioner_path is None:
        return AutocorrelationPreconditioner(h_inv={})

    path = Path(preconditioner_path)
    variant = _detect_variant(path)

    if variant == "ekfac" or variant == "kfac":
        raise NotImplementedError(
            f"{variant.upper()} preconditioner loading is not yet wired "
            "into the build/score paths. Compressed-EKFAC indices are "
            "scored via plain dot-product (see the compressed-ekfac "
            "notebook); for EKFAC/KFAC at score time, use that path "
            "rather than create_scorer."
        )

    h_inv = get_trackstar_preconditioner(
        str(path),
        device=device,
        power=power,
        return_dtype=return_dtype,
    )
    return AutocorrelationPreconditioner(h_inv=h_inv)
