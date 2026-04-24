"""Compressed EKFAC index orchestrator.

Two-step pipeline:

1. ``approximate_hessians`` on the training set with ``ev_correction=True``
   writes EKFAC factors to ``<run_path>/hessian/<method>/`` (Q_A / Q_S /
   Λ / per-factor eigenvalues).
2. ``build`` on the same training set with
   ``preconditioner_path=<run_path>/hessian/<method>`` and
   ``projection_dim > 0`` auto-detects EKFAC via
   :func:`bergson.preconditioners._detect_variant` and bakes
   ``P · H^{-1/2} · g`` per example into a compressed, plain-dot-product
   queryable index at ``<run_path>/index/``.

Design (see §3 of ``COMPRESSED_EKFAC_PLAN.md``): the preconditioner is
applied at build time, not score time; ordering is precondition-then-
project (Grosse semantics); EKFAC artifacts on disk are the source of
truth for variant detection, so no ``preconditioner_type`` config field
is needed.
"""

from copy import deepcopy
from pathlib import Path

from ..build import build
from ..config import HessianConfig, IndexConfig, PreprocessConfig
from ..utils.worker_utils import validate_run_path
from .hessian_approximations import approximate_hessians


def compressed_ekfac_pipeline(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    preprocess_cfg: PreprocessConfig,
    resume: bool = False,
) -> Path:
    """Build a compressed EKFAC index and return its path.

    Parameters mirror the usual ``build`` CLI. ``hessian_cfg.ev_correction``
    is forced ``True`` (EKFAC requires the per-element Λ correction) and
    ``index_cfg.projection_dim`` must be > 0 (a "compressed" index with no
    projection defeats the purpose).

    **Pick ``projection_dim`` with care.** The builder applies a double-sided
    random projection ``L · G · R^T`` after EKFAC preconditioning (§15 of
    ``COMPRESSED_EKFAC_PLAN.md``). This is Kronecker-structured Johnson-
    Lindenstrauss: preserving per-module inner products requires roughly
    ``p ≳ sqrt(max_m(O_m · I_m))``. For pythia-14m that's ~128 (and p=64
    already achieves mean recall@10 ≈ 60 % vs a ground-truth reference in
    ``scripts/validate_compressed_ekfac.py``); p=16 runs but silently
    retrieves noise. See §18 of the plan for the empirical evidence.

    Also prefer ``preprocess_cfg.unit_normalize=True`` so the preconditioner
    applies ``H^{-1/2}`` on both sides, giving ``<q, H^{-1} t>`` rankings
    that match the standard influence-function formulation. The default
    ``unit_normalize=False`` uses ``H^{-1}`` on both sides (``<q, H^{-2} t>``)
    which is a different, over-preconditioned quantity.

    ``preprocess_cfg.preconditioner_path`` is ignored: the orchestrator
    points the build step at the freshly-written Hessian artifacts.
    """
    if index_cfg.projection_dim <= 0:
        raise ValueError(
            "compressed_ekfac requires projection_dim > 0 (got "
            f"{index_cfg.projection_dim}); this pipeline produces a "
            "compressed index, so a non-zero random projection is "
            "mandatory."
        )

    run_path = Path(index_cfg.run_path)
    hessian_base_path = run_path / "hessian"
    # approximate_hessians appends "/{method}" to the run path, so the
    # final factor dir is nested one level deeper.
    hessian_method_path = hessian_base_path / hessian_cfg.method
    index_path = run_path / "index"

    # ── Step 1: Fit EKFAC factors ──────────────────────────────────────
    if resume and hessian_method_path.exists():
        print(
            f"Step 1/2: Skipping Hessian fit — artifacts exist at "
            f"{hessian_method_path}"
        )
    else:
        print(
            f"Step 1/2: Fitting {hessian_cfg.method} factors with "
            f"eigenvalue correction at {hessian_method_path} ..."
        )
        hessian_index_cfg = deepcopy(index_cfg)
        hessian_index_cfg.run_path = str(hessian_base_path)
        # EKFAC needs Λ correction — force it on regardless of caller intent.
        hessian_cfg.ev_correction = True
        validate_run_path(hessian_index_cfg)
        approximate_hessians(hessian_index_cfg, hessian_cfg)

    # ── Step 2: Build compressed index with EKFAC baked in ─────────────
    if resume and index_path.exists():
        print(
            f"Step 2/2: Skipping index build — artifacts exist at "
            f"{index_path}"
        )
    else:
        print(
            f"Step 2/2: Building compressed index at {index_path} "
            f"(projection_dim={index_cfg.projection_dim}) ..."
        )
        build_index_cfg = deepcopy(index_cfg)
        build_index_cfg.run_path = str(index_path)
        # The builder fits its own autocorrelation preconditioner when
        # skip_preconditioners is False. We're supplying EKFAC factors
        # directly, so skip that step.
        build_index_cfg.skip_preconditioners = True

        build_preprocess_cfg = deepcopy(preprocess_cfg)
        build_preprocess_cfg.preconditioner_path = str(hessian_method_path)

        validate_run_path(build_index_cfg)
        build(build_index_cfg, build_preprocess_cfg)

    print(f"Done. Compressed EKFAC index at: {index_path}")
    return index_path
