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

Also exports :func:`embed_query` for the common one-off case of
embedding a handful of held-out queries against pre-computed EKFAC
factors without manually orchestrating the full pipeline.
"""

import json
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ..build import build
from ..config import DataConfig, HessianConfig, IndexConfig, PreprocessConfig
from ..data import load_gradients
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

    # Fail fast: ``build_worker`` raises the same error in step 2 once a
    # factored preconditioner is detected, but by then step 1 has already
    # burned the (slow) Hessian fit. Catch it up front so users get the
    # error before any work runs.
    if index_cfg.include_bias:
        raise NotImplementedError(
            "include_bias=True with compressed_ekfac is not yet supported. "
            "The factored Q_A matrix does not cover the extra bias column; "
            "see §15.5 of COMPRESSED_EKFAC_PLAN.md."
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
        print(f"Step 2/2: Skipping index build — artifacts exist at " f"{index_path}")
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


def _flatten_structured(mmap) -> np.ndarray:
    """Concatenate every module's gradients into ``[N, total_dim]`` as float32.

    Bergson writes bf16 as ``ml_dtypes.bfloat16`` in the structured mmap;
    numpy can't cast that natively, so we round-trip through torch.
    Mirror of the helper in ``scripts/validate_compressed_ekfac.py`` —
    consider promoting to ``bergson.utils`` if a third caller appears.
    """
    parts: list[np.ndarray] = []
    for name in mmap.dtype.names:
        arr = np.ascontiguousarray(mmap[name]).reshape(len(mmap), -1)
        if arr.dtype == np.float32:
            parts.append(arr.astype(np.float32, copy=False))
        else:
            t = torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
            parts.append(t.float().numpy())
    return np.concatenate(parts, axis=-1)


def embed_query(
    queries: list[str],
    *,
    model: str,
    ekfac_path: str | Path,
    projection_dim: int,
    unit_normalize: bool = True,
    precision: str = "bf16",
    token_batch_size: int = 1024,
    truncation: bool = True,
    debug: bool = False,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Embed a batch of queries into the compressed-EKFAC space.

    Returns a ``[len(queries), total_compressed_dim]`` float32 numpy array
    of EKFAC-preconditioned, randomly-projected gradient vectors that can
    be dot-producted directly against a compressed-EKFAC index built with
    the same ``ekfac_path`` and ``projection_dim``. Same projection
    matrices are used at index- and query-time because
    :func:`bergson.collector.collector.create_projection_matrix` seeds
    deterministically from each module's name.

    Internally writes the queries to a temp jsonl, runs a tiny
    ``build`` against the supplied EKFAC factors, and reads the result
    back. For one-off queries this is the simplest API; for repeated
    use against a fixed query set, prefer the orchestrator directly so
    the on-disk artifact persists.

    Parameters
    ----------
    queries : list[str]
        Query texts. Must be non-empty.
    model : str
        HuggingFace model id (must match the model used to build the
        index — mismatched architectures will produce wrong embeddings).
    ekfac_path : str | Path
        Path to a directory containing EKFAC factors (e.g. the
        ``<run_path>/hessian/kfac`` produced by
        :func:`compressed_ekfac_pipeline`).
    projection_dim : int
        Must match the projection_dim used to build the index against
        which these embeddings will be scored.
    unit_normalize : bool
        Match the index's setting (default True for split
        preconditioning, the standard influence-function quantity).
    cache_dir : str | Path | None
        Where to write the temporary jsonl + intermediate index. When
        ``None`` (default) a temp directory is created and removed on
        return.
    """
    if not queries:
        raise ValueError("queries must be non-empty")

    ekfac_path = str(Path(ekfac_path).resolve())
    cleanup_ctx: tempfile.TemporaryDirectory | None = None
    if cache_dir is None:
        cleanup_ctx = tempfile.TemporaryDirectory(prefix="bergson_embed_query_")
        cache_root = Path(cleanup_ctx.name)
    else:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

    try:
        jsonl_path = cache_root / "queries.jsonl"
        with jsonl_path.open("w") as f:
            for text in queries:
                f.write(json.dumps({"text": text}) + "\n")

        index_path = cache_root / "embedding"
        cfg = IndexConfig(
            run_path=str(index_path),
            model=model,
            precision=precision,
            projection_dim=projection_dim,
            token_batch_size=token_batch_size,
            skip_preconditioners=True,
            data=DataConfig(
                dataset=str(jsonl_path),
                split="train",
                truncation=truncation,
            ),
            debug=debug,
        )
        cfg.distributed.nproc_per_node = 1

        validate_run_path(cfg)
        build(
            cfg,
            PreprocessConfig(
                preconditioner_path=ekfac_path,
                unit_normalize=unit_normalize,
            ),
        )

        return _flatten_structured(load_gradients(index_path))
    finally:
        if cleanup_ctx is not None:
            cleanup_ctx.cleanup()
