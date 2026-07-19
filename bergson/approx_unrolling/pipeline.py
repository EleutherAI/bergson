"""Top-level orchestrator for the approximate unrolling
training-data attribution pipeline.

We follow Training Data Attribution via Approximate Unrolled Differentiation
(Bae et al.). We compute Equation (15)/(22) by first gathering Hessian approximations
that we need. In Step 6 we multiply the query gradient with (16) to get pullbacked
per-segment query gradients and in Step 7 we multiply by F_segment(sigma) (20) which
allows us to compute per-segment average scores in Step 8 that sum to the final score.

Pipeline steps
--------------
1. Per-checkpoint covariance precompute (raw cov shards only, no eigvecs).
2. Per-segment cov aggregation + eigendecomposition (one combined launch).
3. Per-checkpoint lambda in segment eigenbasis.
4. Per-segment lambda aggregation.
5. Mean query gradient at the final checkpoint.
6. Phase 1: walk query backwards via F_backward to build
query_grad_0.. query_grad_{L-1} and save at
   ``<run>/segment_{l}/query_grad/`` (query_grad_{L-1} is just the top-level query).
7. Phase 2: Apply per-segment F_segment on query_grad_l to build query_grad_segment at
   ``<run>/segment_{l}/query_grad_segment/``.
8. Phase 3: per-segment scores at ``<run>/segment_{l}/scores/``, summed
   into ``<run>/scores/``.
"""

import gc
import shutil
from copy import deepcopy
from pathlib import Path

from ..build import build
from ..cli.commands import Build
from ..config import (
    ApproxUnrollingConfig,
    HessianConfig,
    IndexConfig,
    PreprocessConfig,
)
from ..config.config_io import save_run_config
from ..utils.logger import get_logger
from .approx_unrolling_math import (
    compute_lr_times_steps_per_segment,
    score_per_segment_and_aggregate,
    walk_query_phase1,
    walk_query_phase2,
)
from .precompute_checkpoints import (
    precompute_checkpoint_averaged_lambdas,
    precompute_checkpoint_hessians,
)
from .segment_aggregation import (
    aggregate_segment_covariances,
    aggregate_segment_lambdas,
)

# Total number of steps in the full approximate unrolling pipeline. Used only for the
# user-facing "Step k/N_TOTAL_STEPS:" prefix. Bump as steps land.
_N_TOTAL_STEPS = 8


def approx_unrolling_pipeline(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    approx_unrolling_cfg: ApproxUnrollingConfig,
):
    """Run the approximate unrolling (approximate unrolling) training-data
    attribution pipeline.

    Parameters
    ----------
    index_cfg : IndexConfig
        Base run config. ``index_cfg.run_path`` is the parent directory for
        **all** approximate unrolling artifacts (per-checkpoint outputs, per-segment
        outputs, transformed query, scores).
    hessian_cfg : HessianConfig
        EKFAC method and dtype. When ``ev_correction=True``, steps 3-4
        produce the segment-eigenbasis lambda; otherwise they're skipped.
    approx_unrolling_cfg : ApproxUnrollingConfig
        Checkpoints, segment count, query dataset, etc.
    resume : bool
        If ``True``, skip steps whose output directories already exist.
    """
    logger = get_logger("approx_unrolling_pipeline")

    n_ckpts = len(approx_unrolling_cfg.checkpoints)
    n_segments = approx_unrolling_cfg.segments
    if n_ckpts == 0:
        raise ValueError("approx_unrolling_cfg.checkpoints is empty.")
    if n_segments < 1:
        raise ValueError(
            f"approx_unrolling_cfg.segments must be ≥ 1, got {n_segments}."
        )
    if n_ckpts % n_segments != 0:
        raise ValueError(
            f"checkpoints ({n_ckpts}) must be divisible by segments "
            f"({n_segments}); got {n_ckpts}/{n_segments} = "
            f"{n_ckpts / n_segments:.3f} per segment."
        )

    assert hessian_cfg.ev_correction, "Approximate unrolling pipeline currently only "
    "supports EV correction on."

    lr_times_steps_per_segment = compute_lr_times_steps_per_segment(
        approx_unrolling_cfg
    )

    logger.info("=" * 70)
    logger.info(f"approximate unrolling pipeline -> {index_cfg.run_path}")
    logger.info(f"  base model        : {index_cfg.model}")
    logger.info(f"  checkpoints (C)   : {len(approx_unrolling_cfg.checkpoints)}")
    logger.info(f"  segments (L)      : {approx_unrolling_cfg.segments}")
    logger.info(f"  hessian method    : {hessian_cfg.method}")
    logger.info(f"  overwrite         : {index_cfg.overwrite}")
    logger.info("=" * 70)

    # ── Step 1: Per-checkpoint Hessian precompute
    logger.info(
        f"Step 1/{_N_TOTAL_STEPS}: "
        f"Precomputing {hessian_cfg.method} factors at each checkpoint..."
    )
    precompute_checkpoint_hessians(
        index_cfg,
        hessian_cfg,
        approx_unrolling_cfg,
        overwrite=index_cfg.overwrite,
    )
    # Encourage GC between expensive steps; matters most in single-GPU mode
    # where the worker ran in-process and may still hold model references.
    gc.collect()

    # ── Step 2: Per-segment covariance aggregation + eigendecomposition
    logger.info(
        f"Step 2/{_N_TOTAL_STEPS}: "
        f"Aggregating per-checkpoint covariances into segment averages..."
    )
    aggregate_segment_covariances(
        run_path=index_cfg.run_path,
        method=hessian_cfg.method,
        n_segments=n_segments,
        per_segment=n_ckpts // n_segments,
        distributed=index_cfg.distributed,
        resume=index_cfg.overwrite,
    )

    # ── Step 3: Per-checkpoint lambda in segment eigenbasis

    logger.info(
        f"Step 3/{_N_TOTAL_STEPS}: Per-checkpoint lambda using segment eigvecs..."
    )
    precompute_checkpoint_averaged_lambdas(
        index_cfg,
        hessian_cfg,
        approx_unrolling_cfg,
        resume=index_cfg.overwrite,
    )

    # ── Step 4: Per-segment lambda aggregation
    logger.info(
        f"Step 4/{_N_TOTAL_STEPS}: "
        f"Aggregating per-checkpoint lambdas into segment lambdas..."
    )
    aggregate_segment_lambdas(
        run_path=index_cfg.run_path,
        method=hessian_cfg.method,
        n_segments=n_segments,
        per_segment=n_ckpts // n_segments,
        distributed=index_cfg.distributed,
        resume=index_cfg.overwrite,
    )

    # ── Step 5: Mean query gradient at the final checkpoint
    logger.info(
        f"Step 5/{_N_TOTAL_STEPS}: "
        f"Building mean query gradient at the final checkpoint..."
    )
    query_path = Path(index_cfg.run_path) / "query"
    if index_cfg.overwrite and query_path.exists():
        logger.info(f"  skip — exists at {query_path}")
    else:
        if query_path.exists():
            shutil.rmtree(query_path)
        query_cfg = deepcopy(index_cfg)
        query_cfg.model = str(approx_unrolling_cfg.checkpoints[-1])
        query_cfg.data = approx_unrolling_cfg.query
        query_cfg.run_path = str(query_path)
        query_cfg.projection_dim = 0
        query_preprocess_cfg = PreprocessConfig(
            aggregation=approx_unrolling_cfg.query_aggregation
        )
        save_run_config(
            Build(query_cfg, query_preprocess_cfg),
            query_cfg.partial_run_path,
        )
        build(query_cfg, query_preprocess_cfg)

    # ── Step 6: Phase 1 -- walk query backwards to get segment queries
    logger.info(
        f"Step 6/{_N_TOTAL_STEPS}: "
        f"Phase 1 -- walking query backwards via F_backward to build "
        f"query_grad_0..query_grad_(L-1)..."
    )
    logger.info(f"  lr_times_steps per segment: {lr_times_steps_per_segment}")
    query_grad_paths = walk_query_phase1(
        run_path=index_cfg.run_path,
        method=hessian_cfg.method,
        lr_times_steps_per_segment=lr_times_steps_per_segment,
        distributed=index_cfg.distributed,
    )

    # ── Step 7: Phase 2 -- Get per-ckpt queries from segment queries
    logger.info(
        f"Step 7/{_N_TOTAL_STEPS}: "
        f"Phase 2 -- per-segment F_segment on query_grad_l to build "
        f"query_grad_segment_0..query_grad_segment_(L-1)..."
    )
    query_grad_segment_paths = walk_query_phase2(
        run_path=index_cfg.run_path,
        method=hessian_cfg.method,
        lr_times_steps_per_segment=lr_times_steps_per_segment,
        query_grad_paths=query_grad_paths,
        distributed=index_cfg.distributed,
    )

    # ── Step 8: Phase 3 -- per-segment scoring + sum
    logger.info(
        f"Step 8/{_N_TOTAL_STEPS}: Phase 3 -- per-segment scoring + aggregation..."
    )
    out_path = score_per_segment_and_aggregate(
        index_cfg=index_cfg,
        query_grad_segment_paths=query_grad_segment_paths,
        final_checkpoint=str(approx_unrolling_cfg.checkpoints[-1]),
    )
    logger.info(f"[approximate unrolling pipeline] DONE. Final scores at {out_path}")
