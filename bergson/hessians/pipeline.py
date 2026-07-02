import shutil
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from ..build import build
from ..cli.commands import Build, Score
from ..config.config import (
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
)
from ..config.config_io import save_run_config
from ..distributed import launch_distributed_run
from ..score.score import score_dataset
from ..utils.worker_utils import validate_run_path
from .apply_hessian import EkfacConfig, apply_worker
from .hessian_approximations import approximate_hessians


def _step_complete(path: str, resume: bool) -> bool:
    """Check if a step's output already exists and should be skipped."""
    if not resume:
        return False
    if Path(path).exists():
        print(f"  Skipping (output exists at {path})")
        return True
    return False


def damping_tag(lam: float) -> str:
    """Filesystem-safe tag for a damping factor (e.g. 0.01 -> lam_1e-02)."""
    return "lam_" + f"{lam:.0e}".replace("+", "")


@contextmanager
def _timed(label: str, durations: dict[str, float]):
    """Time a pipeline step and print the wall-clock duration on exit."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        durations[label] = elapsed
        print(f"  [{label}] took {elapsed:.1f}s")


def hessian_pipeline(
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    hessian_pipeline_cfg: HessianPipelineConfig,
):
    """Run the full Hessian-preconditioned influence pipeline.

    1. Build mean query gradient.
    2. Fit Hessian factors (kfac, tkfac, shampoo) on the training dataset.
    3. Apply the inverse Hessian to the mean query gradient.
    4. Score each training example against the transformed query gradient.
    """
    run_path = index_cfg.run_path
    method = hessian_cfg.method
    query_path = f"{run_path}/query"
    hessian_path = f"{run_path}/hessian"
    resume = hessian_pipeline_cfg.resume

    # A damping sweep reuses the query gradients and Hessian factors, running
    # the cheap apply + score steps once per factor in suffixed directories.
    sweep = hessian_pipeline_cfg.lambda_damp_factors
    if sweep:
        damp_paths = [
            (
                lam,
                f"{run_path}/{method}_query_{damping_tag(lam)}",
                f"{run_path}/scores_{damping_tag(lam)}",
            )
            for lam in sweep
        ]
    else:
        damp_paths = [
            (
                hessian_pipeline_cfg.lambda_damp_factor,
                f"{run_path}/{method}_query",
                f"{run_path}/scores",
            )
        ]

    def _validate(cfg: IndexConfig):
        if resume and cfg.partial_run_path.exists():
            return
        validate_run_path(cfg)

    durations: dict[str, float] = {}

    # ── Step 1: Build query gradient(s) ───────────────────────────────────
    aggregation = hessian_pipeline_cfg.query_aggregation
    print(f"Step 1/4: Building query gradients (aggregation={aggregation})...")
    if not _step_complete(query_path, resume):
        with _timed("step1_build_query", durations):
            query_cfg = deepcopy(index_cfg)
            query_cfg.run_path = query_path
            query_cfg.data = hessian_pipeline_cfg.query
            query_cfg.projection_dim = 0
            _validate(query_cfg)

            query_preprocess_cfg = PreprocessConfig(aggregation=aggregation)
            save_run_config(
                Build(query_cfg, query_preprocess_cfg, None),
                query_cfg.partial_run_path,
            )
            build(query_cfg, query_preprocess_cfg, None)

    # ── Step 2: Fit Hessian factors on training data ──────────────────────
    print(f"Step 2/4: Fitting {method} factors on training data...")
    if not _step_complete(hessian_path, resume):
        with _timed("step2_fit_hessian", durations):
            hessian_index_cfg = deepcopy(index_cfg)
            # approximate_hessians writes to this exact path; step 3 reads it
            # back from `{hessian_path}/{method}`.
            hessian_index_cfg.run_path = f"{hessian_path}/{method}"
            _validate(hessian_index_cfg)

            approximate_hessians(hessian_index_cfg, hessian_cfg)

    # ── Steps 3+4: Apply inverse Hessian and score, once per damping ──────
    for i, (lam, transformed_query_path, scores_path) in enumerate(damp_paths):
        suffix = f" [{damping_tag(lam)} {i + 1}/{len(damp_paths)}]" if sweep else ""

        print(
            f"Step 3/4: Applying {method} inverse Hessian to query "
            f"gradients...{suffix}"
        )
        if not _step_complete(transformed_query_path, resume):
            hessian_method_path = f"{hessian_path}/{method}"
            ekfac_cfg = EkfacConfig(
                hessian_method_path=hessian_method_path,
                gradient_path=query_path,
                run_path=transformed_query_path,
                ev_correction=hessian_cfg.ev_correction,
                lambda_damp_factor=lam,
            )
            launch_distributed_run(
                "apply_hessian",
                apply_worker,
                [ekfac_cfg],
                index_cfg.distributed,
            )

        print(
            f"Step 4/4: Scoring training data against transformed " f"query...{suffix}"
        )
        if not _step_complete(scores_path, resume):
            score_index_cfg = deepcopy(index_cfg)
            score_index_cfg.run_path = scores_path
            score_index_cfg.projection_dim = 0
            damp_score_cfg = deepcopy(score_cfg)
            damp_score_cfg.query_path = transformed_query_path
            damp_score_cfg.higher_is_better = True
            _validate(score_index_cfg)

            save_run_config(
                Score(damp_score_cfg, score_index_cfg, preprocess_cfg),
                score_index_cfg.partial_run_path,
            )
            score_dataset(score_index_cfg, damp_score_cfg, preprocess_cfg)

        if (
            hessian_pipeline_cfg.cleanup_transformed_query
            and Path(scores_path).exists()
        ):
            print(f"  Cleaning up {transformed_query_path}")
            shutil.rmtree(transformed_query_path, ignore_errors=True)

    print(f"Done! Scores saved to: {', '.join(p for _, _, p in damp_paths)}")
    if durations:
        total = sum(durations.values())
        print(f"Step timings (s): {durations} | total {total:.1f}s")
