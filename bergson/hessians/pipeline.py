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
from ..data import average_gradient_indices
from ..distributed import launch_distributed_run
from ..score.score import score_dataset
from ..utils.step_state import prepare_step
from ..utils.worker_utils import validate_run_path
from .apply_hessian import EkfacConfig, apply_worker
from .hessian_approximations import approximate_hessians


def _step_complete(path: str, resume: bool) -> bool:
    """Whether the step writing to `path` is already done and can be skipped.

    Clears any interrupted ``.part`` output so the step restarts cleanly.
    """
    if prepare_step(path, resume=resume):
        return False
    print(f"  Skipping (already complete at {path})")
    return True


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
    if preprocess_cfg.unit_normalize:
        raise ValueError(
            "preprocess_cfg.unit_normalize (cosine similarity) is not "
            "supported with Kronecker-factored Hessians; hessian_pipeline "
            "only ever fits and applies a factored (kfac/tkfac/shampoo) "
            "Hessian."
        )

    run_path = index_cfg.run_path
    method = hessian_cfg.method
    query_path = f"{run_path}/query"
    hessian_path = f"{run_path}/hessian"
    transformed_query_path = f"{run_path}/{method}_query"
    scores_path = f"{run_path}/scores"
    resume = hessian_pipeline_cfg.resume

    def _validate(cfg: IndexConfig):
        if resume and cfg.partial_run_path.exists():
            return
        validate_run_path(cfg)

    durations: dict[str, float] = {}

    # ── Step 1: Build query gradient(s) ───────────────────────────────────
    aggregation = hessian_pipeline_cfg.query_aggregation
    print(f"Step 1/4: Building query gradient(s) (aggregation={aggregation})...")
    if not _step_complete(query_path, resume):
        with _timed("step1_build_query", durations):
            query_cfg = deepcopy(index_cfg)
            query_cfg.run_path = query_path
            query_cfg.data = hessian_pipeline_cfg.query
            query_cfg.projection_dim = 0

            # Query aggregation is not compatible with query-side token
            # attribution: aggregating collapses per-token query gradients
            # into a single target gradient, so a per-token query index is
            # invalid. Build a per-example query instead. Mirrors the same
            # guard in cli/trackstar.py's query build step.
            if aggregation != "none" and query_cfg.attribute_tokens:
                print(
                    "Query aggregation is not compatible with query-side "
                    "token attribution; building a per-example query instead."
                )
                query_cfg.attribute_tokens = False

            _validate(query_cfg)

            query_preprocess_cfg = PreprocessConfig(aggregation=aggregation)
            save_run_config(
                Build(query_cfg, query_preprocess_cfg),
                query_cfg.partial_run_path,
            )

            ckpt_models = list(hessian_pipeline_cfg.query_model_paths)
            if len(ckpt_models) > 1:
                parts = []
                for i, model_path in enumerate(ckpt_models):
                    part_cfg = deepcopy(query_cfg)
                    part_cfg.model = model_path
                    part_cfg.run_path = f"{query_path}__ckpt{i}"
                    _validate(part_cfg)
                    build(part_cfg, query_preprocess_cfg)
                    parts.append(Path(part_cfg.run_path))
                print(f"  averaging query gradients over {len(parts)} checkpoints")
                average_gradient_indices(parts, Path(query_path))
            else:
                if ckpt_models:
                    query_cfg.model = ckpt_models[0]
                build(query_cfg, query_preprocess_cfg)

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

    # ── Step 3: Apply inverse Hessian to the mean query gradient ──────────
    print(f"Step 3/4: Applying {method} inverse Hessian to mean query gradient...")
    if not _step_complete(transformed_query_path, resume):
        hessian_method_path = f"{hessian_path}/{method}"
        ekfac_cfg = EkfacConfig(
            hessian_method_path=hessian_method_path,
            gradient_path=query_path,
            run_path=transformed_query_path,
            ev_correction=hessian_cfg.ev_correction,
            projection_dim=index_cfg.projection_dim,
            projection_type=index_cfg.projection_type,
        )
        launch_distributed_run(
            "apply_hessian",
            apply_worker,
            [ekfac_cfg, hessian_pipeline_cfg.inversion_cfg],
            index_cfg.distributed,
        )

    # ── Step 4: Score training examples ───────────────────────────────────
    print("Step 4/4: Scoring training data against transformed query...")
    if not _step_complete(scores_path, resume):
        score_index_cfg = deepcopy(index_cfg)
        score_index_cfg.run_path = scores_path
        score_cfg.query_path = transformed_query_path
        score_cfg.higher_is_better = True
        _validate(score_index_cfg)

        save_run_config(
            Score(score_cfg, score_index_cfg, preprocess_cfg),
            score_index_cfg.partial_run_path,
        )
        score_dataset(score_index_cfg, score_cfg, preprocess_cfg)

    print(f"Done! Scores saved to: {scores_path}")
    if durations:
        total = sum(durations.values())
        print(f"Step timings (s): {durations} | total {total:.1f}s")
