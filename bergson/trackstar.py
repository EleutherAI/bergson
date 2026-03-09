from copy import deepcopy

from .build import build
from .config import (
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
    TrackstarConfig,
)
from .process_grads import mix_preconditioners
from .score.score import score_dataset
from .utils.worker_utils import validate_run_path


def _limit_split_for_precond(cfg: IndexConfig) -> None:
    """Limit the data split to stats_sample_size for preconditioner-only steps."""
    # TODO this code is hacky and bad

    if cfg.stats_sample_size is not None:
        split = cfg.data.split
        # Append HF slice notation if not already present
        if "[" not in split:
            cfg.data.split = f"{split}[:{cfg.stats_sample_size}]"
        else:
            base_split = split.split("[")[0]
            cfg.data.split = f"{base_split}[:{cfg.stats_sample_size}]"


def trackstar(
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    trackstar_cfg: TrackstarConfig,
):
    """Run the full trackstar pipeline: preconditioners -> mix -> build -> score."""
    run_path = index_cfg.run_path
    value_precond_path = f"{run_path}/value_preconditioner"
    query_precond_path = f"{run_path}/query_preconditioner"
    mixed_precond_path = f"{run_path}/mixed_preconditioner"
    query_path = f"{run_path}/query"
    scores_path = f"{run_path}/scores"

    # Steps 1-2 only compute preconditioners, so don't preprocess grads.
    precond_preprocess_cfg = PreprocessConfig()

    # Step 1: Compute normalizers and preconditioners on value dataset
    print("Step 1/5: Computing normalizers and preconditioners on value dataset...")
    value_precond_cfg = deepcopy(index_cfg)
    value_precond_cfg.run_path = value_precond_path
    value_precond_cfg.skip_index = True
    value_precond_cfg.skip_preconditioners = False
    if trackstar_cfg.num_stats_sample_preconditioner:
        _limit_split_for_precond(value_precond_cfg)
    validate_run_path(value_precond_cfg)
    build(value_precond_cfg, precond_preprocess_cfg)

    # Step 2: Compute normalizers and preconditioners on query dataset
    print("Step 2/5: Computing normalizers and preconditioners on query dataset...")
    query_precond_cfg = deepcopy(index_cfg)
    query_precond_cfg.run_path = query_precond_path
    query_precond_cfg.data = trackstar_cfg.query
    query_precond_cfg.skip_index = True
    query_precond_cfg.skip_preconditioners = False
    if trackstar_cfg.num_stats_sample_preconditioner:
        _limit_split_for_precond(query_precond_cfg)
    validate_run_path(query_precond_cfg)
    build(query_precond_cfg, precond_preprocess_cfg)

    # Step 3: Mix query and value preconditioners
    print("Step 3/5: Mixing preconditioners...")
    mix_preconditioners(
        query_path=query_precond_path,
        index_path=value_precond_path,
        output_path=mixed_precond_path,
        target_downweight_components=trackstar_cfg.target_downweight_components,
    )

    # Step 4: Build query gradient index using query-specific normalizer.
    # The mixed preconditioner is set here but only applied during build if the
    # user is aggregating the query dataset (preprocess_cfg.aggregation != "none").
    # Otherwise, preconditioning will be deferred to score time in step 5.
    print("Step 4/5: Building query gradient index...")
    preprocess_cfg.preconditioner_path = mixed_precond_path
    query_cfg = deepcopy(index_cfg)
    query_cfg.run_path = query_path
    query_cfg.data = trackstar_cfg.query
    query_cfg.processor_path = query_precond_path
    query_cfg.skip_preconditioners = True
    validate_run_path(query_cfg)
    build(query_cfg, preprocess_cfg)

    # Step 5: Score value dataset against query using mixed preconditioner
    print("Step 5/5: Scoring value dataset...")
    score_index_cfg = deepcopy(index_cfg)
    score_index_cfg.run_path = scores_path
    score_index_cfg.processor_path = value_precond_path
    score_index_cfg.skip_preconditioners = True
    score_cfg.query_path = query_path
    validate_run_path(score_index_cfg)
    score_dataset(score_index_cfg, score_cfg, preprocess_cfg)
