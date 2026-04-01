from copy import deepcopy
from pathlib import Path

from .build import build
from .config import (
    IndexConfig,
    PreprocessConfig,
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


def _step_complete(path: str, resume: bool) -> bool:
    """Check if a step's output already exists and should be skipped."""
    if not resume:
        return False
    if Path(path).exists():
        print(f"  Skipping (output exists at {path})")
        return True
    return False


def trackstar(index_cfg: IndexConfig, trackstar_cfg: TrackstarConfig):
    """Run the full trackstar pipeline: preconditioners -> mix -> build -> score."""
    run_path = index_cfg.run_path
    value_precond_path = f"{run_path}/value_preconditioner"
    query_precond_path = f"{run_path}/query_preconditioner"
    mixed_precond_path = f"{run_path}/mixed_preconditioner"
    query_path = f"{run_path}/query"
    scores_path = f"{run_path}/scores"
    resume = trackstar_cfg.resume

    # Steps 1-2 only compute preconditioners, so don't preprocess grads.
    precond_preprocess_cfg = PreprocessConfig()

    def _validate(cfg: IndexConfig):
        """Validate run path, skipping when resume would preserve partial output."""
        if resume and cfg.partial_run_path.exists():
            return
        validate_run_path(cfg)

    # Step 1: Compute normalizers and preconditioners on value dataset
    print("Step 1/5: Computing normalizers and preconditioners on value dataset...")
    if not _step_complete(value_precond_path, resume):
        value_precond_cfg = deepcopy(index_cfg)
        value_precond_cfg.run_path = value_precond_path
        value_precond_cfg.skip_index = True
        value_precond_cfg.skip_preconditioners = False
        if trackstar_cfg.num_stats_sample_preconditioner:
            _limit_split_for_precond(value_precond_cfg)
        _validate(value_precond_cfg)
        build(value_precond_cfg, precond_preprocess_cfg)

    # Step 2: Compute normalizers and preconditioners on query dataset
    print("Step 2/5: Computing normalizers and preconditioners on query dataset...")
    if not _step_complete(query_precond_path, resume):
        query_precond_cfg = deepcopy(index_cfg)
        query_precond_cfg.run_path = query_precond_path
        query_precond_cfg.data = trackstar_cfg.query
        query_precond_cfg.skip_index = True
        query_precond_cfg.skip_preconditioners = False
        if trackstar_cfg.num_stats_sample_preconditioner:
            _limit_split_for_precond(query_precond_cfg)
        _validate(query_precond_cfg)
        build(query_precond_cfg, precond_preprocess_cfg)

    # Step 3: Mix query and value preconditioners
    print("Step 3/5: Mixing preconditioners...")
    if not _step_complete(mixed_precond_path, resume):
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
    trackstar_cfg.preprocess_cfg.preconditioner_path = mixed_precond_path
    if not _step_complete(query_path, resume):
        query_cfg = deepcopy(index_cfg)
        query_cfg.run_path = query_path
        query_cfg.data = trackstar_cfg.query
        query_cfg.processor_path = query_precond_path
        query_cfg.skip_preconditioners = True
        _validate(query_cfg)
        build(query_cfg, trackstar_cfg.preprocess_cfg)

    # Step 5: Score value dataset against query using mixed preconditioner
    print("Step 5/5: Scoring value dataset...")
    if not _step_complete(scores_path, resume):
        score_index_cfg = deepcopy(index_cfg)
        score_index_cfg.run_path = scores_path
        score_index_cfg.processor_path = value_precond_path
        score_index_cfg.skip_preconditioners = True
        trackstar_cfg.score_cfg.query_path = query_path
        _validate(score_index_cfg)
        score_dataset(
            score_index_cfg, trackstar_cfg.score_cfg, trackstar_cfg.preprocess_cfg
        )
