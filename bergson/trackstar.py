from copy import deepcopy
from pathlib import Path

from .build import build
from .config import (
    IndexConfig,
    PreprocessConfig,
    ScoreConfig,
    TrackStarConfig,
)
from .process_grads import mix_preconditioners
from .score.score import score_dataset
from .utils.worker_utils import validate_run_path


def limit_split_for_precond(cfg: IndexConfig) -> None:
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


def validate(resume, cfg: IndexConfig):
    """Validate run path, skipping when resume would preserve partial output."""
    if resume and cfg.partial_run_path.exists():
        return
    validate_run_path(cfg)


def is_step_complete(path: str, resume: bool) -> bool:
    """Check if a step's output already exists and should be skipped."""
    if not resume:
        return False
    if Path(path).exists():
        print(f"  Skipping (output exists at {path})")
        return True
    return False


def trackstar(
    index_cfg: IndexConfig,
    score_cfg: ScoreConfig,
    preprocess_cfg: PreprocessConfig,
    trackstar_cfg: TrackStarConfig,
):
    """Run the full trackstar pipeline: preconditioners -> mix -> build -> score."""
    run_path = index_cfg.run_path
    value_processor_path = f"{run_path}/value_processor"
    query_processsor_path = f"{run_path}/query_processor"
    mixed_preconditioner_path = f"{run_path}/mixed_preconditioner"
    query_path = f"{run_path}/query"
    scores_path = f"{run_path}/scores"

    # Steps 1-2 only compute preconditioners, so don't preprocess grads.
    precond_preprocess_cfg = PreprocessConfig()

    # Steps 1-2 upcast activations and gradients to fp32 for normalizer fitting,
    # so they may need a smaller token batch size than the main collection.
    precond_batch_size = (
        trackstar_cfg.stats_token_batch_size or index_cfg.token_batch_size
    )

    # Step 1: Compute normalizers and preconditioners on value dataset
    print("Step 1/5: Fit normalizers then compute preconditioners on index data...")
    if not is_step_complete(value_processor_path, trackstar_cfg.resume):
        value_processor_cfg = deepcopy(index_cfg)
        value_processor_cfg.run_path = value_processor_path
        value_processor_cfg.skip_index = True
        value_processor_cfg.skip_preconditioners = False
        value_processor_cfg.token_batch_size = precond_batch_size
        if trackstar_cfg.sample_preconditioners:
            limit_split_for_precond(value_processor_cfg)
        validate(trackstar_cfg.resume, value_processor_cfg)
        build(value_processor_cfg, precond_preprocess_cfg)

    # Step 2: Compute normalizers and preconditioners on query dataset
    print("Step 2/5: Fit normalizers then compute preconditioners on query data...")
    if not is_step_complete(query_processsor_path, trackstar_cfg.resume):
        query_processor_cfg = deepcopy(index_cfg)
        query_processor_cfg.run_path = query_processsor_path
        query_processor_cfg.data = trackstar_cfg.query
        query_processor_cfg.skip_index = True
        query_processor_cfg.skip_preconditioners = False
        query_processor_cfg.token_batch_size = precond_batch_size
        if trackstar_cfg.sample_preconditioners:
            limit_split_for_precond(query_processor_cfg)
        validate(trackstar_cfg.resume, query_processor_cfg)
        build(query_processor_cfg, precond_preprocess_cfg)

    # Step 3: Mix query and value preconditioners
    print("Step 3/5: Mixing preconditioners...")
    if not is_step_complete(mixed_preconditioner_path, trackstar_cfg.resume):
        mix_preconditioners(
            query_path=query_processsor_path,
            index_path=value_processor_path,
            output_path=mixed_preconditioner_path,
            target_downweight_components=trackstar_cfg.target_downweight_components,
        )

    preprocess_cfg.preconditioner_path = mixed_preconditioner_path

    # Step 4: Build query gradient index using query-specific normalizer.
    # The mixed preconditioner is set here but only applied during build if the
    # user is aggregating the query dataset (preprocess_cfg.aggregation != "none").
    # Otherwise, preconditioning will be deferred to score time in step 5.
    print("Step 4/5: Building query gradient index...")
    if not is_step_complete(query_path, trackstar_cfg.resume):
        query_index_cfg = deepcopy(index_cfg)
        query_index_cfg.run_path = query_path
        query_index_cfg.data = trackstar_cfg.query
        query_index_cfg.processor_path = query_processsor_path
        query_index_cfg.skip_preconditioners = True
        # Step 4 applies the mixed preconditioner during gradient collection,
        # which uses more memory than steps 1-2. Use the smaller batch size.
        query_index_cfg.token_batch_size = precond_batch_size
        validate(trackstar_cfg.resume, query_index_cfg)
        build(query_index_cfg, preprocess_cfg)

    # Step 5: Score value dataset against query using mixed preconditioner
    print("Step 5/5: Scoring value dataset...")
    if not is_step_complete(scores_path, trackstar_cfg.resume):
        score_index_cfg = deepcopy(index_cfg)
        score_index_cfg.run_path = scores_path
        score_index_cfg.processor_path = value_processor_path
        score_index_cfg.skip_preconditioners = True
        # Scoring applies preconditioners during gradient collection, so use
        # the smaller batch size to avoid OOM.
        score_index_cfg.token_batch_size = precond_batch_size
        score_cfg.query_path = query_path
        validate(trackstar_cfg.resume, score_index_cfg)
        score_dataset(score_index_cfg, score_cfg, preprocess_cfg)
