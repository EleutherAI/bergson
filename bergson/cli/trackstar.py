from copy import deepcopy
from pathlib import Path

from ..build import build_contrast
from ..config.config import (
    HessianConfig,
    IndexConfig,
    TrackstarConfig,
)
from ..config.config_io import save_run_config
from ..distributed import parent_barrier
from ..hessians.hessian_approximations import approximate_hessians
from ..process_grads import mix_autocorrelation_matrices
from ..score.score import score_dataset
from ..utils.worker_utils import validate_run_path
from .commands import Build, Hessian, Mix, Score


def _limit_split_for_hess(cfg: IndexConfig, stats_sample_size: int | None) -> None:
    """Limit the data split to stats_sample_size for hessian-only steps."""
    # TODO this code is hacky and bad

    if stats_sample_size is not None:
        split = cfg.data.split
        # Append HF slice notation if not already present
        if "[" not in split:
            cfg.data.split = f"{split}[:{stats_sample_size}]"
        else:
            base_split = split.split("[")[0]
            cfg.data.split = f"{base_split}[:{stats_sample_size}]"


def _step_complete(path: str, resume: bool) -> bool:
    """Check if a step's output already exists and should be skipped."""
    if not resume:
        return False
    if Path(path).exists():
        print(f"  Skipping (output exists at {path})")
        return True
    return False


def trackstar(index_cfg: IndexConfig, trackstar_cfg: TrackstarConfig):
    """Run the full trackstar pipeline: hessians -> mix -> build -> score."""
    if index_cfg.projection_dim == 0:
        raise ValueError(
            "Trackstar scores via compressed random-projected gradients and "
            "requires a nonzero index_cfg.projection_dim; got 0. Leave "
            "projection_dim unset to use TrackstarIndexConfig's default of "
            "16, or set it explicitly."
        )

    run_path = index_cfg.run_path
    value_hess_path = f"{run_path}/value_hessian"
    query_hess_path = f"{run_path}/query_hessian"
    mixed_hess_path = f"{run_path}/mixed_hessian"
    query_path = f"{run_path}/query"
    scores_path = f"{run_path}/scores"
    resume = trackstar_cfg.resume

    hess_cfg = HessianConfig(method="autocorrelation")

    def _validate(cfg: IndexConfig):
        """Validate run path, skipping when resume would preserve partial output."""
        if resume and cfg.partial_run_path.exists():
            return
        validate_run_path(cfg)

    # Step 1: Compute hessians on value dataset
    print("Step 1/5: Computing hessians on value dataset...")
    if not _step_complete(value_hess_path, resume):
        value_hess_cfg = deepcopy(index_cfg)
        value_hess_cfg.run_path = value_hess_path
        _limit_split_for_hess(value_hess_cfg, trackstar_cfg.stats_sample_size)
        _validate(value_hess_cfg)
        save_run_config(
            Hessian(hessian_cfg=hess_cfg, index_cfg=value_hess_cfg),
            value_hess_cfg.partial_run_path,
        )
        approximate_hessians(value_hess_cfg, hess_cfg)

    if not trackstar_cfg.mix_hessians:
        print("Steps 2-3/5: Skipped (mix_hessians=false); using the value hessian.")
        precondition_hess_path = value_hess_path
    else:
        # Step 2: Compute hessians on query dataset
        print("Step 2/5: Computing hessians on query dataset...")
        if not _step_complete(query_hess_path, resume):
            query_hess_cfg = deepcopy(index_cfg)
            query_hess_cfg.run_path = query_hess_path
            query_hess_cfg.data = deepcopy(trackstar_cfg.query)
            _limit_split_for_hess(query_hess_cfg, trackstar_cfg.stats_sample_size)
            _validate(query_hess_cfg)
            save_run_config(
                Hessian(hessian_cfg=hess_cfg, index_cfg=query_hess_cfg),
                query_hess_cfg.partial_run_path,
            )
            approximate_hessians(query_hess_cfg, hess_cfg)

        # Step 3: Mix query and value hessians
        print("Step 3/5: Mixing hessians...")
        if not _step_complete(mixed_hess_path, resume):
            if index_cfg.distributed._node_rank == 0:
                save_run_config(
                    Mix(
                        query_path=query_hess_path,
                        index_path=value_hess_path,
                        output_path=mixed_hess_path,
                        target_downweight_components=(
                            trackstar_cfg.target_downweight_components
                        ),
                    ),
                    mixed_hess_path,
                )
                mix_autocorrelation_matrices(
                    query_path=query_hess_path,
                    index_path=value_hess_path,
                    output_path=mixed_hess_path,
                    target_downweight_components=(
                        trackstar_cfg.target_downweight_components
                    ),
                )
        precondition_hess_path = mixed_hess_path
    parent_barrier(index_cfg.distributed)

    # The preconditioning hessian is set here but only applied during step 4. if the
    # user is aggregating the query dataset (preprocess_cfg.aggregation != "none").
    # Otherwise, preconditioning will be deferred to score time in step 5.
    trackstar_cfg.preprocess_cfg.hessian_path = precondition_hess_path

    # Step 4: Build query gradient index using query-specific normalizer.
    print("Step 4/5: Building query gradient index...")
    if not _step_complete(query_path, resume):
        query_cfg = deepcopy(index_cfg)
        query_cfg.run_path = query_path
        query_cfg.data = deepcopy(trackstar_cfg.query)

        # query-side aggregation is currently not compatible with token attribution
        # only. When aggregating the query (aggregation != "none"), per-token
        # query gradients are collapsed into a single target gradient, so a
        # per-token query index is invalid. Build a per-example query index.
        if (
            trackstar_cfg.preprocess_cfg.aggregation != "none"
            and query_cfg.attribute_tokens
        ):
            print(
                "Query aggregation is currently not compatible with"
                "query-side token attribution. Any query unit normalization"
                "will be applied to sequence gradients."
            )
            query_cfg.attribute_tokens = False

        _validate(query_cfg)
        save_run_config(
            Build(query_cfg, trackstar_cfg.preprocess_cfg),
            query_cfg.partial_run_path,
        )
        build_contrast(
            query_cfg, trackstar_cfg.query_contrast, trackstar_cfg.preprocess_cfg
        )

    # Step 5: Score value dataset against query using mixed hessian
    print("Step 5/5: Scoring value dataset...")
    if not _step_complete(scores_path, resume):
        score_index_cfg = deepcopy(index_cfg)
        score_index_cfg.run_path = scores_path
        trackstar_cfg.score_cfg.query_path = query_path
        trackstar_cfg.score_cfg.higher_is_better = True
        _validate(score_index_cfg)
        save_run_config(
            Score(
                trackstar_cfg.score_cfg, score_index_cfg, trackstar_cfg.preprocess_cfg
            ),
            score_index_cfg.partial_run_path,
        )
        score_dataset(
            score_index_cfg, trackstar_cfg.score_cfg, trackstar_cfg.preprocess_cfg
        )
