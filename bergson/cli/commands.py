"""CLI command definitions.

Each command is a thin dataclass that validates, persists its ``config.yaml``,
and dispatches. They live here rather than in ``__main__`` so pipelines can
import lower-level commands for config serialization without importing the
corresponding CLI entrypoint.
"""

from dataclasses import dataclass

from simple_parsing import Serializable, field

from ..build import build
from ..config.config import (
    ApproxUnrollingConfig,
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    MixConfig,
    PreprocessConfig,
    QueryConfig,
    ScoreConfig,
    TrackstarConfig,
    TrainingConfig,
    ValidationConfig,
)
from ..config.config_io import save_run_config
from ..diagnose import DiagnoseConfig, diagnose
from ..hessians.hessian_approximations import approximate_hessians
from ..magic import MagicConfig, run_magic
from ..process_grads import mix_autocorrelation_matrices
from ..query.query_index import query
from ..score.score import score_dataset
from ..utils.worker_utils import prepare_shard, validate_run_path


@dataclass
class ApproxUnrolling(Serializable):
    """Run the SOURCE (approximate unrolling) training-data attribution pipeline.

    Currently only step 1 (per-checkpoint Hessian precompute) is wired up;
    later steps land incrementally. See
    :mod:`bergson.approx_unrolling.pipeline` for the step list.
    """

    index_cfg: IndexConfig

    hessian_cfg: HessianConfig

    approx_unrolling_cfg: ApproxUnrollingConfig

    def execute(self):
        from ..approx_unrolling.pipeline import approx_unrolling_pipeline

        if self.index_cfg.sharded:
            raise ValueError(
                "approx_unrolling does not support sharded runs yet; "
                "shard the build and score steps individually."
            )

        save_run_config(self, self.index_cfg.run_path)
        approx_unrolling_pipeline(
            self.index_cfg,
            self.hessian_cfg,
            self.approx_unrolling_cfg,
        )


@dataclass
class Build(Serializable):
    """
    Build a gradient index. Simultaneously approximate an autocorrelation Hessian
    by passing `--method autocorrelation`."""

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    # Pass `--method autocorrelation` to simultaneously approximate a Hessian.
    # `build` only supports autocorrelation Hessians; other methods go through
    # the `hessian` command.
    hessian_cfg: HessianConfig | None = None

    def execute(self):
        """Build the gradient index."""
        if self.index_cfg.skip_index and self.hessian_cfg is None:
            raise ValueError(
                "if skip_index is True HessianConfig.method must be provided"
            )

        if (
            self.hessian_cfg is not None
            and self.hessian_cfg.method != "autocorrelation"
        ):
            raise ValueError(
                f"build only supports autocorrelation Hessians, got "
                f"'{self.hessian_cfg.method}'. Use the `hessian` command for "
                f"{self.hessian_cfg.method}."
            )

        if self.index_cfg.sharded:
            if self.hessian_cfg is not None:
                raise ValueError(
                    "Sharded builds do not support simultaneous Hessian "
                    "estimation; Hessian factors cannot be merged across "
                    "independent shards yet. Run `bergson hessian` separately."
                )
            if prepare_shard(self, self.index_cfg):
                return
        else:
            validate_run_path(self.index_cfg)

        save_run_config(self, self.index_cfg.partial_run_path)
        build(self.index_cfg, self.preprocess_cfg, self.hessian_cfg)


@dataclass
class Ekfac(Serializable):
    """Run the full EKFAC influence pipeline end-to-end."""

    index_cfg: IndexConfig

    hessian_cfg: HessianConfig

    score_cfg: ScoreConfig

    preprocess_cfg: PreprocessConfig

    hessian_pipeline_cfg: HessianPipelineConfig

    def execute(self):
        from ..hessians.pipeline import hessian_pipeline

        if self.index_cfg.sharded:
            raise ValueError(
                "ekfac does not support sharded runs yet; "
                "shard the build and score steps individually."
            )

        save_run_config(self, self.index_cfg.run_path)
        hessian_pipeline(
            self.index_cfg,
            self.hessian_cfg,
            self.score_cfg,
            self.preprocess_cfg,
            self.hessian_pipeline_cfg,
        )


@dataclass
class Hessian(Serializable):
    """Approximate Hessian matrices using KFAC or EKFAC."""

    hessian_cfg: HessianConfig
    index_cfg: IndexConfig

    def execute(self):
        """Compute Hessian approximation."""
        if self.index_cfg.sharded:
            raise ValueError(
                "hessian does not support sharded runs; Hessian factors "
                "cannot be merged across independent shards yet."
            )

        validate_run_path(self.index_cfg)

        if self.hessian_cfg.method == "autocorrelation":
            self.index_cfg.skip_index = True
            save_run_config(self, self.index_cfg.partial_run_path)
            build(self.index_cfg, PreprocessConfig(), self.hessian_cfg)
        else:
            save_run_config(self, self.index_cfg.partial_run_path)
            approximate_hessians(self.index_cfg, self.hessian_cfg)


@dataclass
class Magic(MagicConfig):
    """Run MAGIC attribution."""

    def execute(self):
        """Run MAGIC attribution."""
        run_magic(self)


@dataclass
class Mix(MixConfig):
    """Mix two autocorrelation hessians into a single GradientProcessor.

    Loads autocorrelation hessians from ``query_path`` and ``index_path``,
    computes a mixing coefficient via the §A.1.3 procedure of Chang et al.
    (2024), and writes the mixed GradientProcessor to ``output_path``.
    """

    def execute(self):
        if not self.query_path or not self.index_path or not self.output_path:
            raise ValueError(
                "mix requires --query_path, --index_path, and --output_path to be set."
            )
        save_run_config(self, self.output_path)
        mix_autocorrelation_matrices(
            query_path=self.query_path,
            index_path=self.index_path,
            output_path=self.output_path,
            target_downweight_components=self.target_downweight_components,
        )


@dataclass
class Query(QueryConfig):
    """Query an existing gradient index."""

    def execute(self):
        """Query an existing gradient index."""
        query(self)


@dataclass
class Reduce(Serializable):
    """Reduce a gradient index."""

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    def execute(self):
        """Reduce a gradient index."""
        if self.index_cfg.sharded:
            raise ValueError(
                "reduce does not support sharded runs; per-shard aggregates "
                "would be concatenated instead of summed."
            )

        if self.index_cfg.projection_dim != 0:
            print(f"Using a projection dimension of {self.index_cfg.projection_dim}. ")

        validate_run_path(self.index_cfg)
        save_run_config(self, self.index_cfg.partial_run_path)
        build(self.index_cfg, self.preprocess_cfg)


@dataclass
class Score(Serializable):
    """Score a dataset against an existing gradient index."""

    score_cfg: ScoreConfig

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    def execute(self):
        """Score a dataset against an existing gradient index."""
        assert self.score_cfg.query_path

        if self.index_cfg.projection_dim != 0:
            print(f"Using a projection dimension of {self.index_cfg.projection_dim}. ")

        if self.index_cfg.sharded:
            if prepare_shard(self, self.index_cfg):
                return
        else:
            validate_run_path(self.index_cfg)

        save_run_config(self, self.index_cfg.partial_run_path)
        score_dataset(self.index_cfg, self.score_cfg, self.preprocess_cfg)


@dataclass
class Trackstar(Serializable):
    """Run hessians, build, and score as a single pipeline."""

    index_cfg: IndexConfig

    trackstar_cfg: TrackstarConfig

    def execute(self):
        from .trackstar import trackstar

        if self.index_cfg.sharded:
            raise ValueError(
                "trackstar does not support sharded runs yet; "
                "shard the build and score steps individually."
            )

        save_run_config(self, self.index_cfg.run_path)
        trackstar(self.index_cfg, self.trackstar_cfg)


@dataclass
class Train(TrainingConfig):
    """Train a model with the MAGIC trainer, but don't actually run MAGIC."""

    def execute(self):
        """Train the model."""
        run_magic(self)


@dataclass
class Status(Serializable):
    """Report the progress of a sharded run: which shards are published,
    in progress, or missing."""

    run_path: str = field(positional=True)

    def execute(self):
        """Print the shard inventory of a run path."""
        from ..sharding import shard_status

        published, partial, num_shards = shard_status(self.run_path)
        if num_shards is None:
            print(f"{self.run_path} is not a sharded run.")
            return

        missing = sorted(set(range(num_shards)) - published.keys() - partial.keys())
        print(f"{self.run_path}: {len(published)}/{num_shards} shards published")
        if partial:
            print(f"  in progress or crashed: {sorted(partial)}")
        if missing:
            print(f"  not started: {missing}")
        if not partial and not missing:
            print("  run complete")


@dataclass
class Test_Model_Configuration:
    """Test gradient consistency across padding and batch composition.

    Tests whether a model produces consistent gradients regardless of how
    documents are batched together. If inconsistencies are found, recommends
    using --force_math_sdp on build/score/trackstar commands."""

    diagnose_cfg: DiagnoseConfig

    def execute(self):
        """Run the diagnostic."""
        diagnose(self.diagnose_cfg)


@dataclass
class Validate(ValidationConfig):
    """Run leave-k-out validation of attribution scores."""

    scores: str = ""
    """Path to saved attribution scores for validation."""

    def execute(self):
        """Run the validation."""
        assert self.scores, "Path to attribution scores must be provided."
        run_magic(self, score_path=self.scores)
