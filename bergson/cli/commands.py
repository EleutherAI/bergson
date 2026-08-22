"""CLI command definitions.

Each command is a thin dataclass that validates, persists its ``config.yaml``,
and dispatches. They live here rather than in ``__main__`` so pipelines can
import lower-level commands for config serialization without importing the
corresponding CLI entrypoint.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from simple_parsing import Serializable
from simple_parsing.helpers.serialization import register_decoding_fn

from ..build import build
from ..config.config import (
    ApproxUnrollingConfig,
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    MetasmoothnessConfig,
    MixConfig,
    PreprocessConfig,
    QueryConfig,
    RecallConfig,
    ScoreConfig,
    TrackstarConfig,
    TrackstarIndexConfig,
    TrainingConfig,
    ValidationConfig,
)
from ..config.config_io import save_run_config
from ..diagnose import DiagnoseConfig, diagnose
from ..hessians.hessian_approximations import approximate_hessians
from ..magic import MagicConfig, run_magic
from ..magic.metasmoothness import run_metasmoothness
from ..process_grads import mix_autocorrelation_matrices
from ..query.query_index import query
from ..recall.recall import run_recall
from ..score.score import score_dataset
from ..utils.worker_utils import validate_run_path
from ..validate import evaluate_retrained

register_decoding_fn(cast(type, str | list[str]), lambda v: v)


@dataclass
class ApproxUnrolling(Serializable):
    """Run the SOURCE (approximate unrolling) training-data attribution pipeline.

    Runs from per-checkpoint Hessian precompute through per-segment scoring and
    aggregation into ``<run>/scores/``. See
    :mod:`bergson.approx_unrolling.pipeline` for the step list.
    """

    index_cfg: IndexConfig

    hessian_cfg: HessianConfig

    approx_unrolling_cfg: ApproxUnrollingConfig

    def execute(self):
        from ..approx_unrolling.pipeline import approx_unrolling_pipeline

        save_run_config(self, self.index_cfg.run_path)
        approx_unrolling_pipeline(
            self.index_cfg,
            self.hessian_cfg,
            self.approx_unrolling_cfg,
        )


@dataclass
class Build(Serializable):
    """Build a gradient index."""

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    def execute(self):
        """Build the gradient index."""
        validate_run_path(self.index_cfg)

        save_run_config(self, self.index_cfg.partial_run_path)
        build(self.index_cfg, self.preprocess_cfg)


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
        validate_run_path(self.index_cfg)
        save_run_config(self, self.index_cfg.partial_run_path)
        approximate_hessians(self.index_cfg, self.hessian_cfg)


@dataclass
class Magic(MagicConfig):
    """Run MAGIC attribution."""

    def execute(self):
        """Run MAGIC attribution, then validate the scores unless skipped."""
        run_magic(self)
        if self.skip_validation:
            return

        run_path = Path(self.run_path)
        base = run_path / "retrained" / "base"
        shared = {
            k: v
            for k, v in self.to_dict().items()
            if k in ValidationConfig.__dataclass_fields__
        }
        Validate.from_dict(
            shared
            | {
                "scores": str(run_path / "scores"),
                "baseline_model": str(base) if base.exists() else "",
                "resume": True,
            }
        ).execute()
        # The validation step wrote its own config over this run's.
        save_run_config(self, run_path)


@dataclass
class Metasmoothness(MetasmoothnessConfig):
    """Estimate empirical metasmoothness (arXiv 2503.13751, Def. 2) of the
    MAGIC training configuration via three perturbed-data-weight trainings."""

    def execute(self):
        save_run_config(self, self.run_path)
        run_metasmoothness(self)


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
class Recall(RecallConfig):
    """Evaluate attribution scores by synthetic factual recall (MRR, Recall@k)."""

    def execute(self):
        """Run the recall evaluation."""
        assert self.scores, "Path to attribution scores must be provided."
        save_run_config(self, self.run_path)
        run_recall(self)


@dataclass
class Reduce(Serializable):
    """Reduce a gradient index."""

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    def execute(self):
        """Reduce a gradient index."""
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

        validate_run_path(self.index_cfg)
        save_run_config(self, self.index_cfg.partial_run_path)
        score_dataset(self.index_cfg, self.score_cfg, self.preprocess_cfg)


@dataclass
class Trackstar(Serializable):
    """Run hessians, build, and score as a single pipeline."""

    # Trackstar uses random-projection compression, so override the default
    # projection_dim of 0.
    index_cfg: TrackstarIndexConfig

    trackstar_cfg: TrackstarConfig

    def execute(self):
        from .trackstar import trackstar

        save_run_config(self, self.index_cfg.run_path)
        trackstar(self.index_cfg, self.trackstar_cfg)


@dataclass
class Train(TrainingConfig):
    """Train a model."""

    def execute(self):
        """Train the model."""
        run_magic(self)


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

    retrained_dir: str | list[str] = ""
    """Optional: evaluate on existing run directories of leave-k-out re-trained
    models written with ``save_models=true``; multiple directories (a yaml list,
    or comma-separated on the CLI) average the query losses over the runs."""

    baseline_model: str = ""
    """Optional path to baseline model trained on the full dataset."""

    @property
    def retrained_dirs(self) -> list[str]:
        """``retrained_dir`` normalized to a list of paths."""
        if isinstance(self.retrained_dir, str):
            return [d for d in self.retrained_dir.split(",") if d]
        return list(self.retrained_dir)

    def execute(self):
        """Run the validation."""
        assert self.scores, "Path to attribution scores must be provided."
        dirs = self.retrained_dirs
        # A filter-* run still retrains its score-dependent arm, so a bank
        # replaces the whole run only for lds.
        if dirs and self.method == "lds":
            evaluate_retrained(self, dirs, score_path=self.scores)
        else:
            run_magic(
                self,
                score_path=self.scores,
                validate=True,
                baseline_model=self.baseline_model,
                retrained_dir=dirs,
            )
