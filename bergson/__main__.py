from dataclasses import dataclass
from typing import Optional, Union

from simple_parsing import ArgumentParser, ConflictResolution

from .build import build
from .config import (
    DistributedConfig,
    HessianConfig,
    IndexConfig,
    PreprocessConfig,
    QueryConfig,
    ScoreConfig,
    TrackstarConfig,
)
from .diagnose import DiagnoseConfig, diagnose
from .hessians.hessian_approximations import approximate_hessians
from .magic import MagicConfig, run_magic
from .query.query_index import query
from .score.score import score_dataset
from .trackstar import trackstar
from .utils.worker_utils import validate_run_path


@dataclass
class Build:
    """Build a gradient index."""

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    def execute(self):
        """Build the gradient index."""
        if self.index_cfg.skip_index and self.index_cfg.skip_preconditioners:
            raise ValueError("Either skip_index or skip_preconditioners must be False")

        validate_run_path(self.index_cfg)

        build(self.index_cfg, self.preprocess_cfg)


@dataclass
class Preconditioners:
    """Compute normalizers and preconditioners without gradient collection."""

    index_cfg: IndexConfig

    def execute(self):
        """Compute normalizers and preconditioners."""
        self.index_cfg.skip_index = True
        self.index_cfg.skip_preconditioners = False
        validate_run_path(self.index_cfg)
        build(self.index_cfg, PreprocessConfig())


@dataclass
class Reduce:
    """Reduce a gradient index."""

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    def execute(self):
        """Reduce a gradient index."""
        if self.index_cfg.projection_dim != 0:
            print(
                f"Using a projection dimension of " f"{self.index_cfg.projection_dim}. "
            )

        validate_run_path(self.index_cfg)
        build(self.index_cfg, self.preprocess_cfg)


@dataclass
class Score:
    """Score a dataset against an existing gradient index."""

    score_cfg: ScoreConfig

    index_cfg: IndexConfig

    preprocess_cfg: PreprocessConfig

    def execute(self):
        """Score a dataset against an existing gradient index."""
        assert self.score_cfg.query_path

        if self.index_cfg.projection_dim != 0:
            print(
                f"Using a projection dimension of " f"{self.index_cfg.projection_dim}. "
            )

        validate_run_path(self.index_cfg)
        score_dataset(self.index_cfg, self.score_cfg, self.preprocess_cfg)


@dataclass
class Query:
    """Query an existing gradient index."""

    query_cfg: QueryConfig

    def execute(self):
        """Query an existing gradient index."""
        query(self.query_cfg)


@dataclass
class Hessian:
    """Approximate Hessian matrices using KFAC or EKFAC."""

    hessian_cfg: HessianConfig
    index_cfg: IndexConfig

    def execute(self):
        """Compute Hessian approximation."""
        validate_run_path(self.index_cfg)
        approximate_hessians(self.index_cfg, self.hessian_cfg)


@dataclass
class Trackstar:
    """Run preconditioners, build, and score as a single pipeline."""

    index_cfg: IndexConfig

    score_cfg: ScoreConfig

    preprocess_cfg: PreprocessConfig

    trackstar_cfg: TrackstarConfig

    def execute(self):
        trackstar(
            self.index_cfg, self.score_cfg, self.preprocess_cfg, self.trackstar_cfg
        )


@dataclass
class Magic:
    """Run MAGIC attribution."""

    run_cfg: MagicConfig
    dist_cfg: DistributedConfig

    def execute(self):
        """Run MAGIC attribution."""
        run_magic(self.run_cfg, self.dist_cfg)


@dataclass
class Test_Numerical_Stability:
    """Test gradient consistency across padding and batch composition.

    Tests whether a model produces consistent gradients regardless of how
    documents are batched together. If inconsistencies are found, recommends
    using --force_math_sdp on build/score/trackstar commands."""

    diagnose_cfg: DiagnoseConfig

    def execute(self):
        """Run the diagnostic."""
        diagnose(self.diagnose_cfg)


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[
        Build,
        Query,
        Preconditioners,
        Reduce,
        Score,
        Hessian,
        Trackstar,
        Magic,
        Test_Numerical_Stability,
    ]

    def execute(self):
        """Run the script."""
        self.command.execute()


def main(args: Optional[list[str]] = None):
    """Parse CLI arguments and dispatch to the selected subcommand."""
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()
