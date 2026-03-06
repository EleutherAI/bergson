import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from simple_parsing import ArgumentParser, ConflictResolution

from .build import build
from .config import (
    DistributedConfig,
    HessianConfig,
    IndexConfig,
    QueryConfig,
    ReduceConfig,
    ScoreConfig,
    TrackstarConfig,
)
from .double_backward import DoubleBackwardConfig, double_backward
from .hessians.hessian_approximations import approximate_hessians
from .query.query_index import query
from .reduce import reduce
from .score.score import score_dataset


def validate_run_path(index_cfg: IndexConfig):
    """Validate the run path."""
    if index_cfg.distributed.rank != 0:
        return

    for path in [Path(index_cfg.run_path), Path(index_cfg.partial_run_path)]:
        if not path.exists():
            continue

        if index_cfg.overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"Run path {path} already exists. Use --overwrite to overwrite it."
            )


@dataclass
class Build:
    """Build a gradient index."""

    index_cfg: IndexConfig

    def execute(self):
        """Build the gradient index."""
        if self.index_cfg.skip_index and self.index_cfg.skip_preconditioners:
            raise ValueError("Either skip_index or skip_preconditioners must be False")

        validate_run_path(self.index_cfg)

        build(self.index_cfg)


@dataclass
class Preconditioners:
    """Compute normalizers and preconditioners without gradient collection."""

    index_cfg: IndexConfig

    def execute(self):
        """Compute normalizers and preconditioners."""
        self.index_cfg.skip_index = True
        self.index_cfg.skip_preconditioners = False
        validate_run_path(self.index_cfg)
        build(self.index_cfg)


@dataclass
class Reduce:
    """Reduce a gradient index."""

    index_cfg: IndexConfig

    reduce_cfg: ReduceConfig

    def execute(self):
        """Reduce a gradient index."""
        if self.index_cfg.projection_dim != 0:
            print(
                f"Using a projection dimension of " f"{self.index_cfg.projection_dim}. "
            )

        validate_run_path(self.index_cfg)

        reduce(self.index_cfg, self.reduce_cfg)


@dataclass
class Score:
    """Score a dataset against an existing gradient index."""

    score_cfg: ScoreConfig

    index_cfg: IndexConfig

    def execute(self):
        """Score a dataset against an existing gradient index."""
        assert self.score_cfg.query_path

        if self.index_cfg.projection_dim != 0:
            print(
                f"Using a projection dimension of " f"{self.index_cfg.projection_dim}. "
            )

        validate_run_path(self.index_cfg)

        score_dataset(self.index_cfg, self.score_cfg)


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

    trackstar_cfg: TrackstarConfig

    score_cfg: ScoreConfig

    def execute(self):
        """Run the full trackstar pipeline: preconditioners -> build -> score."""
        run_path = self.index_cfg.run_path
        value_precond_path = f"{run_path}/value_preconditioner"
        query_precond_path = f"{run_path}/query_preconditioner"
        query_path = f"{run_path}/query"
        scores_path = f"{run_path}/scores"

        # Step 1: Compute normalizers and preconditioners on value dataset
        print("Step 1/4: Computing normalizers and preconditioners on value dataset...")
        value_precond_cfg = deepcopy(self.index_cfg)
        value_precond_cfg.run_path = value_precond_path
        value_precond_cfg.skip_index = True
        value_precond_cfg.skip_preconditioners = False
        validate_run_path(value_precond_cfg)
        build(value_precond_cfg)

        # Step 2: Compute normalizers and preconditioners on query dataset
        print("Step 2/4: Computing normalizers and preconditioners on query dataset...")
        query_precond_cfg = deepcopy(self.index_cfg)
        query_precond_cfg.run_path = query_precond_path
        query_precond_cfg.data = self.trackstar_cfg.query
        query_precond_cfg.skip_index = True
        query_precond_cfg.skip_preconditioners = False
        validate_run_path(query_precond_cfg)
        build(query_precond_cfg)

        # Step 3: Build per-item query gradient index
        print("Step 3/4: Building query gradient index...")
        query_cfg = deepcopy(self.index_cfg)
        query_cfg.run_path = query_path
        query_cfg.data = self.trackstar_cfg.query
        query_cfg.processor_path = query_precond_path
        query_cfg.skip_preconditioners = True
        validate_run_path(query_cfg)
        build(query_cfg)

        # Step 4: Score value dataset against query using both preconditioners
        print("Step 4/4: Scoring value dataset...")
        score_index_cfg = deepcopy(self.index_cfg)
        score_index_cfg.run_path = scores_path
        score_index_cfg.processor_path = value_precond_path
        score_index_cfg.skip_preconditioners = True
        self.score_cfg.query_path = query_path
        self.score_cfg.query_preconditioner_path = query_precond_path
        self.score_cfg.index_preconditioner_path = value_precond_path
        validate_run_path(score_index_cfg)
        score_dataset(score_index_cfg, self.score_cfg)


@dataclass
class Magic:
    """Run MAGIC attribution."""

    run_cfg: DoubleBackwardConfig
    dist_cfg: DistributedConfig

    def execute(self):
        """Run MAGIC attribution."""
        double_backward(self.run_cfg, self.dist_cfg)


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[
        Build, Query, Preconditioners, Reduce, Score, Hessian, Trackstar, Magic
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
