import logging
import os
import sys
from dataclasses import dataclass
from typing import Union, get_args

from simple_parsing import ArgumentParser, ConflictResolution, Serializable

from bergson.hessians.pipeline import hessian_pipeline

from .build import build
from .config import (
    HessianConfig,
    HessianPipelineConfig,
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

# ignore httpx
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class Build(Serializable):
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
class Ekfac(Serializable):
    """Run the full EKFAC influence pipeline end-to-end."""

    index_cfg: IndexConfig

    hessian_cfg: HessianConfig

    score_cfg: ScoreConfig

    preprocess_cfg: PreprocessConfig

    hessian_pipeline_cfg: HessianPipelineConfig

    def execute(self):
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
        approximate_hessians(self.index_cfg, self.hessian_cfg)


@dataclass
class Magic(MagicConfig):
    """Run MAGIC attribution."""

    def execute(self):
        """Run MAGIC attribution."""
        run_magic(self)


@dataclass
class Preconditioners(IndexConfig):
    """Compute normalizers and preconditioners without gradient collection."""

    def execute(self):
        """Compute normalizers and preconditioners."""
        self.skip_index = True
        self.skip_preconditioners = False
        validate_run_path(self)
        build(self, PreprocessConfig())


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
        if self.index_cfg.projection_dim != 0:
            print(f"Using a projection dimension of {self.index_cfg.projection_dim}. ")

        validate_run_path(self.index_cfg)
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
        score_dataset(self.index_cfg, self.score_cfg, self.preprocess_cfg)


@dataclass
class Trackstar(Serializable):
    """Run preconditioners, build, and score as a single pipeline."""

    index_cfg: IndexConfig

    trackstar_cfg: TrackstarConfig

    def execute(self):
        trackstar(self.index_cfg, self.trackstar_cfg)


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
class Main:
    """Routes to the subcommands."""

    command: Union[
        Build,
        Ekfac,
        Hessian,
        Magic,
        Preconditioners,
        Query,
        Reduce,
        Score,
        Trackstar,
        Test_Model_Configuration,
    ]

    def execute(self):
        """Run the script."""
        self.command.execute()


def main():
    """Parse CLI arguments and dispatch to the selected subcommand."""
    # Check to see if the user is passing in a yaml or json config file directly
    args = sys.argv[1:]
    if len(args) == 2 and os.path.isfile(args[-1]):
        cmd_str, config_path = args

        args = get_args(Main.__dataclass_fields__["command"].type)
        names = {cls.__name__.lower(): cls for cls in args}
        try:
            cmd_cls = names[cmd_str.lower()]
        except KeyError:
            print(f"Invalid command '{cmd_str}'. Valid commands are: {list(names)}")
            sys.exit(1)

        try:
            prog = cmd_cls.load(config_path)
        except RuntimeError as e:
            print(f"Failed to load config file {config_path}: {e}")
            sys.exit(1)
    else:
        parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
        parser.add_arguments(Main, dest="prog")
        prog: Main = parser.parse_args().prog

    prog.execute()


if __name__ == "__main__":
    main()
