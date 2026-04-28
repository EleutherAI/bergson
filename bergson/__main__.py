import os
import sys
from dataclasses import dataclass, field
from typing import Union, get_args

from simple_parsing import ArgumentParser, ConflictResolution, Serializable

from bergson.hessians.pipeline import hessian_pipeline

from .build import build
from .config import (
    DistributedConfig,
    HessianConfig,
    HessianPipelineConfig,
    IndexConfig,
    PreprocessConfig,
    QueryConfig,
    ScoreConfig,
    TrackstarConfig,
)
from .diagnose import DiagnoseConfig, diagnose
from .distributed import launch_distributed_run
from .hessians.apply_hessian import EkfacConfig, apply_worker
from .hessians.hessian_approximations import approximate_hessians
from .magic import MagicConfig, run_magic
from .query.query_index import query
from .score.score import score_dataset
from .trackstar import trackstar
from .utils.worker_utils import validate_run_path
from .yaml_pipeline import run_pipeline


@dataclass
class Apply_Hessian(Serializable):
    """Apply a precomputed inverse Hessian to a saved query gradient.

    Reads the Hessian factors written by `bergson hessian` and the mean
    query gradient written by `bergson build` (with `aggregation: mean`),
    and writes the inverse-Hessian-vector product (transformed query) to
    `ekfac_cfg.run_path`. Used as the middle step of the EKFAC influence
    flow, between building the query gradient and scoring training data
    against it.
    """

    ekfac_cfg: EkfacConfig

    distributed_cfg: DistributedConfig = field(default_factory=DistributedConfig)

    def execute(self):
        launch_distributed_run(
            "apply_hessian",
            apply_worker,
            [self.ekfac_cfg],
            self.distributed_cfg,
        )


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
        Apply_Hessian,
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
    """Parse CLI arguments and dispatch to the selected subcommand.

    Three input shapes are supported:
      1. `bergson pipeline <file.yaml>`        — multi-step pipeline mode
      2. `bergson <command> <file.yaml|json>`  — single-command config-file mode
      3. `bergson <command> --flag value ...`  — single-command CLI-flag mode
    """
    args = sys.argv[1:]

    # Build the {command_name: command_class} lookup once; both the pipeline
    # branch and the single-command branch use it to resolve the user's verb.
    command_classes = get_args(Main.__dataclass_fields__["command"].type)
    command_registry = {cls.__name__.lower(): cls for cls in command_classes}

    # Branch 1 — Pipeline mode: a YAML listing several commands to run in
    # sequence. Delegates parsing + execution to `run_pipeline`.
    if len(args) == 2 and args[0].lower() == "pipeline" and os.path.isfile(args[-1]):
        run_pipeline(args[1], command_registry)
        return

    # Branch 2 — Single-command config-file mode: the user passes a command
    # name and a path to a YAML or JSON file. `Serializable.load` sniffs the
    # file extension and hydrates the matching dataclass.
    if len(args) == 2 and os.path.isfile(args[-1]):
        cmd_str, config_path = args
        try:
            cmd_cls = command_registry[cmd_str.lower()]
        except KeyError:
            print(
                f"Invalid command '{cmd_str}'. "
                f"Valid commands are: {list(command_registry)}"
            )
            sys.exit(1)

        try:
            prog = cmd_cls.load(config_path)
        except RuntimeError as e:
            print(f"Failed to load config file {config_path}: {e}")
            sys.exit(1)
    # Branch 3 — CLI-flag mode: standard argparse-style flag parsing.
    else:
        parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
        parser.add_arguments(Main, dest="prog")
        prog: Main = parser.parse_args().prog

    prog.execute()


if __name__ == "__main__":
    main()
