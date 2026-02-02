import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from simple_parsing import ArgumentParser, ConflictResolution

from .build import build
from .config import IndexConfig, QueryConfig, ReduceConfig, ScoreConfig
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
class Main:
    """Routes to the subcommands."""

    command: Union[Build, Query, Reduce, Score]

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
