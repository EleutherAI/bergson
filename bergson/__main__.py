import os
import sys
from dataclasses import dataclass
from typing import Union, get_args

from simple_parsing import ArgumentParser, ConflictResolution

from bergson.config.config_io import parse_steps, read_config, save_pipeline_config

from .cli.commands import (
    ApproxUnrolling,
    Build,
    Ekfac,
    Hessian,
    Magic,
    Metasmoothness,
    Mix,
    Query,
    Recall,
    Reduce,
    Score,
    Test_Model_Configuration,
    Trackstar,
    Train,
    Validate,
)


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[
        ApproxUnrolling,
        Build,
        Ekfac,
        Hessian,
        Magic,
        Metasmoothness,
        Mix,
        Query,
        Recall,
        Reduce,
        Score,
        Trackstar,
        Train,
        Test_Model_Configuration,
        Validate,
    ]

    def execute(self):
        """Run the script."""
        self.command.execute()


def run_config(config_path: str, command_registry: dict[str, type]) -> None:
    """Execute each step of a bergson config YAML in order.

    A fully resolved version of any multi-step config (including default values)
    is written to ``run_path`` (auto-named under ``runs/`` if not given).
    Each step also writes its own component ``config.yaml`` into its run directory.
    """
    config = read_config(config_path)

    steps = parse_steps(config["steps"], command_registry)

    multi = len(steps) > 1

    if multi:
        # Optional top-level run path for a multi-step pipeline
        run_path = config.get("run_path")

        save_pipeline_config(steps, run_path)

    for i, (cmd_name, cmd) in enumerate(steps, start=1):
        if multi:
            print(f"\n[Pipeline] Step {i}/{len(steps)}: {cmd_name.title()}")
        cmd.execute()


def main():
    """Parse CLI arguments and dispatch to the selected subcommand.

    Two input shapes are supported:
      `bergson <command> --flag value ...`  — CLI-flag mode
      `bergson <file.yaml>`                 — run a config file: a mapping with a
            ``steps:`` list of ``- command: {...}`` entries, run in sequence (a
            single run is a one-step list).

    Every run writes such a ``config.yaml`` into its run
    directory, so a completed run can be replayed with
    `bergson <run_dir>/config.yaml`.
    """
    args = sys.argv[1:]

    command_classes = get_args(Main.__dataclass_fields__["command"].type)
    command_registry = {cls.__name__.lower(): cls for cls in command_classes}

    # Config-file mode: accept a YAML file as the sole argument.
    # Leading command words (e.g. `bergson build run/config.yaml`) are ignored.
    config_path: str | None = None
    if len(args) == 1:
        config_path = args[0]
    elif len(args) == 2:
        config_path = args[1]

    looks_like_config_file = (
        config_path.endswith((".yaml", ".yml")) or os.sep in config_path
    )

    if config_path is not None and looks_like_config_file:
        if not os.path.isfile(config_path):
            raise SystemExit(f"bergson: no such config file: {config_path}")

        run_config(config_path, command_registry)
        return

    # CLI-flag mode: argparse-style flag parsing.
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    prog: Main = parser.parse_args().prog
    prog.execute()


if __name__ == "__main__":
    main()
