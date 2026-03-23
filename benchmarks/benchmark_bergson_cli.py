"""Benchmark Bergson using CLI subprocess calls (build, reduce, score)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from simple_parsing import ArgumentParser, ConflictResolution, field

from benchmarks.benchmark_utils import (
    MODEL_SPECS,
    get_hardware_info,
    get_run_path,
    parse_tokens,
    prepare_benchmark_ds_path,
    save_record,
    timestamp,
)
from bergson.config import IndexConfig

SCHEMA_VERSION = 1


@dataclass
class CLIRunRecord:
    """Record of a benchmark run using CLI subprocess calls."""

    schema_version: int
    status: str
    model_key: str
    model_name: str
    params: float
    train_tokens: int
    eval_tokens: int
    dataset: str
    batch_size: int
    build_seconds: float | None
    reduce_seconds: float | None
    score_seconds: float | None
    total_runtime_seconds: float | None
    start_time: str
    end_time: str
    run_path: str
    notes: str | None
    error: str | None
    num_gpus: int = 1
    hardware: str | None = None
    max_length: int | None = None
    token_batch_size: int | None = None
    projection_dim: int | None = None


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    model: str = field(positional=True)
    """Key for the model to benchmark (e.g., pythia-14m, pythia-70m)."""

    train_tokens: str = field(positional=True)
    """Target training tokens (e.g., 1M, 10M)."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    token_batch_size: int = 8192
    """Batch size in tokens for the bergson build command."""

    dataset: str = ""
    """Dataset to use for benchmarking."""

    run_path: str | None = None
    """Explicit run path (overrides auto-generated path)."""

    tag: str | None = None
    """Tag for the run (used in auto-generated path)."""

    notes: str | None = None
    """Optional notes for the run."""

    fsdp: bool = False
    """Enable FSDP (automatically enabled for models >= 1B parameters)."""

    auto_batch_size: bool = True
    """Automatically determine optimal token_batch_size for hardware."""

    num_gpus: int = 1
    """Number of GPUs to use for benchmarking."""

    skip_existing: bool = True
    """Skip benchmark if successful run exists for this model/token combo."""

    projection_dim: int = 16
    """Dimension to project gradients to. Matches bergson default."""


def run_cli_command(cmd: list[str], description: str) -> tuple[bool, float, str]:
    """Run a CLI command and return (success, elapsed_time, error_message)."""
    print(f"Running: {' '.join(cmd)}")
    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Changed to False so output is visible
            text=True,
        )
        elapsed = time.perf_counter() - start
        if result.returncode != 0:
            return (
                False,
                elapsed,
                f"{description} failed with return code {result.returncode}",
            )
        print(f"{description} completed in {elapsed:.2f}s")
        return True, elapsed, ""
    except Exception as e:
        elapsed = time.perf_counter() - start
        return False, elapsed, f"{description} error: {str(e)}"


def load_records(root: Path) -> list[CLIRunRecord]:
    """Load all benchmark records from a directory tree."""
    records: list[CLIRunRecord] = []
    for meta in root.rglob("benchmark_cli.json"):
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            records.append(CLIRunRecord(**payload))
        except Exception as exc:
            print(f"Warning: failed to read {meta}: {exc}", file=sys.stderr)
    return records


def find_existing_successful_run(
    run_root: Path,
    model_key: str,
    train_tokens: int,
    dataset: str,
    num_gpus: int,
) -> CLIRunRecord | None:
    """
    Check if a successful benchmark run already exists for the given configuration.

    Returns the existing record if found, None otherwise.
    """
    records = load_records(run_root)

    for record in records:
        if (
            record.status == "success"
            and record.model_key == model_key
            and record.train_tokens == train_tokens
            and record.dataset == dataset
            and record.num_gpus == num_gpus
        ):
            return record

    return None


@dataclass
class Run:
    """Execute a single Bergson CLI benchmark run."""

    run_cfg: RunConfig

    def execute(self) -> None:
        """Run the benchmark."""
        if not self.run_cfg.dataset:
            self.run_cfg.dataset = str(prepare_benchmark_ds_path())

        if self.run_cfg.model not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{self.run_cfg.model}'")

        eval_seqs = 1
        eval_tokens = 1024

        spec = MODEL_SPECS[self.run_cfg.model]
        train_tokens = parse_tokens(self.run_cfg.train_tokens)

        print(
            f"Running Bergson CLI benchmark for {self.run_cfg.model} with "
            f"{train_tokens} train tokens and {eval_seqs} eval sequences. "
        )

        run_root = Path(self.run_cfg.run_root).resolve()
        run_root.mkdir(parents=True, exist_ok=True)

        # Check if a successful run already exists
        if self.run_cfg.skip_existing:
            existing_run = find_existing_successful_run(
                run_root=run_root,
                model_key=spec.key,
                train_tokens=train_tokens,
                dataset=self.run_cfg.dataset,
                num_gpus=self.run_cfg.num_gpus,
            )

            if existing_run is not None:
                print(
                    f"⏭️  Skipping: Found existing successful run at "
                    f"{existing_run.run_path}."
                )
                print(
                    f"   Completed at {existing_run.end_time} "
                    f"(runtime: {existing_run.total_runtime_seconds:.1f}s)"
                )
                print("   Use --skip_existing=False to force re-run")
                return
        benchmark_path = (
            Path(self.run_cfg.run_path).resolve()
            if self.run_cfg.run_path
            else get_run_path(
                run_root,
                spec,
                train_tokens,
                eval_tokens,
                eval_seqs,
                self.run_cfg.tag,
                self.run_cfg.num_gpus,
            )
        )

        # Create directories for bergson artifacts
        index_path = benchmark_path / "index"
        score_path = benchmark_path / "score"

        # UNTIMED: Create and build 1-sequence query index
        query_index_path = benchmark_path / "query_index"
        query_dataset_path = benchmark_path / "query_dataset"

        # Create a 1-example dataset
        print("Creating 1-example query dataset (untimed)...")
        from datasets import Dataset

        query_dataset = Dataset.from_dict({"text": ["Hello, world!"]})
        query_dataset.save_to_disk(str(query_dataset_path))

        # Build query index from the 1-example dataset
        print("Building query index (untimed)...")
        query_cmd = [
            "bergson",
            "build",
            str(query_index_path),
            "--model",
            spec.hf_id,
            "--dataset",
            str(query_dataset_path),
            "--skip_preconditioners",
            "--overwrite",
            "--nproc_per_node",
            "1",
            "--auto_batch_size",
        ]

        success, _, err = run_cli_command(query_cmd, "Query index build")
        if not success:
            raise RuntimeError(f"Failed to build query index: {err}")

        # Read the determined batch size from the query index
        query_cfg = IndexConfig.load_json(query_index_path / "index_config.json")
        determined_batch_size = query_cfg.token_batch_size
        print(
            f"Using token_batch_size: {determined_batch_size}"
            " (determined before timing)"
        )

        # Common args for timed commands - use explicit batch size
        common_args = [
            "--model",
            spec.hf_id,
            "--dataset",
            self.run_cfg.dataset,
            "--split",
            "train",
            "--skip_preconditioners",
            "--overwrite",
            "--truncation",
            "--max_tokens",
            str(train_tokens),
            "--nproc_per_node",
            str(self.run_cfg.num_gpus),
            "--token_batch_size",
            str(determined_batch_size),
        ]

        if self.run_cfg.fsdp:
            common_args.append("--fsdp")

        start_wall = timestamp()
        start = time.perf_counter()
        status = "success"
        error_message: str | None = None
        build_time: float | None = None
        score_time: float | None = None

        try:
            # Step 1: Build the gradient index (timed)
            build_cmd = [
                "bergson",
                "build",
                str(index_path),
                *common_args,
            ]

            success, build_time, err = run_cli_command(build_cmd, "Build")
            if not success:
                raise RuntimeError(err)

            # Step 2: Score using 1-sequence query index (timed)
            score_cmd = [
                "bergson",
                "score",
                str(score_path),
                "--query_path",
                str(query_index_path),
                "--score",
                "mean",
                *common_args,
            ]

            success, score_time, err = run_cli_command(score_cmd, "Score")
            if not success:
                print(f"Warning: Score step failed: {err}")
                score_time = None

        except Exception as exc:
            status = "error"
            error_message = repr(exc)
            import traceback

            traceback.print_exc()

        runtime = time.perf_counter() - start
        end_wall = timestamp()

        # Load index config
        index_cfg = IndexConfig.load_json(index_path / "index_config.json")
        token_batch_size = index_cfg.token_batch_size

        record = CLIRunRecord(
            schema_version=SCHEMA_VERSION,
            status=status,
            model_key=spec.key,
            model_name=spec.hf_id,
            params=spec.params,
            train_tokens=train_tokens,
            eval_tokens=eval_seqs,
            dataset=self.run_cfg.dataset,
            batch_size=self.run_cfg.token_batch_size,
            build_seconds=build_time,
            # Reduce step removed - same runtime as score
            reduce_seconds=None,
            score_seconds=score_time,
            total_runtime_seconds=runtime,
            start_time=start_wall,
            end_time=end_wall,
            run_path=str(benchmark_path),
            notes=self.run_cfg.notes,
            error=error_message,
            num_gpus=self.run_cfg.num_gpus,
            hardware=get_hardware_info(),
            token_batch_size=token_batch_size,
            projection_dim=self.run_cfg.projection_dim,
        )
        save_record(benchmark_path, record, "benchmark_cli.json")

        print(json.dumps(asdict(record), indent=2))

        if status != "success":
            sys.exit(1)


@dataclass
class Main:
    """Benchmark Bergson CLI (build, reduce, score)."""

    command: Run

    def execute(self) -> None:
        """Run the selected command."""
        self.command.execute()


def get_parser() -> ArgumentParser:
    """Get the argument parser. Used for documentation generation."""
    parser = ArgumentParser(
        conflict_resolution=ConflictResolution.EXPLICIT,
        description="Benchmark Bergson CLI (build, reduce, score)",
    )
    parser.add_arguments(Main, dest="prog")
    return parser


def main(args: Optional[list[str]] = None) -> None:
    """Parse CLI arguments and dispatch to the selected subcommand."""
    parser = get_parser()
    prog: Main = parser.parse_args(args=args).prog
    prog.execute()


if __name__ == "__main__":
    main()
