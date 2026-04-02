"""Benchmark Bergson influence analysis scaling (in-memory reduce + score)."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from simple_parsing import ArgumentParser, ConflictResolution, field

from benchmarks.benchmark_utils import (
    MODEL_SPECS,
    get_hardware_details,
    get_run_path,
    parse_tokens,
    prepare_benchmark_ds_path,
    save_record,
)
from bergson.collector.collector import CollectorComputer
from bergson.collector.gradient_collectors import GradientCollector
from bergson.collector.in_memory_collector import InMemoryCollector
from bergson.config import DataConfig, IndexConfig
from bergson.data import allocate_batches
from bergson.gradients import GradientProcessor
from bergson.score.score_writer import InMemorySequenceScoreWriter
from bergson.score.scorer import Scorer
from bergson.utils.auto_batch_size import determine_batch_size
from bergson.utils.worker_utils import (
    setup_data_pipeline,
    setup_model_and_peft,
)

SCHEMA_VERSION = 1
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "validation"


@dataclass
class RunRecord:
    """Record of an in-memory benchmark run."""

    schema_version: int
    status: str
    model_key: str
    model_name: str
    params: float
    train_tokens: int
    eval_tokens: int
    dataset: str
    train_split: str
    eval_split: str
    batch_size: int
    max_length: int
    # Time to collect query gradients
    query_seconds: float | None
    # Time to build the index
    build_seconds: float | None
    # Time to collect index gradients and compute inner products
    score_seconds: float | None
    run_path: str
    notes: str | None
    error: str | None
    num_gpus: int = 1
    token_batch_size: int | None = None
    projection_dim: int | None = None
    hardware: str | None = None
    gpu_name: str | None = None
    num_gpus_available: int | None = None
    gpu_vram_gb: float | None = None


@dataclass
class RunConfig:
    """Configuration for an in-memory benchmark run."""

    model: str = field(positional=True)
    """Key for the model to benchmark (e.g., pythia-14m, pythia-70m)."""

    train_tokens: str = field(positional=True)
    """Target training tokens (e.g., 1M, 10M)."""

    run_root: str = field(positional=True)
    """Root directory for benchmark runs."""

    eval_tokens: int = 1024
    """Target evaluation tokens per sequence. Not
    analogous to train_tokens."""

    eval_sequences: int = 1
    """Target evaluation sequences."""

    batch_size: int = 4
    """Batch size for gradient collection."""

    max_length: int = 1024
    """Maximum sequence length."""

    max_eval_examples: int = 10
    """Maximum number of evaluation examples to score."""

    dataset: str = ""
    """Dataset to use for benchmarking."""

    train_split: str = DEFAULT_TRAIN_SPLIT
    """Dataset split to use for training gradients."""

    eval_split: str = DEFAULT_EVAL_SPLIT
    """Dataset split to use for evaluation gradients."""

    auto_batch_size: bool = True
    """Automatically determine optimal token_batch_size for hardware."""

    token_batch_size: int | None = None
    """Override auto-tuned token_batch_size with a fixed value."""

    skip_preconditioners: bool = True
    """Skip preconditioners."""

    projection_dim: int = 16
    """Dimension to project gradients to. 0 = no projection (full gradients)."""

    run_path: str | None = None
    """Explicit run path (overrides auto-generated path)."""

    tag: str | None = None
    """Tag for the run (used in auto-generated path)."""

    skip_score: bool = False
    """Skip the in-memory query + score phase."""

    skip_build: bool = False
    """Skip the disk-based build phase."""

    notes: str | None = None
    """Optional notes for the run."""


def load_records(root: Path) -> list[RunRecord]:
    """Load all benchmark records from a directory tree."""
    records: list[RunRecord] = []
    for meta in root.rglob("benchmark.json"):
        try:
            with open(meta, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            records.append(RunRecord(**payload))
        except Exception as exc:
            print(f"Warning: failed to read {meta}: {exc}", file=sys.stderr)
    return records


def get_token_batch_size(model, index_cfg, eval_ds, projection_dim=16):
    """Determine optimal token batch size for the model.

    Args:
        projection_dim: Gradient projection dimension. 0 means no projection.
    """
    proj_dim = projection_dim if projection_dim > 0 else None

    optimal_token_batch_size = determine_batch_size(
        root=Path(".cache"),
        cfg=index_cfg,
        model=model,
        collector=InMemoryCollector(
            model=model.base_model,  # type: ignore
            processor=GradientProcessor(projection_dim=proj_dim),
            data=eval_ds,
            cfg=index_cfg,
        ),
    )
    return optimal_token_batch_size


@dataclass
class Run:
    """Execute a single in-memory Bergson benchmark run."""

    run_cfg: RunConfig

    def execute(self) -> None:
        """Run the benchmark."""
        precision = "bf16"

        if not self.run_cfg.dataset:
            self.run_cfg.dataset = str(prepare_benchmark_ds_path())

        if self.run_cfg.model not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{self.run_cfg.model}'")

        assert self.run_cfg.auto_batch_size or self.run_cfg.token_batch_size is not None
        assert self.run_cfg.eval_sequences == 1
        assert self.run_cfg.eval_tokens == 1024

        spec = MODEL_SPECS[self.run_cfg.model]
        train_tokens = parse_tokens(self.run_cfg.train_tokens)
        eval_tokens = self.run_cfg.eval_tokens
        eval_sequences = self.run_cfg.eval_sequences

        print(
            f"Running Bergson benchmark for {self.run_cfg.model} with "
            f"{train_tokens} train and {eval_tokens} eval tokens"
        )

        run_root = Path(self.run_cfg.run_root).resolve()
        run_root.mkdir(parents=True, exist_ok=True)
        run_path = (
            Path(self.run_cfg.run_path).resolve()
            if self.run_cfg.run_path
            else get_run_path(
                run_root,
                spec,
                train_tokens,
                eval_tokens,
                eval_sequences,
                self.run_cfg.tag,
            )
        )

        # Initialize timing variables
        status = "success"
        error_message: str | None = None
        build_time: float | None = None
        query_time: float | None = None
        score_time: float | None = None

        # Create configs for data loading
        index_cfg = IndexConfig(
            run_path=str(run_path),
            model=spec.hf_id,
            data=DataConfig(
                dataset=self.run_cfg.dataset,
                split=self.run_cfg.train_split,
                prompt_column="text",
            ),
            token_batch_size=self.run_cfg.max_length,
            max_tokens=train_tokens,
            precision=precision,
            skip_preconditioners=self.run_cfg.skip_preconditioners,
        )

        model, _ = setup_model_and_peft(index_cfg, device_map_auto=True)

        eval_ds = Dataset.from_dict(
            {
                "input_ids": [list(range(eval_tokens))],
                "attention_mask": [list(range(eval_tokens))],
                "length": [eval_tokens],
            }
        )

        # Get batch size BEFORE timing (this is setup overhead, not benchmark time)
        # Convert projection_dim=0 to None (meaning no projection)
        proj_dim = (
            self.run_cfg.projection_dim if self.run_cfg.projection_dim > 0 else None
        )
        print(f"Projection dim: {proj_dim}")
        if self.run_cfg.token_batch_size is not None:
            optimal_token_batch_size = self.run_cfg.token_batch_size
            print(f"Using manual token_batch_size:" f" {optimal_token_batch_size}")
        else:
            print("Determining optimal batch size" " (not timed)...")
            optimal_token_batch_size = get_token_batch_size(
                model,
                index_cfg,
                eval_ds,
                self.run_cfg.projection_dim,
            )
            print(f"Using auto-tuned batch size:" f" {optimal_token_batch_size}")
        index_cfg.token_batch_size = optimal_token_batch_size

        ds, _ = setup_data_pipeline(index_cfg)
        batches = allocate_batches(
            ds["length"][:], optimal_token_batch_size  # type: ignore
        )

        # In-memory query + score phase
        if not self.run_cfg.skip_score:
            query_collector = InMemoryCollector(
                model=model.base_model,  # type: ignore
                processor=GradientProcessor(projection_dim=proj_dim),
                data=eval_ds,
                cfg=index_cfg,
            )
            query_batches = allocate_batches(
                eval_ds["length"][:], optimal_token_batch_size
            )
            query_computer = CollectorComputer(
                model=model,
                data=eval_ds,
                collector=query_collector,
                batches=query_batches,
                cfg=index_cfg,
            )

            print("Collecting query gradients...")
            query_start = time.perf_counter()
            query_computer.run_with_collector_hooks(desc="query gradients")
            query_time = time.perf_counter() - query_start
            print(f"Query phase completed in " f"{query_time:.2f} seconds")

            query_grads = query_collector.gradients
            modules = list(query_grads.keys())
            num_queries = len(query_grads[modules[0]])
            writer = InMemorySequenceScoreWriter(
                len(ds), num_queries, dtype=torch.bfloat16
            )
            scorer = Scorer(
                query_grads=query_grads,
                modules=modules,
                writer=writer,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            score_collector = InMemoryCollector(
                model=model.base_model,  # type: ignore
                processor=GradientProcessor(projection_dim=proj_dim),
                data=ds,
                cfg=index_cfg,
                scorer=scorer,
            )
            score_computer = CollectorComputer(
                model=model,
                data=ds,
                collector=score_collector,
                batches=batches,
                cfg=index_cfg,
            )

            print("Collecting training gradients...")
            score_start = time.perf_counter()
            score_computer.run_with_collector_hooks(desc="training gradients")
            score_time = time.perf_counter() - score_start
            print(f"Score phase completed in " f"{score_time:.2f} seconds")

        # Build phase: Save gradients to disk
        if not self.run_cfg.skip_build:
            print("Building index...")
            grad_collector = GradientCollector(
                model=model.base_model,  # type: ignore
                processor=GradientProcessor(projection_dim=proj_dim),
                data=ds,
                cfg=index_cfg,
            )
            build_computer = CollectorComputer(
                model=model,
                data=ds,
                collector=grad_collector,
                batches=batches,
                cfg=index_cfg,
            )
            build_start = time.perf_counter()
            build_computer.run_with_collector_hooks(desc="building index")
            build_time = time.perf_counter() - build_start
            print(f"Build phase completed in " f"{build_time:.2f} seconds")

        record = RunRecord(
            schema_version=SCHEMA_VERSION,
            status=status,
            model_key=spec.key,
            model_name=spec.hf_id,
            params=spec.params,
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            dataset=self.run_cfg.dataset,
            train_split=self.run_cfg.train_split,
            eval_split=self.run_cfg.eval_split,
            batch_size=self.run_cfg.batch_size,
            max_length=self.run_cfg.max_length,
            num_gpus=1,
            build_seconds=build_time,
            query_seconds=query_time,
            score_seconds=score_time,
            run_path=str(run_path),
            notes=self.run_cfg.notes,
            error=error_message,
            token_batch_size=optimal_token_batch_size,
            projection_dim=self.run_cfg.projection_dim,
            **vars(get_hardware_details()),
        )
        save_record(run_path, record)

        print(json.dumps(asdict(record), indent=2))

        if status != "success":
            sys.exit(1)


@dataclass
class Main:
    """Benchmark Bergson influence analysis scaling."""

    command: Run

    def execute(self) -> None:
        """Run the selected command."""
        self.command.execute()


def get_parser() -> ArgumentParser:
    """Get the argument parser. Used for documentation generation."""
    parser = ArgumentParser(
        conflict_resolution=ConflictResolution.EXPLICIT,
        description="Benchmark Bergson influence analysis scaling (in-memory)",
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
